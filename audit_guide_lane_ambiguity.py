"""Audit guide/exit-lane ambiguity using independent physical lane geometry.

No training/inference. Equal road cross-sections, tolerance and horizon coverage
are explicit. Full-route audit is separate from actual 81-point input audit.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from Dataset.audit_lane_visibility import sample_at
from diagnose_paths import to_world, target_lane_polygon
from evaluate import EvaluationDataset
from train import inspect_split
from World.intersection_generator_v4 import arm_xy
from World.road_warp import ArmWarp, deform


def geometry(world, route):
    md = world['metadata']; k = int(route['exit_arm'])
    s = np.arange(md['entry_distances_m'][k], md['arm_length_m'], .2)
    s = np.r_[s, md['arm_length_m']]
    w = md['arm_warp']
    warp = ArmWarp(tuple(w['angle_deg']), tuple(w['lateral_m']), w['start_m'], w['full_m'])
    target = int(route['exit_lane_index']); count = int(md['num_out'][k])
    reference = deform(arm_xy(k, s, -(target+.5)*md['lane_width_m']), warp)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=-1))]
    polygons = [target_lane_polygon(world, {**route, 'exit_lane_index':lane}) for lane in range(count)]
    return dict(s=s, reference=reference, arc=arc, warp=warp, width=md['lane_width_m'],
                arm=k, target=target, count=count, polygons=polygons)


def road_distances(points, geo):
    """Project to target reference, compare all lanes at the SAME base station."""
    a = geo['reference'][:-1]; v = np.diff(geo['reference'],axis=0)
    u = np.clip(((points[:,None]-a)*v).sum(-1)/np.maximum((v*v).sum(-1),1e-12),0,1)
    projections = a+u[...,None]*v
    j = ((points[:,None]-projections)**2).sum(-1).argmin(-1)
    u = u[np.arange(len(points)),j]
    s = geo['s'][j]+u*(geo['s'][j+1]-geo['s'][j])
    arc = geo['arc'][j]+u*(geo['arc'][j+1]-geo['arc'][j])
    centers = np.stack([deform(arm_xy(geo['arm'],s,-(lane+.5)*geo['width']),geo['warp'])
                        for lane in range(geo['count'])],axis=1)
    return np.linalg.norm(points[:,None]-centers,axis=-1), arc


def classify_margin(margin, tolerance, clear_fraction=.8):
    target = float(np.mean(margin>tolerance))
    neighbor = float(np.mean(margin < -tolerance))
    tied = float(np.mean(np.abs(margin)<=tolerance))
    group = ('target_clear' if target>=clear_fraction else 'neighbor_clear' if neighbor>=clear_fraction
             else 'ambiguous_or_mixed')
    detail = 'near_tie' if tied>=clear_fraction else 'mixed' if group=='ambiguous_or_mixed' else group
    return group,detail,target,neighbor,tied


def exit_window(points, geo, args, arc):
    # Other arms can project onto a distant curved exit centerline. Restrict to
    # the path suffix after its closest approach to the physical exit center.
    # This works for complete routes and horizons that already start after exit.
    after_exit = np.arange(len(points)) >= np.linalg.norm(points-geo['reference'][0],axis=-1).argmin()
    return after_exit & (arc>=args.start_m) & (arc<=args.end_m)


def audit_guide(points, geo, args):
    distance, arc = road_distances(points,geo)
    mask = exit_window(points,geo,args,arc)
    n = int(mask.sum())
    result = {'checked_points':n, 'checked_span_m':float(np.ptp(arc[mask])) if n else 0.,
              'target_lane':geo['target'], 'exit_lanes':geo['count']}
    if n<args.min_points:
        return {**result,'group':'insufficient_horizon','detail':'insufficient_horizon'},np.array([])
    d = distance[mask]; target=d[:,geo['target']]
    result.update(target_distance_mean_m=float(target.mean()),target_distance_median_m=float(np.median(target)))
    if geo['count']==1:
        return {**result,'group':'single_lane','detail':'single_lane'},np.array([])
    # Include every alternative outgoing lane, not only the immediately adjacent one.
    neighbor=d[:,np.arange(geo['count'])!=geo['target']].min(-1)
    margin=neighbor-target  # positive means target is closer
    group,detail,ft,fn,fa=classify_margin(margin,args.tolerance_m,args.clear_fraction)
    result.update(group=group,detail=detail,target_clear_fraction=ft,neighbor_clear_fraction=fn,
                  near_tie_fraction=fa,margin_mean_m=float(margin.mean()),margin_median_m=float(np.median(margin)),
                  neighbor_distance_mean_m=float(neighbor.mean()))
    return result,margin


def prediction_lane(points,geo,args):
    _,arc=road_distances(points,geo)
    pts=points[exit_window(points,geo,args,arc)]
    if len(pts)<args.min_points:
        return None,None,None
    inside=np.array([[polygon.covers(Point(p)) for polygon in geo['polygons']] for p in pts])
    target=float(inside[:,geo['target']].mean())
    neighbor=float(inside[:,np.arange(geo['count'])!=geo['target']].any(-1).mean())
    outside=float((~inside.any(-1)).mean())
    return neighbor>=args.clear_fraction,target,outside


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def summarize(rows):
    result={'samples':len(rows),'worlds':len({r['world_id'] for r in rows})}
    for key in ('original_ade_m','expanded_ade_m','guide_ade_m','expanded_wrong_direction_rate',
                'original_neighbor_lane_choice','expanded_neighbor_lane_choice_rate','expanded_outside_fraction',
                'expanded_target_fraction'):
        values=[r[key] for r in rows if r.get(key) is not None]
        if values:
            result[key]=float(np.mean(values))
            if key in ('original_neighbor_lane_choice','expanded_neighbor_lane_choice_rate'):
                result[key+'_eligible_samples']=len(values)
    return result


def run(args):
    comparison=json.loads((args.evaluation_dir/'summary.json').read_text())
    source=json.loads((args.source_dir/'summary.json').read_text())
    if inspect_split(source['input'])!=comparison['data_manifest']:
        raise ValueError('Input fingerprint mismatch')
    base=EvaluationDataset(source['input'],32,32)
    with np.load(args.evaluation_dir/'predictions.npz',allow_pickle=False) as archive:
        ids=archive['ids'].tolist()
        pred={k:archive[k] for k in archive.files if k.startswith('mixed_')}
    if ids!=[base.raw_row(i)['id'] for i in range(len(base))]:
        raise ValueError('Prediction identity mismatch')
    with (args.evaluation_dir/'per_sample.csv').open() as f:
        errors={(int(r['index']),r['input']):r for r in csv.DictReader(f) if r['model']=='mixed'}
    with (args.evaluation_dir/'failure_analysis/per_sample.csv').open() as f:
        dominant={int(r['index']):r for r in csv.DictReader(f)}
    expanded=[k for k in pred if k.startswith('mixed_expanded_seed_')]
    cache,route_audit,rows,margins={},{},[],{}
    for i in range(len(base)):
        row=base.raw_row(i);wid=row['world_id'];nav=row['navigation']
        key=(wid,nav)
        if wid not in cache:
            world=json.loads((args.worlds_dir/f'{wid}.json').read_text())
            routes=json.loads((args.routes_dir/f'{wid}.json').read_text())
            cache[wid]=(world,routes,{})
        world,routes,geos=cache[wid];route=routes[nav]
        if nav not in geos:geos[nav]=geometry(world,route)
        geo=geos[nav]
        if key not in route_audit:
            full=np.asarray(route['guide_world']);length=np.linalg.norm(np.diff(full,axis=0),axis=-1).sum()
            full=sample_at(full,np.arange(0,length,1.))
            result,_=audit_guide(full,geo,args)
            route_audit[key]={'world_id':wid,'navigation':nav,**result,
                'guide_offset_m':route['guide_parameters']['lateral_offset_m'],
                'entry_lane':route['entry_lane_index']}
        result,margin=audit_guide(to_world(np.asarray(row['reference_line']),row['metadata']),geo,args)
        margins[i]=margin
        result={'index':i,'id':row['id'],'world_id':wid,'navigation':nav,
                'pose':row['metadata']['pose_name'],'quality':row['metadata']['quality'],**result}
        gt=np.asarray(row['gt_path']);guide=np.asarray(row['reference_line'])
        result.update(guide_ade_m=float(np.linalg.norm(guide[1:]-gt[1:],axis=-1).mean()),
                      original_ade_m=float(errors[i,'original']['ade_m']),
                      expanded_ade_m=float(np.mean([float(errors[i,k.removeprefix('mixed_')]['ade_m']) for k in expanded])),
                      expanded_wrong_direction_rate=float(np.mean([errors[i,k.removeprefix('mixed_')]['wrong_direction']=='True' for k in expanded])),
                      lateral_dominant=dominant[i]['error_dominant'])
        choices=[prediction_lane(to_world(pred[k][i],row['metadata']),geo,args) for k in expanded]
        valid=[x for x in choices if x[0] is not None]
        original=prediction_lane(to_world(pred['mixed_original'][i],row['metadata']),geo,args)
        result.update(original_neighbor_lane_choice=original[0],
                      expanded_neighbor_lane_choice_rate=float(np.mean([x[0] for x in valid])) if valid else None,
                      expanded_target_fraction=float(np.mean([x[1] for x in valid])) if valid else None,
                      expanded_outside_fraction=float(np.mean([x[2] for x in valid])) if valid else None)
        rows.append(result)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'per_sample.csv',rows);write_csv(out/'per_route.csv',list(route_audit.values()))
    groups={group:summarize([r for r in rows if r['group']==group]) for group in sorted({r['group'] for r in rows})}
    focus=[r for r in rows if r['pose'] in ('entry','approach_10')]
    focused={group:summarize([r for r in focus if r['group']==group]) for group in sorted({r['group'] for r in focus})}
    bad=[r for r in focus if r['expanded_ade_m']>1]
    wrong=[r for r in bad if r['lateral_dominant']=='wrong_direction']
    sensitivity={}
    for tolerance in (.1,.3,.5):
        counts=Counter(classify_margin(m,tolerance,args.clear_fraction)[0] for m in margins.values() if len(m))
        sensitivity[str(tolerance)]=dict(counts)
    summary={'config':vars(args)|{'evaluation_dir':str(args.evaluation_dir),'source_dir':str(args.source_dir),
        'worlds_dir':str(args.worlds_dir),'routes_dir':str(args.routes_dir),'output_dir':str(args.output_dir)},
        'data_manifest':comparison['data_manifest'],'groups':groups,'focus_groups':focused,
        'route_groups':dict(Counter(r['group'] for r in route_audit.values())),
        'focus_high_error_groups':dict(Counter(r['group'] for r in bad)),
        'focus_high_error_wrong_direction_groups':dict(Counter(r['group'] for r in wrong)),
        'tolerance_sensitivity':sensitivity,
        'notes':['Distances use independently reconstructed warped lane centers at identical road cross-sections.',
          'Window is 10-40 m along physical target center from exit; only actual input guide points are audited per sample.',
          'Only path suffix after closest approach to physical exit center is considered, avoiding unrelated incoming-arm projections.',
          'Full-route audit is separate and not supplied to the model. Near-tie/mixed groups do not prove non-identifiability.',
          'Prediction lane choice requires >=80% checked points inside another outgoing lane; outside road is not counted as another lane.',
          'Repeated qualities/poses/seeds are not independent worlds. Associations are descriptive, not causal.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for group,color in [('target_clear','#32975a'),('ambiguous_or_mixed','#e8a126'),('neighbor_clear','#c54545')]:
        rs=[r for r in rows if r['group']==group]
        ax.scatter([r['margin_median_m'] for r in rs],[r['expanded_ade_m'] for r in rs],s=9,alpha=.35,c=color,label=f'{group} ({len(rs)})')
    ax.axvspan(-args.tolerance_m,args.tolerance_m,color='gray',alpha=.1)
    ax.set(xlabel='Median distance advantage of target over alternative lane (m)',ylabel='Mixed / expanded ADE (m)',
           title='Guide lane geometry vs prediction error (repeated poses/qualities)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'association.png',dpi=140);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config','data_manifest','notes')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-dir',type=Path,default=Path('outputs/context_comparison/evaluation'))
    p.add_argument('--source-dir',type=Path,default=Path('outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A'))
    p.add_argument('--worlds-dir',type=Path,default=Path('outputs/v4_world300_fixedguide/worlds'))
    p.add_argument('--routes-dir',type=Path,default=Path('outputs/v4_world300_fixedguide/routes'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/context_comparison/guide_lane_audit'))
    p.add_argument('--start-m',type=float,default=10.)
    p.add_argument('--end-m',type=float,default=40.)
    p.add_argument('--tolerance-m',type=float,default=.3)
    p.add_argument('--clear-fraction',type=float,default=.8)
    p.add_argument('--min-points',type=int,default=5)
    args=p.parse_args()
    if not (0<=args.start_m<args.end_m and args.tolerance_m>0 and .5<args.clear_fraction<=1 and args.min_points>=2):
        p.error('Invalid audit thresholds')
    run(args)
