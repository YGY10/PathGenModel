"""Frozen-world, multi-Guide validation benchmark. No training or World regeneration.

Examples (from repository root):
  python scripts/evaluate_guide_conflicts.py --limit-worlds 2 --qualities clean --poses entry
  python scripts/evaluate_guide_conflicts.py --device cpu
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import csv
import numpy as np
import torch
import shapely
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dataset.dataset import encode_sample
from Dataset.audit_drivable import drivable_polygon
from Dataset.audit_lane_visibility import sample_at
from Scenario.build_dataset import arclength
from Scenario.build_dataset_v4 import route_base, sample_world_guide
from World.road_warp import ArmWarp, deform
from World.intersection_generator_v4 import OUT
from diagnose_paths import to_world, tangent, target_lane_polygon
from evaluate import load_model
from train import inspect_split


def variants():
    """Original + one-factor sweeps + four joint extremes; positive timing=later."""
    return ([{'name':'original','timing_m':0.,'lateral_m':0.,'family':'original'}]
        + [{'name':f'timing_{v:+g}m','timing_m':v,'lateral_m':0.,'family':'timing'} for v in (-6.,-3.,3.,6.)]
        + [{'name':f'lateral_{v:+g}m','timing_m':0.,'lateral_m':v,'family':'lateral'} for v in (-3.,-1.5,1.5,3.)]
        + [{'name':f'joint_t{t:+g}_l{l:+g}','timing_m':t,'lateral_m':l,'family':'joint'} for t in (-6.,6.) for l in (-3.,3.)])


def smoothstep(value):
    t=np.clip(value,0.,1.)
    return t*t*(3.-2.*t)


def perturb_canonical(points, entry_s, exit_s, timing_m, lateral_m):
    """Guide-only deformation in canonical map coordinates, before common World warp.

    Timing: translate turn vicinity along incoming direction (world north), with
    smooth ramps before entry / after exit. Far approach and far exit are fixed.
    Lateral: signed offset along original Guide's left normal, full route.
    Timing is a spatial deformation, not a claim of exact arc-station shift.
    """
    p=np.asarray(points,dtype=float)
    s=arclength(p)
    weight=smoothstep((s-(entry_s-30.))/25.)*(1.-smoothstep((s-(exit_s+5.))/30.))
    t=tangent(p);normal=np.column_stack([-t[:,1],t[:,0]])
    return p + timing_m*weight[:,None]*(-OUT[0]) + lateral_m*normal


def build_guides(world, route, navigation):
    cfg=SimpleNamespace(**world['metadata'])
    base,entry,exit_,_=route_base(cfg,cfg,navigation,route['entry_lane_index'],route['exit_lane_index'],True,route['guide_parameters'])
    wm=world['metadata']['arm_warp']
    warp=ArmWarp(**{k:tuple(v) if isinstance(v,list) else v for k,v in wm.items()})
    original=np.asarray(route['guide_world'])
    closure=float(np.abs(deform(base,warp)-original).max())
    if closure>2e-5:
        raise ValueError(f'Guide reconstruction mismatch {closure}')
    result={}; rejected=[]
    for variant in variants():
        p=original if variant['name']=='original' else deform(perturb_canonical(base,entry,exit_,variant['timing_m'],variant['lateral_m']),warp)
        if not np.isfinite(p).all() or not LineString(p).is_simple:
            rejected.append({**variant,'reason':'non-finite or self-intersecting Guide route'})
            continue
        result[variant['name']]=p
    return result,rejected,closure


def path_geometry_metrics(path, road, road_guard, curbs, sample_step=.1):
    """Centerline metrics. Max penetration sampled densely (<=step/2 underest.)."""
    line=LineString(path)
    outside=float(line.difference(road_guard).length)
    # Full segment intersection, not only discrete vertex inclusion.
    crosses=any(line.crosses(boundary) for boundary in curbs if line.envelope.intersects(boundary.envelope))
    s=arclength(np.asarray(path))
    if s[-1]>0:
        ss=np.r_[np.arange(0.,s[-1],sample_step),s[-1]]
        dense=np.column_stack([np.interp(ss,s,np.asarray(path)[:,k]) for k in (0,1)])
        # A prepared road makes containment cheap. Compute expensive distances
        # only on the subset lying outside the unbuffered physical road.
        inside=shapely.intersects_xy(road,dense[:,0],dense[:,1])
        exterior=dense[~inside]
        depth=float(np.max(shapely.distance(shapely.points(exterior),road))) if len(exterior) else 0.
    else:
        depth=float(line.distance(road))
    return {'outside_road_length_m':outside,'road_violation':outside>.25,
            'curb_crossing':crosses,'max_penetration_m':depth}


def observed_curb_metrics(guide, pred, gt, encoded):
    """Crossings of retained, actually encoded curb fragments in ego coordinates.

    Fragments also intersecting GT are excluded from the GT-clear constraint
    metric, to avoid calling noisy observations an obstacle GT should obey.
    """
    boundaries=[]
    for i in range(len(encoded['bev_points'])):
        mask=encoded['point_mask'][i]
        if mask.sum()>=2 and encoded['polyline_attr'][i,1]>.5:
            boundaries.append(LineString(encoded['bev_points'][i,mask,:2].numpy()))
    truth=LineString(gt);gline=LineString(guide);pline=LineString(pred)
    safe=[b for b in boundaries if not truth.intersects(b)]
    return {'observed_curb_fragments':len(boundaries),'gt_clear_observed_curb_fragments':len(safe),
        'guide_observed_curb_conflict':any(gline.crosses(b) for b in safe),
        'observed_curb_crossing':any(pline.crosses(b) for b in safe),
        'observed_curb_crossing_raw':any(pline.crosses(b) for b in boundaries)}


def severity(depth):
    return ('none_or_under_8cm' if depth<=.08 else '8cm_to_0.5m' if depth<=.5 else
            '0.5_to_1m' if depth<=1. else '1_to_2m' if depth<=2. else 'over_2m')


def path_metrics(path, gt, row, route, world_path, context):
    _,road,guard,curbs,lane=context
    metrics=path_geometry_metrics(world_path,road,guard,curbs)
    err=np.linalg.norm(path-gt,axis=1)
    metrics.update(ade_m=float(err[1:].mean()),fde_m=float(err[-1]))
    future=row['metadata']['ego_route_s_m']+np.arange(81)
    turn=(future>=route['entry_s']-5)&(future<=route['exit_s']+3)
    metrics['turn_ade_m']=float(err[turn].mean()) if turn.sum()>=2 else None
    # Same GT world stations, fixed for every Guide variant of this base sample.
    offsets=np.arange(3.,31.)
    query=route['exit_s']+offsets-row['metadata']['ego_route_s_m']
    query=query[(query>=0)&(query<=80)]
    metrics['exit_stations']=len(query)
    if len(query)>=5:
        pp=np.column_stack([np.interp(query,np.arange(81),path[:,k]) for k in (0,1)])
        gg=np.column_stack([np.interp(query,np.arange(81),gt[:,k]) for k in (0,1)])
        directions=tangent(gg);delta=pp-gg
        lat=directions[:,0]*delta[:,1]-directions[:,1]*delta[:,0]
        metrics.update(exit_lateral_error_m=float(np.abs(lat).mean()),exit_signed_lateral_error_m=float(lat.mean()),
                       exit_ade_m=float(np.linalg.norm(delta,axis=1).mean()))
    else:
        metrics.update(exit_lateral_error_m=None,exit_signed_lateral_error_m=None,exit_ade_m=None)
    mask=future>=route['exit_s']+3
    metrics['exit_lane_violation']=bool(LineString(world_path[mask]).difference(lane).length>.25) if mask.sum()>=2 else None
    return metrics


def non_guide_fingerprint(row):
    # Guide and its crop station are the ONLY mutable input fields in this work.
    payload={k:v for k,v in row.items() if k not in ('reference_line',)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def summarize(rows):
    result={'cases':len(rows),'base_samples':len({r['base_id'] for r in rows}),
            'worlds':len({r['world_id'] for r in rows}),'routes':len({(r['world_id'],r['navigation']) for r in rows})}
    if not rows:return result
    for key in ('ade_m','fde_m','turn_ade_m','exit_lateral_error_m','exit_ade_m',
                'outside_road_length_m','road_violation','curb_crossing','max_penetration_m','exit_lane_violation',
                'guide_road_violation','guide_max_penetration_m','prediction_displacement_m','exit_change_from_original_m',
                'guide_observed_curb_conflict','observed_curb_crossing','observed_curb_crossing_raw'):
        values=[r[key] for r in rows if r.get(key) is not None]
        if values:result[key+'_mean']=float(np.mean(values))
    conflicts=[r for r in rows if r['guide_road_violation']]
    result['guide_road_conflict_cases']=len(conflicts)
    result['conflict_repair_rate']=float(np.mean([not r['road_violation'] for r in conflicts])) if conflicts else None
    if conflicts:
        result['conflict_ade_m']=float(np.mean([r['ade_m'] for r in conflicts]))
    observed=[r for r in rows if r.get('guide_observed_curb_conflict')]
    result['observed_guide_conflict_cases']=len(observed)
    result['observed_conflict_repair_rate']=float(np.mean([not r['observed_curb_crossing'] for r in observed])) if observed else None
    return result


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def run(args):
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    source=args.source/'audit/val.jsonl'
    manifest=inspect_split(source)
    train_manifest=inspect_split(args.source/'audit/train.jsonl')
    assert not set(manifest['worlds'])&set(train_manifest['worlds'])
    worlds=manifest['worlds'][:args.limit_worlds] if args.limit_worlds else manifest['worlds']
    raw=[json.loads(line) for line in source.open()]
    base_rows=[r for r in raw if r['world_id'] in worlds and r['navigation'] in ('left','right')
               and r['metadata']['pose_name'] in args.poses and r['metadata']['quality'] in args.qualities]
    if not base_rows:raise ValueError('No matching base samples')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model,saved=load_model(args.checkpoint,device)
    cfg=saved['config'];contexts={};routes_by_world={};guide_routes={};rejections=[];closure=0.;gt_out=0.
    for wid in worlds:
        w=json.loads((args.source/'worlds'/f'{wid}.json').read_text())
        routes=json.loads((args.source/'routes'/f'{wid}.json').read_text());routes_by_world[wid]=routes
        road=drivable_polygon(w);shapely.prepare(road);guard=road.buffer(.08)
        curbs=[LineString(e['points']) for e in w['physical']['elements'] if e['type']=='curb']
        for nav in ('left','right'):
            lane=target_lane_polygon(w,routes[nav]).buffer(.08)
            contexts[wid,nav]=(w,road,guard,curbs,lane)
            guides,rejected,error=build_guides(w,routes[nav],nav);guide_routes[wid,nav]=guides
            rejections.extend({'world_id':wid,'navigation':nav,**r} for r in rejected)
            closure=max(closure,error)
    # Cache model-independent geometry across the identical quality copies.
    geom_cache={};guide_cache={};bases={r['id']:r for r in base_rows}
    cases=[];encoded=[];guides=[];gts=[];queue=[];predictions=[]
    baseline_archive=np.load(args.baseline_predictions,allow_pickle=False)
    baseline_lookup={str(v):i for i,v in enumerate(baseline_archive['ids'])}
    baseline_max=0.
    variant_lookup={v['name']:v for v in variants()}
    recipe_file=(args.output_dir/'cases.jsonl').open('w')
    def flush():
        nonlocal baseline_max
        if not queue:return
        batch={k:torch.stack([e[k] for e in encoded]).to(device) for k in encoded[0] if k!='gt_path'}
        with torch.inference_mode():
            pp=model.forward_batch(batch).cpu().numpy()
        if not np.isfinite(pp).all():raise FloatingPointError('Non-finite prediction')
        for rec,row,guide,gt,p,e in zip(queue,[bases[x['base_id']] for x in queue],guides,gts,pp,encoded):
            rt=routes_by_world[row['world_id']][row['navigation']];context=contexts[row['world_id'],row['navigation']]
            pm=path_metrics(p,gt,row,rt,to_world(p,row['metadata']),context)
            rec.update(pm);rec.update(observed_curb_metrics(guide,p,gt,e));cases.append(rec);predictions.append(p)
            if rec['variant']=='original':
                old=baseline_archive['pred'][baseline_lookup[row['id']]]
                baseline_max=max(baseline_max,float(np.abs(old-p).max()))
            recipe_file.write(json.dumps({'case_id':rec['case_id'],'base_id':row['id'],
                'variant':variant_lookup[rec['variant']],'reference_line':guide.tolist(),
                'non_guide_sha256':rec['non_guide_sha256']},separators=(',',':'))+'\n')
        queue.clear();encoded.clear();guides.clear();gts.clear()
        if len(cases)%256==0:print(f'Predicted {len(cases)} cases',flush=True)
    for bi,row in enumerate(base_rows):
        wid,nav=row['world_id'],row['navigation'];md=row['metadata'];rt=routes_by_world[wid][nav]
        gt=np.asarray(row['gt_path']);fingerprint=non_guide_fingerprint(row)
        gt_world=to_world(gt,md)
        gt_out=max(gt_out,float(LineString(gt_world).difference(contexts[wid,nav][2]).length))
        for name,world_guide in guide_routes[wid,nav].items():
            key=(wid,nav,md['pose_name'],name)
            if key not in guide_cache:
                h=md['heading_rad'];direction=np.array([np.cos(h),np.sin(h)])
                reconstructed,start=sample_world_guide(world_guide,np.array(md['ego_world_xy']),direction)
                if name=='original':
                    delta=float(np.max(np.abs(reconstructed-np.asarray(row['reference_line']))))
                    if delta>1e-4:raise ValueError(f'Original crop mismatch {delta}')
                    reconstructed=np.asarray(row['reference_line'])
                step=np.linalg.norm(np.diff(reconstructed,axis=0),axis=1)
                if step.max()>1.001 or step.min()<.75:
                    raise ValueError(f'Invalid Guide spacing {key}: {step.min()}, {step.max()}')
                guide_cache[key]=(reconstructed,start)
            guide,start=guide_cache[key]
            changed={**row,'reference_line':guide.tolist()}
            assert non_guide_fingerprint(changed)==fingerprint
            if key not in geom_cache:
                gm=path_metrics(guide,gt,row,rt,to_world(guide,md),contexts[wid,nav])
                geom_cache[key]={'guide_'+k:v for k,v in gm.items()}
            variant=variant_lookup[name]
            rec={'case_id':f'{row["id"]}__{name}','base_id':row['id'],'world_id':wid,'navigation':nav,
                'pose':md['pose_name'],'quality':md['quality'],'variant':name,
                'family':variant['family'],'timing_m':variant['timing_m'],'lateral_m':variant['lateral_m'],
                'guide_crop_s_m':start,'non_guide_sha256':fingerprint,**geom_cache[key]}
            rec['severity']=severity(rec['guide_max_penetration_m'])
            queue.append(rec);encoded.append(encode_sample(changed,cfg.get('max_lines',32),cfg.get('max_points',32)))
            guides.append(guide);gts.append(gt)
            if len(queue)>=args.batch_size:flush()
    flush();recipe_file.close()
    if gt_out>1e-4:raise ValueError(f'GT leaves physical road: {gt_out}')
    if baseline_max>.005:raise ValueError(f'Original predictions mismatch {baseline_max}')
    pred=np.stack(predictions);np.savez_compressed(args.output_dir/'predictions.npz',ids=np.array([r['case_id'] for r in cases]),pred=pred)
    original={r['base_id']:(i,r) for i,r in enumerate(cases) if r['variant']=='original'}
    variant_spread=[]
    for i,rec in enumerate(cases):
        j,baseline=original[rec['base_id']]
        rec['prediction_displacement_m']=float(np.linalg.norm(pred[i,1:]-pred[j,1:],axis=1).mean())
        a,b=rec.get('exit_signed_lateral_error_m'),baseline.get('exit_signed_lateral_error_m')
        rec['exit_change_from_original_m']=abs(a-b) if a is not None and b is not None else None
    pools=defaultdict(list)
    for r in cases:pools[r['base_id']].append(r)
    for key,rs in pools.items():
        values=[r['exit_signed_lateral_error_m'] for r in rs if r['exit_signed_lateral_error_m'] is not None]
        variant_spread.append({'base_id':key,'world_id':rs[0]['world_id'],'navigation':rs[0]['navigation'],
            'quality':rs[0]['quality'],'pose':rs[0]['pose'],'variants':len(rs),
            'exit_signed_lateral_range_m':float(np.ptp(values)) if values else None,
            'any_road_violation':any(r['road_violation'] for r in rs)})
    write_csv(args.output_dir/'per_case.csv',cases);write_csv(args.output_dir/'per_base_stability.csv',variant_spread)
    paired=[]
    for name in variant_lookup:
        rs=[r for r in cases if r['variant']==name]
        if not rs:continue
        indices=[original[r['base_id']][1] for r in rs]
        paired.append({'variant':name,'matched_cases':len(rs),
            'ade_delta_m':float(np.mean([r['ade_m']-o['ade_m'] for r,o in zip(rs,indices)])),
            'road_violation_delta':float(np.mean([int(r['road_violation'])-int(o['road_violation']) for r,o in zip(rs,indices)])),
            'new_road_violation_cases':sum(r['road_violation'] and not o['road_violation'] for r,o in zip(rs,indices)),
            'fixed_road_violation_cases':sum(not r['road_violation'] and o['road_violation'] for r,o in zip(rs,indices))})
    summary={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'checkpoint_epoch':saved.get('epoch'),'device':str(device),'torch':torch.__version__,'source':str(source.resolve()),
        'source_manifest':manifest,'selected_worlds':worlds,'poses':args.poses,'qualities':args.qualities,
        'variants':variants(),'cases':len(cases),'base_samples':len(base_rows),'rejected_routes':rejections,
        'overall':summarize(cases),'by_variant':{n:summarize([r for r in cases if r['variant']==n]) for n in variant_lookup},
        'by_severity':{n:summarize([r for r in cases if r['severity']==n]) for n in sorted({r['severity'] for r in cases})},
        'by_navigation':{n:summarize([r for r in cases if r['navigation']==n]) for n in ('left','right')},
        'by_quality':{n:summarize([r for r in cases if r['quality']==n]) for n in args.qualities},
        'by_pose':{n:summarize([r for r in cases if r['pose']==n]) for n in args.poses},
        'by_observed_guide_conflict':{str(v):summarize([r for r in cases if r['guide_observed_curb_conflict']==v]) for v in (False,True)},
        'paired_vs_original':paired,
        'stability':{'base_samples':len(variant_spread),'mean_exit_lateral_range_m':float(np.mean([r['exit_signed_lateral_range_m'] for r in variant_spread if r['exit_signed_lateral_range_m'] is not None])),
            'bases_with_any_road_violation':sum(r['any_road_violation'] for r in variant_spread)},
        'checks':{'canonical_guide_reconstruction_max_m':closure,'gt_outside_guard_max_length_m':gt_out,
            'original_prediction_max_coordinate_difference_m':baseline_max,'all_non_guide_fingerprints_identical':True,'train_val_world_overlap':[]},
        'limits':['Validation-only diagnostic stress suite; held-out test split untouched. No training.',
            'Normal means original Guide, not GT or zero-bias privileged geometry.',
            'Positive timing translates turn vicinity along incoming direction before World warp; localized ramps restore far approach/exit. Not an exact timing/arc-station change.',
            'Lateral offset uses original canonical Guide left normal, then common World warp. Some combinations intentionally exceed training distribution.',
            'Road violation: >0.25m centerline outside road buffered8cm. Curb crossing: exact full LineString crossing physical curb.',
            'Max penetration uses unbuffered road and0.1m path sampling, underestimation at most0.05m for piecewise linear path.',
            'Exit lateral error uses identical GT world stations exit+3..30m. Exit range is range of MEAN signed errors across Guides, not pointwise worst-case spread.',
            'Repeated poses, qualities, variants are correlated; independent sample size is Worlds/routes, not case count.',
            'Observed curb crossings use retained encoded fragments; GT-intersecting noisy fragments excluded from GT-clear metric and included separately in raw metric.',
            'Self-intersecting Guide routes excluded explicitly; rejection list retained. Vehicle footprint not assessed.']}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot_results(args.output_dir,cases)
    export_examples(args.output_dir,cases,pred,bases,guide_cache,contexts)
    print('OUTPUT',args.output_dir,flush=True)
    print(json.dumps({'cases':len(cases),'base_samples':len(base_rows),'rejected_routes':len(rejections),'checks':summary['checks'],'overall':summary['overall']},indent=2),flush=True)


def plot_results(out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[v['name'] for v in variants()]
    groups=[summarize([r for r in records if r['variant']==n]) for n in names]
    fig,axes=plt.subplots(3,1,figsize=(14,11),sharex=True)
    for ax,key,label in zip(axes,['ade_m_mean','road_violation_mean','exit_lateral_error_m_mean'],['Full 80m ADE (m)','Road violation fraction','Exit lateral error (m)']):
        ax.bar(names,[g.get(key,0) for g in groups]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[-1].tick_params(axis='x',labelrotation=35)
    fig.suptitle('Fixed roads / GT / observations; Guide-only interventions')
    fig.tight_layout();fig.savefig(out/'variant_metrics.png',dpi=150);plt.close(fig)


def export_examples(out,records,pred,bases,guide_cache,contexts):
    import matplotlib.pyplot as plt
    # Select contrasting successful and failed conflicts; preserve replayable input.
    conflict=[(i,r) for i,r in enumerate(records) if r['guide_road_violation'] and r['family']!='original']
    failures=sorted([(i,r) for i,r in conflict if r['road_violation']],key=lambda z:z[1]['max_penetration_m'],reverse=True)
    successes=sorted([(i,r) for i,r in conflict if not r['road_violation']],key=lambda z:z[1]['guide_max_penetration_m'],reverse=True)
    picks=[];seen=set()
    for pool in (failures,successes):
        count=0
        for item in pool:
            if item[1]['world_id'] in seen:continue
            picks.append(item);seen.add(item[1]['world_id']);count+=1
            if count==3:break
    if not picks:return
    example_dir=out/'examples';example_dir.mkdir()
    fig,axes=plt.subplots(2,3,figsize=(18,11));index=[]
    for ax,(i,r) in zip(axes.flat,picks):
        row=bases[r['base_id']];md=row['metadata'];guide=guide_cache[(row['world_id'],row['navigation'],md['pose_name'],r['variant'])][0]
        case={**row,'reference_line':guide.tolist()}
        destination=example_dir/(r['case_id']+'.json');destination.write_text(json.dumps(case,indent=2)+'\n')
        world=contexts[row['world_id'],row['navigation']][0]
        from Dataset.audit_lane_visibility import transform
        for e in world['physical']['elements']:
            if e['type'] not in ('curb','line'):continue
            p=transform(e['points'],md['ego_world_xy'],md['heading_rad'])
            ax.plot(-p[:,1],p[:,0],color='#555555' if e['type']=='curb' else '#cccccc',lw=1)
        lines=[(np.asarray(row['gt_path']),'GT','#159447','-'),(guide,'Guide','#e99a00','--'),(pred[i],'Prediction','#db3030','-')]
        for p,label,color,style in lines:ax.plot(-p[:,1],p[:,0],label=label,color=color,ls=style,lw=2)
        points=np.concatenate([p[:51] for p,*_ in lines])
        ax.set(xlim=(-points[:,1].max()-3,-points[:,1].min()+3),ylim=(points[:,0].min()-3,points[:,0].max()+3),
            title=f'{r["navigation"]}/{r["pose"]}/{r["quality"]}\n{r["variant"]}: ADE={r["ade_m"]:.2f}, depth={r["max_penetration_m"]:.2f}m')
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8);index.append({'input':str(destination),**r})
    for ax in list(axes.flat)[len(picks):]:ax.axis('off')
    fig.tight_layout();fig.savefig(out/'examples.png',dpi=150);plt.close(fig)
    (example_dir/'index.json').write_text(json.dumps(index,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('outputs/v4_world300_inside13'))
    ap.add_argument('--checkpoint',type=Path,default=Path('outputs/context_comparison_inside13/mixed/best.pt'))
    ap.add_argument('--baseline-predictions',type=Path,default=Path('outputs/context_comparison_inside13/mixed/evaluation_best_val/predictions.npz'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/guide_conflict_benchmark')/datetime.now().strftime('%Y%m%d_%H%M%S'))
    ap.add_argument('--device',default='auto');ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--threads',type=int,default=4);ap.add_argument('--limit-worlds',type=int,default=0)
    ap.add_argument('--poses',nargs='+',default=['approach_10','entry','inside_8'])
    ap.add_argument('--qualities',nargs='+',default=['clean','medium','hard'])
    run(ap.parse_args())

if __name__=='__main__':main()
