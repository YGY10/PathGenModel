"""Audit existing splits and frozen predictions; never train or rewrite inputs.

Physical geometry is diagnostic truth only. Observed conflicts use exactly the
retained 32-point representation. Repeated poses/qualities are not independent.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dataset.dataset import _objects, resample
from Dataset.audit_drivable import drivable_polygon
from Dataset.audit_lane_visibility import target_boundaries, transform, sample_at
from Scenario.export_oracle_lanes import entry_boundaries
from diagnose_paths import to_world, tangent
from diagnose_pose_progression import at_world_stations
from train import inspect_split


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows):
    result = {'samples': len(rows), 'worlds': len({r['world_id'] for r in rows}),
              'routes': len({(r['world_id'], r['navigation']) for r in rows})}
    if not rows:
        return result
    for key in ('guide_target_cross', 'guide_curb_cross', 'guide_road_conflict',
                'guide_road_severe', 'guide_exit_outside', 'guide_observed_target_conflict',
                'guide_observed_curb_conflict', 'pred_target_cross', 'pred_curb_cross',
                'pred_road_conflict', 'pred_exit_outside', 'ade_m', 'exit_ade_m',
                'exit_lateral_error_m', 'guide_outside_road_length_m',
                'pred_outside_road_length_m'):
        if any(key in r for r in rows):
            result[key + ('_rate' if key in ('guide_target_cross','guide_curb_cross',
                'guide_road_conflict','guide_road_severe','guide_exit_outside',
                'guide_observed_target_conflict','guide_observed_curb_conflict',
                'pred_target_cross','pred_curb_cross','pred_road_conflict','pred_exit_outside') else '_mean')] = mean(rows, key)
    for key in ('target_conflict_boundaries','target_unrepaired_boundaries',
                'curb_conflict_boundaries','curb_unrepaired_boundaries',
                'observed_target_conflict_boundaries','observed_target_unrepaired_boundaries'):
        result[key] = sum(r.get(key, 0) for r in rows)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('outputs/v4_world300_inside13'))
    ap.add_argument('--evaluation', type=Path, default=Path('outputs/context_comparison_inside13/mixed/evaluation_best_val'))
    ap.add_argument('--pose-evaluation', type=Path, default=Path('outputs/context_comparison_inside13/mixed/pose_context_evaluation'))
    ap.add_argument('--output-dir', type=Path, default=Path('outputs/guide_boundary_audit')/datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    evaluation = json.loads((args.evaluation/'summary.json').read_text())
    assert inspect_split(args.source/'audit/val.jsonl') == evaluation['data_manifest']
    arrays = np.load(args.evaluation/'predictions.npz', allow_pickle=False)
    pose_arrays = np.load(args.pose_evaluation/'predictions.npz', allow_pickle=False)
    assert np.array_equal(arrays['ids'], pose_arrays['ids'])
    contexts = {}; physical_cache = {}; records = []; val_raw = []
    unique_route_params = {}; closure_max = 0.
    for split in ('train', 'val'):
        with (args.source/'audit'/f'{split}.jsonl').open() as stream:
            for index, line in enumerate(stream):
                row = json.loads(line); md = row['metadata']; wid = row['world_id']; nav = row['navigation']
                if wid not in contexts:
                    world = json.loads((args.source/'worlds'/f'{wid}.json').read_text())
                    routes = json.loads((args.source/'routes'/f'{wid}.json').read_text())
                    physical = {e['id']: e for e in world['physical']['elements']}
                    contexts[wid] = (world, routes, physical, drivable_polygon(world).buffer(.08))
                world, routes, physical, road = contexts[wid]; route = routes[nav]
                routekey = (split, wid, nav)
                unique_route_params[routekey] = route['guide_parameters']
                key = (wid, nav, md['pose_name'])
                target_ids = set((*entry_boundaries(world, route), *target_boundaries(world, route)))
                curb_ids = {k for k,v in physical.items() if v['type']=='curb'}
                guide = to_world(np.asarray(row['reference_line']), md)
                gt = to_world(np.asarray(row['gt_path']), md)
                gline, tline = LineString(guide), LineString(gt)
                if key not in physical_cache:
                    relevant = {s: LineString(physical[s]['points']) for s in target_ids|curb_ids}
                    # Discount crossings also present in GT; line identity alone
                    # does not define whether a trajectory is a valid connection.
                    safe = {s: boundary for s,boundary in relevant.items() if not tline.intersects(boundary)}
                    gcross = {s for s,boundary in safe.items() if gline.crosses(boundary)}
                    outside = float(gline.difference(road).length)
                    gout = {'guide_target_cross':bool(gcross & target_ids),
                            'guide_curb_cross':bool(gcross & curb_ids),
                            'guide_outside_road_length_m':outside,
                            'guide_road_conflict':outside>.25, 'guide_road_severe':outside>1.,
                            'gt_outside_road_length_m':float(tline.difference(road).length)}
                    physical_cache[key] = (gout, safe, gcross)
                gout, safe, gcross = physical_cache[key]
                rec = {'split':split,'index':index,'id':row['id'],'world_id':wid,
                       'navigation':nav,'pose':md['pose_name'],'quality':md['quality'],**gout}
                observed = []
                for kind,obj in _objects(row, 32):
                    if kind not in ('lane_marking','road_boundary'):
                        continue
                    source = obj.get('source_id')
                    if source not in target_ids and kind!='road_boundary':
                        continue
                    boundary = LineString(resample(obj['points'], 32))
                    # Avoid treating observation noise intersecting the GT as a
                    # valid constraint the model ought to obey.
                    if not LineString(row['gt_path']).intersects(boundary):
                        observed.append((source, kind, boundary))
                lg = LineString(row['reference_line'])
                observed_gcross = [(s,k,b) for s,k,b in observed if lg.crosses(b)]
                rec['guide_observed_target_conflict'] = any(s in target_ids for s,k,b in observed_gcross)
                rec['guide_observed_curb_conflict'] = any(k=='road_boundary' for s,k,b in observed_gcross)
                exit_mask = np.arange(81) >= route['exit_s']-md['ego_route_s_m']+3
                # Diagnostic equal-index GT stations, not a path-nearest rematch.
                exit_ids = target_boundaries(world, route)
                from diagnose_paths import target_lane_polygon
                lane_key = ('lane',wid,nav)
                if lane_key not in physical_cache:
                    physical_cache[lane_key] = target_lane_polygon(world, route).buffer(.08)
                lane = physical_cache[lane_key]
                if exit_mask.sum()>=5:
                    rec['guide_exit_outside'] = LineString(guide[exit_mask]).difference(lane).length>.25
                if split=='val':
                    assert str(arrays['ids'][index]) == row['id']
                    assert np.max(np.abs(arrays['gt'][index]-np.asarray(row['gt_path'])))<1e-4
                    assert np.max(np.abs(arrays['guide'][index]-np.asarray(row['reference_line'])))<1e-4
                    p = arrays['pred'][index]; pred = to_world(p, md); pline = LineString(pred)
                    pcross = {s for s,boundary in safe.items() if pline.crosses(boundary)}
                    outside = float(pline.difference(road).length)
                    rec.update(pred_target_cross=bool(pcross&target_ids), pred_curb_cross=bool(pcross&curb_ids),
                        pred_outside_road_length_m=outside, pred_road_conflict=outside>.25,
                        ade_m=float(np.linalg.norm(p[1:]-arrays['gt'][index,1:],axis=1).mean()),
                        target_conflict_boundaries=len(gcross&target_ids),
                        target_unrepaired_boundaries=len(gcross&pcross&target_ids),
                        curb_conflict_boundaries=len(gcross&curb_ids),
                        curb_unrepaired_boundaries=len(gcross&pcross&curb_ids),
                        observed_target_conflict_boundaries=sum(s in target_ids for s,k,b in observed_gcross),
                        observed_target_unrepaired_boundaries=sum(s in target_ids and LineString(p).crosses(b) for s,k,b in observed_gcross))
                    if exit_mask.sum()>=5:
                        err=pred[exit_mask]-gt[exit_mask];directions=tangent(gt)[exit_mask]
                        lateral=directions[:,0]*err[:,1]-directions[:,1]*err[:,0]
                        rec.update(exit_ade_m=float(np.linalg.norm(err,axis=1).mean()),
                            exit_lateral_error_m=float(np.abs(lateral).mean()),
                            pred_exit_outside=LineString(pred[exit_mask]).difference(lane).length>.25)
                    val_raw.append(row)
                records.append(rec)
                if (index+1)%2100==0:
                    print(split,index+1,'rows checked',flush=True)
        print('Completed',split,flush=True)
    write_csv(args.output_dir/'per_sample.csv',records)
    summary = {'source':str(args.source),'evaluation':str(args.evaluation),
        'definitions':{'road':'Reconstructed independent physical road, 8cm buffer. Conflict: >0.25m of path outside; severe: >1m outside. Length is not penetration depth.',
        'target_cross':'Cross physical selected incoming/target outgoing boundary, excluding boundaries intersected by GT. Tangencies and paths parallel outside are not counted.',
        'observed':'Retained max32 objects, resampled32 points; exclude boundary fragments intersecting GT; original input only, not stochastic training supplements.',
        'exit':'GT equal-index stations at least3m after exit; >=5 points. Target lane with8cm buffer; >0.25m outside. Auxiliary diagnostic, may include longitudinal phase error.',
        'stability':'Same World/nav/quality, compare entry/inside8/inside13 at identical world route exit+3..30m stations. Pose, observation and Guide crop all change; descriptive, not causal.'},
        'splits':{},'validation_groups':{},'generator':{},'checks':{}}
    for split in ('train','val'):
        rs=[r for r in records if r['split']==split]
        clean=[r for r in rs if r['quality']=='clean']
        summary['splits'][split]={'all':aggregate(rs),'unique_pose_clean':aggregate(clean),
            'routes_with_target_conflict':len({(r['world_id'],r['navigation']) for r in clean if r['guide_target_cross']}),
            'routes_with_road_conflict':len({(r['world_id'],r['navigation']) for r in clean if r['guide_road_conflict']}),
            'by_navigation':{n:aggregate([r for r in rs if r['navigation']==n]) for n in ('left','right','straight')},
            'by_pose':{n:aggregate([r for r in rs if r['pose']==n]) for n in sorted({r['pose'] for r in rs})}}
    vr=[r for r in records if r['split']=='val']
    for field in ('guide_target_cross','guide_road_conflict','guide_road_severe','guide_observed_target_conflict'):
        summary['validation_groups'][field]={str(value):aggregate([r for r in vr if r[field]==value]) for value in (False,True)}
    # Broad context-conditioned stratification: avoid attributing pose/turn mix
    # differences to conflict alone.
    summary['validation_strata']={}
    for nav in ('left','right','straight'):
        for pose in sorted({r['pose'] for r in vr}):
            rs=[r for r in vr if r['navigation']==nav and r['pose']==pose]
            summary['validation_strata'][nav+'/'+pose]={str(v):aggregate([r for r in rs if r['guide_target_cross']==v]) for v in (False,True)}
    # Matched world-station consistency on saved original + expanded predictions.
    lookup=defaultdict(dict)
    by_id={r['id']:r for r in vr}
    for i,row in enumerate(val_raw):
        md=row['metadata']
        if md['pose_name'] in ('entry','inside_8','inside_13'):
            lookup[(row['world_id'],row['navigation'],md['quality'])][md['pose_name']]=i
    stability=[]
    for (wid,nav,quality),indices in lookup.items():
        if len(indices)!=3:
            continue
        route=contexts[wid][1][nav];stations=route['exit_s']+np.arange(3.,31.)
        if any(np.min(stations-val_raw[i]['metadata']['ego_route_s_m'])<0 or np.max(stations-val_raw[i]['metadata']['ego_route_s_m'])>80 for i in indices.values()):
            continue
        gt=sample_at(route['gt_world'],stations);direction=tangent(gt)
        for variant in ('original','expanded_seed_724','expanded_seed_725','expanded_seed_726'):
            lats=[];errs=[]
            for pose in ('entry','inside_8','inside_13'):
                i=indices[pose];md=val_raw[i]['metadata']
                reconstructed=at_world_stations(pose_arrays['gt'][i],md,stations)
                closure_max=max(closure_max,float(np.linalg.norm(reconstructed-gt,axis=1).max()))
                pred=at_world_stations(pose_arrays[variant][i],md,stations);delta=pred-gt
                lats.append(direction[:,0]*delta[:,1]-direction[:,1]*delta[:,0]);errs.append(float(np.linalg.norm(delta,axis=1).mean()))
            lats=np.array(lats);entry=by_id[val_raw[indices['entry']]['id']]
            stability.append({'world_id':wid,'navigation':nav,'quality':quality,'variant':variant,
                'entry_guide_target_cross':entry['guide_target_cross'],'entry_guide_road_conflict':entry['guide_road_conflict'],
                'mean_lateral_spread_across_poses_m':float(np.ptp(lats,axis=0).mean()),
                'entry_inside13_lateral_difference_m':float(np.abs(lats[0]-lats[2]).mean()),
                'entry_error_m':errs[0],'inside8_error_m':errs[1],'inside13_error_m':errs[2]})
    assert closure_max<.02, closure_max
    write_csv(args.output_dir/'pose_stability.csv',stability)
    def stable_summary(rs):
        if not rs:return {'triplets':0}
        return {'triplets':len(rs),'worlds':len({r['world_id'] for r in rs}),
            **{k:mean(rs,k) for k in ('mean_lateral_spread_across_poses_m','entry_inside13_lateral_difference_m','entry_error_m','inside8_error_m','inside13_error_m')},
            'spread_over_0_5m_rate':float(np.mean([r['mean_lateral_spread_across_poses_m']>.5 for r in rs])),
            'spread_over_1m_rate':float(np.mean([r['mean_lateral_spread_across_poses_m']>1 for r in rs]))}
    summary['stability']={}
    for variant in ('original','expanded_seeds_pooled'):
        rs=[r for r in stability if (r['variant']=='original')==(variant=='original')]
        summary['stability'][variant]={'all':stable_summary(rs),
            'by_entry_target_conflict':{str(v):stable_summary([r for r in rs if r['entry_guide_target_cross']==v]) for v in (False,True)},
            'by_entry_road_conflict':{str(v):stable_summary([r for r in rs if r['entry_guide_road_conflict']==v]) for v in (False,True)},
            'by_navigation':{n:stable_summary([r for r in rs if r['navigation']==n]) for n in ('left','right','straight')}}
    for split in ('train','val'):
        params=[v for k,v in unique_route_params.items() if k[0]==split]
        summary['generator'][split]={'routes':len(params),**{key:{'min':float(np.min([v[key] for v in params])),
            'max':float(np.max([v[key] for v in params]))} for key in ('lateral_offset_m','entry_lateral_m','handle_scale')}}
    summary['checks']={'matched_gt_closure_max_m':closure_max,
        'gt_outside_road_max_length_m':max(r['gt_outside_road_length_m'] for r in records),
        'validation_prediction_ids_and_gt_and_guide':'all checked',
        'train_val_world_overlap':sorted(set(r['world_id'] for r in records if r['split']=='train')&set(r['world_id'] for r in vr))}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('OUTPUT',args.output_dir,flush=True)
    print(json.dumps({'splits':{s:{k:v for k,v in d.items() if k in ('all','unique_pose_clean','routes_with_target_conflict','routes_with_road_conflict')} for s,d in summary['splits'].items()},'validation_groups':summary['validation_groups'],'stability':summary['stability'],'checks':summary['checks']},indent=2),flush=True)

if __name__=='__main__':
    main()
