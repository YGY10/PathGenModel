"""Oracle diagnostic: append ideal exit-arm boundaries to frozen original inputs.

Uses true route/physical identities ONLY to construct a diagnostic intervention.
Does not represent a deployable perception or token-selection policy.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import LineString, Point
from shapely.ops import substring

from analyze_context_failures import plot_case
from analyze_corrections import classify_correction
from Dataset.context_augmentation import append_observations
from Dataset.dataset import _objects
from Dataset.audit_drivable import drivable_polygon
from Dataset.audit_lane_visibility import ARMS, sample_at, target_boundaries, transform
from diagnose_paths import target_lane_polygon
from evaluate import EvaluationDataset, load_model
from evaluate_context_comparison import inference
from experiment_fov import exit_coverage, metrics
from train import inspect_split


def ideal_boundaries(row, world, route, end_m=40.):
    """All outgoing lane edges, preserving each physical polyline's point order."""
    centers = sample_at(route['gt_world'], route['exit_s'] + np.array([3., end_m]))
    if centers is None:
        raise ValueError('Exit road too short')
    arm = ARMS[int(route['exit_arm'])]
    n = int(world['metadata']['num_out'][int(route['exit_arm'])])
    ids = [f'{arm}_divider_{-k:+d}' for k in range(n)] + [f'{arm}_curb_outgoing']
    physical = {e['id']: e for e in world['physical']['elements']}
    result = []
    for source in ids:
        obj = physical[source]
        line = LineString(obj['points'])
        start, end = sorted(line.project(Point(p)) for p in centers)
        segment = substring(line, start, end)
        if segment.geom_type != 'LineString' or segment.length < 1:
            raise ValueError('Invalid boundary segment')
        xy = transform(np.asarray(segment.coords), row['metadata']['ego_world_xy'], row['metadata']['heading_rad'])
        kind = 'road_boundary' if obj['type'] == 'curb' else 'lane_marking'
        result.append((kind, {'points':xy.tolist(), 'source_id':source, 'confidence':1.,
                              'color':obj.get('color'), 'line_style':obj.get('style')}))
    return result


def run(args):
    source = json.loads((args.source_dir / 'summary.json').read_text())
    comparison = json.loads((args.evaluation_dir / 'summary.json').read_text())
    if inspect_split(source['input']) != comparison['data_manifest']:
        raise ValueError('Dataset fingerprint mismatch')
    base = EvaluationDataset(source['input'], 32, 32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, saved = load_model(comparison['models']['mixed']['checkpoint'], device)
    if saved['splits']['val'] != comparison['data_manifest']:
        raise ValueError('Checkpoint split mismatch')
    with np.load(args.evaluation_dir / 'predictions.npz') as a:
        original = a['mixed_original']
        ids = a['ids'].tolist()
        seeds = [k for k in a.files if k.startswith('mixed_expanded_seed_')]
        expanded = [a[k] for k in seeds]
    if ids != [base.raw_row(i)['id'] for i in range(len(base))]:
        raise ValueError('Saved prediction IDs differ')
    focus, contexts, datasets, observations, audit = [], [], {'ideal_all_exit':[], 'ideal_target':[]}, {}, []
    cache, skipped = {}, []
    for i in range(len(base)):
        row = base.raw_row(i)
        if row['metadata']['pose_name'] not in args.poses:
            continue
        wid = row['world_id']
        if wid not in cache:
            world = json.loads((args.worlds_dir / f'{wid}.json').read_text())
            routes = json.loads((args.routes_dir / f'{wid}.json').read_text())
            cache[wid] = (world, routes, drivable_polygon(world))
        world, routes, road = cache[wid]; route = routes[row['navigation']]
        encoded = base[i]
        targets = target_boundaries(world, route)
        all_edges = ideal_boundaries(row, world, route, args.exit_end_m)
        choices = {'ideal_all_exit':all_edges,
                   'ideal_target':[item for item in all_edges if item[1]['source_id'] in targets]}
        # Never truncate one intervention silently or displace original tokens.
        free = int((~encoded['polyline_mask']).sum())
        if free < len(all_edges):
            skipped.append({'index':i, 'id':row['id'], 'free_slots':free, 'required_slots':len(all_edges)})
            continue
        obs = _objects(row, 32)
        source_present = {o.get('source_id') for k,o in obs}
        coverage_before = exit_coverage(row, world, route, obs, 1.25)
        for arm, candidates in choices.items():
            supplemented, kept = append_observations(encoded, candidates, 32)
            assert len(kept) == len(candidates)
            datasets[arm].append(supplemented)
            audit.append({'index':i, 'id':row['id'], 'intervention':arm, 'original_tokens':32-free,
                          'added_tokens':len(kept), 'target_source_ids':list(targets),
                          'target_sources_before':sum(t in source_present for t in targets),
                          'two_side_stations_before':coverage_before,
                          'two_side_stations_after':exit_coverage(row, world, route, obs+kept, 1.25)})
        observations[i] = [{'kind':k, **o} for k,o in all_edges]
        focus.append(i)
        contexts.append((row, route, road, target_lane_polygon(world, route)))
    if not focus:
        raise ValueError('No selected samples')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    predictions = {arm:inference(model, data, device, args.batch_size) for arm,data in datasets.items()}
    baseline = inference(model, [base[i] for i in focus], device, args.batch_size)
    max_diff = float(np.max(np.abs(baseline-original[focus])))
    if max_diff > 1e-4:
        raise ValueError(f'Baseline reproduction mismatch {max_diff}')
    rows = []
    stable = []
    for j, i in enumerate(focus):
        row, route, road, lane = contexts[j]
        gt, guide = np.asarray(row['gt_path']), np.asarray(row['reference_line'])
        stable_wrong = all(classify_correction(guide, gt, p[i])['error_dominant']=='wrong_direction' for p in expanded)
        stable.append(stable_wrong)
        for arm, pred in [('original', baseline[j]), *[(arm,p[j]) for arm,p in predictions.items()]]:
            labels = classify_correction(guide, gt, pred)
            rows.append({'index':i, 'id':row['id'], 'world_id':row['world_id'], 'pose':row['metadata']['pose_name'],
                         'navigation':row['navigation'], 'quality':row['metadata']['quality'], 'arm':arm,
                         'stable_wrong_expanded':stable_wrong, **metrics(row,pred,gt,guide,road,lane,route),
                         'error_dominant':labels['error_dominant']})
    result = {}
    for group, subset in [('all_focus', rows), ('stable_wrong_expanded', [r for r in rows if r['stable_wrong_expanded']])]:
        result[group] = {}
        for arm in ['original', *predictions]:
            rs = [r for r in subset if r['arm']==arm]
            result[group][arm] = {'samples':len(rs),
                **{k:float(np.mean([r[k] for r in rs])) if rs else None for k in
                   ['ade_m','fde_m','wrong_direction','outside_road','outside_target_lane']}}
    for filename, data in [('per_sample.csv',rows), ('input_audit.csv',audit)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    np.savez_compressed(out/'predictions.npz', indices=focus, ids=np.array([ids[i] for i in focus]),
                        original=baseline, **predictions)
    (out/'added_observations.jsonl').write_text(''.join(json.dumps({'index':i,'id':ids[i],
        'added_objects':observations[i]})+'\n' for i in focus))
    import matplotlib
    matplotlib.use('Agg')
    full_arrays = {'mixed_original':original}
    for arm, values in predictions.items():
        full_arrays[f'mixed_expanded_seed_{arm}'] = np.full_like(original, np.nan)
        full_arrays[f'mixed_expanded_seed_{arm}'][focus] = values
    pool = [{'id':ids[i], 'added_objects':observations.get(i,[])} for i in range(len(base))]
    selected = [r for r in rows if r['arm']=='original' and r['stable_wrong_expanded']]
    selected = sorted(selected, key=lambda r:-r['ade_m'])[:args.examples]
    for row in selected:
        plot_case(base,pool,full_arrays,row['index'],list(predictions),'ideal_all_exit',out/f'{row["index"]:06d}.png')
    summary = {'results':result, 'skipped_capacity':skipped, 'baseline_max_abs_difference_m':max_diff,
               'exit_segment_m':[3.,args.exit_end_m], 'selected':selected,
               'data_manifest':comparison['data_manifest'],
               'checkpoint':comparison['models']['mixed']['checkpoint'],
               'coverage_check':{arm:{
                   'mean_two_side_stations_before':float(np.mean([r['two_side_stations_before'] for r in audit if r['intervention']==arm])),
                   'min_two_side_stations_after':min(r['two_side_stations_after'] for r in audit if r['intervention']==arm)}
                   for arm in predictions},
               'notes':['ORACLE diagnostic: true exit arm, target identity and ideal physical geometry used.',
                        'Original valid tokens and non-BEV inputs preserved; append only in padding.',
                        'All-exit arm includes every outgoing lane edge; target-only includes two target edges.',
                        'These two interventions differ in token count and target selection; not an isolated attention test.',
                        'Existing noisy/partial detections are retained even when supplemented by an ideal segment.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-dir',type=Path,default=Path('outputs/context_comparison/evaluation'))
    p.add_argument('--source-dir',type=Path,default=Path('outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A'))
    p.add_argument('--worlds-dir',type=Path,default=Path('outputs/v4_world300_fixedguide/worlds'))
    p.add_argument('--routes-dir',type=Path,default=Path('outputs/v4_world300_fixedguide/routes'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/context_comparison/exit_boundary_oracle'))
    p.add_argument('--poses',nargs='+',default=['entry','approach_10'])
    p.add_argument('--exit-end-m',type=float,default=40.)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--examples',type=int,default=4)
    args=p.parse_args()
    if args.exit_end_m<=3 or args.batch_size<1 or args.examples<0:
        p.error('Require exit-end-m > 3, batch-size >= 1, examples >= 0')
    run(args)
