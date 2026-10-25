"""Frozen-checkpoint Near/Far/Random x K ablation on SAVED FOV supplements.

No observation regeneration, model training, or GT-based selection. Candidate pool
is the exact retained supplement from experiment_fov, already range/capacity limited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import LineString

from Dataset.dataset import encode_sample, _objects, resample
from Dataset.audit_lane_visibility import transform, sample_at, target_boundaries, point_polyline_distance
from Dataset.audit_drivable import drivable_polygon
from diagnose_paths import target_lane_polygon
from evaluate import EvaluationDataset, load_model
from experiment_fov import append_observations, metrics, paired_summary
from train import inspect_split


def describe_candidate(item, guide):
    kind, obj = item
    closed = kind in ('stop_line', 'crosswalk')
    xy = resample(obj['points'], 32, closed=closed)
    line = np.vstack([xy, xy[:1]]) if closed else xy
    segments = np.diff(line, axis=0)
    centers = (line[:-1] + line[1:]) / 2
    a, v = guide[:-1], np.diff(guide, axis=0)
    u = np.clip(((centers[:, None] - a[None]) * v[None]).sum(-1) / np.maximum((v*v).sum(-1), 1e-12), 0, 1)
    projection = a[None] + u[..., None] * v[None]
    nearest = ((projection - centers[:, None]) ** 2).sum(-1).argmin(-1)
    matched = v[nearest]
    alignment = np.abs((matched * segments).sum(-1)) / np.maximum(np.linalg.norm(matched, axis=-1) * np.linalg.norm(segments, axis=-1), 1e-12)
    return {'kind': kind, 'distance_to_guide_m': float(LineString(line).distance(LineString(guide))),
            'mean_point_distance_to_guide_m': float(point_polyline_distance(xy, guide).mean()),
            'length_m': float(np.linalg.norm(segments, axis=-1).sum()),
            'mean_abs_direction_alignment': float(alignment.mean()),
            'confidence': float(obj.get('confidence', 1.))}


def select_indices(descriptors, mode, k, sample_id, seed):
    """All K share a single ordering; restore pool order after subset selection."""
    if k < 0 or mode not in ('near', 'far', 'random', 'full', 'baseline'):
        raise ValueError('Invalid selection')
    n = len(descriptors)
    if mode == 'baseline':
        return []
    if mode == 'full':
        return list(range(n))
    if mode == 'random':
        digest = hashlib.sha256(f'{seed}|{sample_id}'.encode()).digest()
        order = np.random.default_rng(int.from_bytes(digest[:8], 'little')).permutation(n).tolist()
    else:
        # Exact segment-to-guide distance, with mean point distance breaking ties.
        sign = 1 if mode == 'near' else -1
        order = sorted(range(n), key=lambda i: (sign * descriptors[i]['distance_to_guide_m'],
                                                sign * descriptors[i]['mean_point_distance_to_guide_m'], i))
    return sorted(order[:min(k, n)])


def support_masks(row, world, route, objects):
    """Evaluation-only per-object [N,2,23] target boundary support masks."""
    md = row['metadata']
    points = sample_at(route['gt_world'], route['exit_s'] + np.arange(3., 26.))
    centers = transform(points, md['ego_world_xy'], md['heading_rad'])
    physical = {obj['id']: obj for obj in world['physical']['elements']}
    masks = np.zeros((len(objects), 2, 23), dtype=bool)
    for side, bid in enumerate(target_boundaries(world, route)):
        xy = transform(physical[bid]['points'], md['ego_world_xy'], md['heading_rad'])
        a, v = xy[:-1], np.diff(xy, axis=0)
        u = np.clip(((centers[:, None] - a[None]) * v[None]).sum(-1) / np.maximum((v*v).sum(-1), 1e-12), 0, 1)
        projected = a[None] + u[..., None] * v[None]
        nearest = ((projected - centers[:, None]) ** 2).sum(-1).argmin(-1)
        expected = projected[np.arange(23), nearest]
        for i, (kind, obj) in enumerate(objects):
            if kind in ('lane_marking', 'road_boundary') and obj.get('source_id') == bid:
                masks[i, side] = point_polyline_distance(expected, resample(obj['points'], 32)) <= 1.25
    return masks


def summarize(rows):
    result = paired_summary(rows)
    if not rows:
        return result
    world_gains = [float(np.mean([r['ade_gain_m'] for r in rows if r['world_id'] == wid])) for wid in sorted({r['world_id'] for r in rows})]
    result.update(mean_added_tokens=float(np.mean([r['added_tokens'] for r in rows])),
                  effective_k_counts=dict(Counter(str(r['added_tokens']) for r in rows)),
                  mean_exit_supported_stations=float(np.mean([r['coverage_after'] for r in rows])),
                  mean_selected_guide_distance_m=float(np.mean([r['selected_mean_distance_m'] for r in rows])),
                  world_equal_mean_ade_gain_m=float(np.mean(world_gains)),
                  worlds_improved_over_1cm=sum(v > .01 for v in world_gains),
                  worlds_worsened_over_1cm=sum(v < -.01 for v in world_gains))
    return result


def run(args):
    supplement = args.supplement_dir
    source = json.loads((supplement / 'summary.json').read_text())
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    model, saved = load_model(source['checkpoint'], device)
    if (saved['config']['max_lines'], saved['config']['max_points']) != (32, 32):
        raise ValueError('Capacity must remain 32x32')
    dataset = EvaluationDataset(source['input'], 32, 32)
    if inspect_split(source['input']) != source['data_manifest']:
        raise ValueError('Source data changed')
    if set(source['data_manifest']['worlds']) & set(saved['splits']['train']['worlds']):
        raise ValueError('Held-out worlds required')
    pools = [json.loads(line) for line in (supplement / 'added_observations.jsonl').open()]
    with np.load(supplement / 'predictions.npz', allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in ('baseline', 'expanded', 'gt', 'guide')}
        ids = archive['ids'].tolist()
    if len(pools) != len(dataset) or ids != [p['id'] for p in pools] or ids != [dataset.raw_row(i)['id'] for i in range(len(dataset))]:
        raise ValueError('Supplement/input/prediction alignment failed')
    out = args.output_dir or supplement / 'selection_ablation'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise ValueError('Report exists; select a new output directory')
    variants = {'baseline': ('baseline', 0), 'full': ('full', 32)}
    variants.update({f'{mode}_{k}': (mode, k) for k in args.budgets for mode in ('near', 'far', 'random')})
    pending = {key: [] for key in variants}
    contexts, results, cache = [], {key: [] for key in variants}, {}
    outputs = {key: [] for key in variants}
    selection_records, candidate_records = [], []
    max_differences = {'baseline': 0., 'full': 0.}

    def flush():
        if not contexts:
            return
        predicted = {}
        with torch.inference_mode():
            for key, items in pending.items():
                batch = {k: torch.stack([b[k] for b in items]).to(device) for k in items[0]}
                predicted[key] = model.forward_batch(batch).cpu().numpy()
                if not np.isfinite(predicted[key]).all():
                    raise FloatingPointError('Nonfinite prediction')
        for j, (row, route, road, lane, fields) in enumerate(contexts):
            index = fields['index']
            for name, reference_key in [('baseline', 'baseline'), ('full', 'expanded')]:
                delta = float(np.max(np.abs(predicted[name][j] - original[reference_key][index])))
                max_differences[name] = max(delta, max_differences[name])
                if not np.allclose(predicted[name][j], original[reference_key][index], atol=1e-3, rtol=1e-4):
                    raise AssertionError(f'{name} reproduction failed at {index}: {delta}')
            gt, guide = original['gt'][index], original['guide'][index]
            baseline_metrics = metrics(row, predicted['baseline'][j], gt, guide, road, lane, route)
            for name in variants:
                value = metrics(row, predicted[name][j], gt, guide, road, lane, route)
                results[name].append({**fields, **fields['variants'][name], 'variant': name,
                                      **{f'baseline_{k}': v for k, v in baseline_metrics.items()},
                                      **{f'expanded_{k}': v for k, v in value.items()},
                                      'ade_gain_m': baseline_metrics['ade_m'] - value['ade_m']})
                results[name][-1].pop('variants')
                outputs[name].append(predicted[name][j])
        contexts.clear()
        for key in pending:
            pending[key].clear()

    for index in range(len(dataset)):
        row = dataset.raw_row(index)
        wid, nav = row['world_id'], row['navigation']
        if wid not in cache:
            world = json.loads((args.worlds_dir / f'{wid}.json').read_text())
            routes = json.loads((args.routes_dir / f'{wid}.json').read_text())
            cache[wid] = (world, routes, drivable_polygon(world), {k: target_lane_polygon(world, rt) for k, rt in routes.items()})
        world, routes, road, lanes = cache[wid]
        pool = [(obj['kind'], {k: v for k, v in obj.items() if k != 'kind'}) for obj in pools[index]['added_objects']]
        guide = original['guide'][index]
        descriptors = [describe_candidate(item, guide) for item in pool]
        base = encode_sample(row)
        if len(pool) > 32 - int(base['polyline_mask'].sum()):
            raise ValueError('Retained pool exceeds original padding capacity')
        for i, descriptor in enumerate(descriptors):
            candidate_records.append({'index': index, 'id': row['id'], 'pool_index': i, **descriptor})
        original_objects = _objects(row, 32)
        masks = support_masks(row, world, routes[nav], original_objects + pool)
        old_support = masks[:len(original_objects)].any(axis=0)
        new_support = masks[len(original_objects):]
        variant_fields = {}
        for name, (mode, k) in variants.items():
            selected = select_indices(descriptors, mode, k, row['id'], args.selection_seed)
            augmented, kept = append_observations(base, [pool[i] for i in selected], 32)
            if len(kept) != len(selected):
                raise AssertionError('Unexpected capacity truncation')
            support = old_support | (new_support[selected].any(axis=0) if selected else np.zeros_like(old_support))
            coverage = int(support.all(axis=0).sum())
            variant_fields[name] = {'added_tokens': len(selected), 'budget': k,
                                    'full_budget_available': len(pool) >= k if mode not in ('full', 'baseline') else True,
                                    'coverage_before': int(old_support.all(axis=0).sum()), 'coverage_after': coverage,
                                    'selected_mean_distance_m': float(np.mean([descriptors[i]['distance_to_guide_m'] for i in selected])) if selected else 0.}
            selection_records.append({'index': index, 'id': row['id'], 'variant': name, 'pool_indices': selected,
                                      'kind_counts': dict(Counter(descriptors[i]['kind'] for i in selected))})
            pending[name].append(augmented)
        contexts.append((row, routes[nav], road, lanes[nav],
                         {'index': index, 'id': row['id'], 'world_id': wid, 'navigation': nav,
                          'pose': row['metadata']['pose_name'], 'quality': row['metadata']['quality'],
                          'pool_size': len(pool), 'variants': variant_fields}))
        if len(contexts) == args.batch_size:
            flush()
            if (index + 1) % 160 == 0:
                print(f'{index+1}/{len(dataset)} paired samples evaluated', flush=True)
    flush()
    report = {'checkpoint': source['checkpoint'], 'input': source['input'], 'supplement_dir': str(supplement.resolve()),
              'supplement_seed': source['config']['seed'], 'selection_seed': args.selection_seed,
              'budgets': args.budgets, 'max_reproduction_difference_m': max_differences,
              'all': {name: summarize(rows) for name, rows in results.items()},
              'common_budget_subset': {name: summarize([r for r in rows if r['pool_size'] >= max(args.budgets)]) for name, rows in results.items()},
              'full_budget_subset': {name: summarize([r for r in rows if r['full_budget_available']]) for name, rows in results.items()},
              'groups': {key: {value: {name: summarize([r for r in rows if r[key] == value]) for name, rows in results.items()}
                               for value in sorted({r[key] for r in results['baseline']})} for key in ('world_id', 'navigation', 'pose', 'quality')},
              'notes': ['Only original retained candidate pool is reused; earlier range/capacity-discarded candidates are outside this experiment.',
                        'Near/Far use exact encoded line-to-guide distance; mean point distance breaks ties. Random ordering is fixed per sample.',
                        'At each K all policies use identical effective counts; larger budgets use nested candidate sets.',
                        'All original slots preserved; selected candidates retain original pool order.',
                        'GT/target IDs used for metrics only, never selection. Geometry descriptors reveal type/length/direction confounding.',
                        'No training; both spatial encoder and cross-attention can mediate contextual effects.']}
    flat = [r for rows in results.values() for r in rows]
    with (out / 'per_sample.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    with (out / 'candidates.csv').open('w', newline='') as f:
        fields = ['index', 'id', 'pool_index', 'kind', 'distance_to_guide_m',
                  'mean_point_distance_to_guide_m', 'length_m', 'mean_abs_direction_alignment', 'confidence']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(candidate_records)
    with (out / 'selections.jsonl').open('w') as f:
        for row in selection_records:
            f.write(json.dumps(row) + '\n')
    np.savez_compressed(out / 'predictions.npz', ids=np.array(ids), gt=original['gt'], guide=original['guide'],
                        **{name: np.stack(values) for name, values in outputs.items()})
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('variant  ADE  FDE  wrong-direction  lane-violation')
    for name, values in report['all'].items():
        print(name, *(round(values[key]['after'], 4) for key in ('ade_m', 'fde_m', 'wrong_direction', 'outside_target_lane')))
    print(f'Results: {out}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement-dir', type=Path, required=True)
    parser.add_argument('--worlds-dir', type=Path, required=True)
    parser.add_argument('--routes-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--budgets', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--selection-seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1 or not args.budgets or any(k < 1 or k > 32 for k in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        parser.error('Invalid batch size or budgets')
    run(args)


if __name__ == '__main__':
    main()
