"""Aggregate guide-frame correction errors from saved evaluation predictions.

No inference or training. Categories are descriptive thresholds, not causal labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import LineString

from Dataset.dataset import _objects
from Dataset.audit_drivable import drivable_polygon
from Dataset.audit_lane_visibility import sample_at, transform, target_boundaries
from diagnose_paths import components, tangent, to_world, target_lane_polygon
from evaluate import EvaluationDataset
from train import inspect_split


KINDS = ('overshoot', 'undershoot', 'wrong_direction', 'unnecessary')


def classify_correction(guide, gt, pred, tolerance=.1, min_fraction=.25):
    """Ignore origin. Each lateral station belongs to at most one category.

    Project desired correction d and actual correction a into the guide frame.
    Meaningful target: |d|>tol. Wrong: a*sign(d)<-tol; overshoot: a*sign(d)>
    |d|+tol; undershoot: -tol<=a*sign(d)<|d|-tol. Unnecessary: |d|<=tol and
    |a|>tol. A sample flag requires >=min_fraction of ALL future stations.
    """
    if tolerance <= 0 or not 0 < min_fraction <= 1:
        raise ValueError('Invalid classification thresholds')
    direction = tangent(guide)
    desired = components(gt - guide, direction)[1:]
    actual = components(pred - guide, direction)[1:]
    d, a = desired[:, 1], actual[:, 1]
    meaningful = np.abs(d) > tolerance
    aligned = a * np.sign(d)
    masks = {
        'wrong_direction': meaningful & (aligned < -tolerance),
        'overshoot': meaningful & (aligned > np.abs(d) + tolerance),
        'undershoot': meaningful & (aligned >= -tolerance) & (aligned < np.abs(d) - tolerance),
        'unnecessary': ~meaningful & (np.abs(a) > tolerance),
    }
    error = actual - desired
    mse = np.mean(error ** 2, axis=0)
    share = float(mse[0] / mse.sum()) if mse.sum() > 1e-15 else 0.
    contributions = {key: float(np.mean(np.abs(error[:, 1]) * mask)) for key, mask in masks.items()}
    dominant = max(contributions, key=contributions.get)
    if contributions[dominant] < tolerance:
        dominant = 'small_or_mixed_lateral_error'
    result = {'lateral_dominant': dominant, 'longitudinal_mse_share': share,
              'mean_abs_longitudinal_error_m': float(np.abs(error[:, 0]).mean()),
              'mean_abs_lateral_error_m': float(np.abs(error[:, 1]).mean()),
              'mean_abs_required_lateral_m': float(np.abs(d).mean()),
              'mean_abs_actual_lateral_m': float(np.abs(a).mean()),
              'meaningful_target_station_fraction': float(meaningful.mean())}
    for key, mask in masks.items():
        result[key + '_station_fraction'] = float(mask.mean())
        result[key + '_error_contribution_m'] = contributions[key]
        result[key] = bool(mask.mean() >= min_fraction)
    result['longitudinal_dominant'] = bool(share >= .7 and result['mean_abs_longitudinal_error_m'] > tolerance)
    result['error_dominant'] = 'longitudinal' if result['longitudinal_dominant'] else dominant
    return result


def summarize(rows):
    n = len(rows)
    if not n:
        return {'samples': 0}
    def count(key):
        c = sum(bool(r[key]) for r in rows)
        return {'count': c, 'pct': 100 * c / n}
    return {'samples': n, 'worlds': len({r['world_id'] for r in rows}),
            'flags': {key: count(key) for key in (*KINDS, 'longitudinal_dominant', 'regressed', 'outside_road', 'outside_target_lane')},
            'dominant_error_counts': {key: sum(r['error_dominant'] == key for r in rows)
                                      for key in sorted({r['error_dominant'] for r in rows})},
            'ade_m': float(np.mean([r['ade_m'] for r in rows])),
            'guide_ade_m': float(np.mean([r['guide_ade_m'] for r in rows])),
            'gt_closure_max_m': max(r['gt_closure_max_m'] for r in rows)}


def analyze(args):
    summary = json.loads((args.evaluation_dir / 'summary.json').read_text())
    saved = torch.load(summary['checkpoint'], map_location='cpu', weights_only=False)
    dataset = EvaluationDataset(summary['input'], saved['config']['max_lines'], saved['config']['max_points'])
    if inspect_split(summary['input']) != summary['data_manifest']:
        raise ValueError('Source input changed')
    with np.load(args.evaluation_dir / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in ('guide', 'gt', 'pred')}
        ids = archive['ids'].tolist()
    if ids != [dataset.raw_row(i)['id'] for i in range(len(dataset))]:
        raise ValueError('Prediction IDs mismatch')
    out = args.output_dir or args.evaluation_dir / 'correction_statistics'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise ValueError('Report exists; choose a new --output-dir')
    cache, rows = {}, []
    for i in range(len(dataset)):
        row = dataset.raw_row(i)
        wid, nav, md = row['world_id'], row['navigation'], row['metadata']
        if wid not in cache:
            world = json.loads((args.worlds_dir / f'{wid}.json').read_text())
            routes = json.loads((args.routes_dir / f'{wid}.json').read_text())
            cache[wid] = (world, routes, drivable_polygon(world),
                          {k: target_lane_polygon(world, r) for k, r in routes.items()})
        world, routes, road, lanes = cache[wid]
        route = routes[nav]
        guide, gt, pred = (arrays[k][i] for k in ('guide', 'gt', 'pred'))
        result = classify_correction(guide, gt, pred, args.tolerance_m, args.min_station_fraction)
        ade = float(np.linalg.norm(pred[1:] - gt[1:], axis=1).mean())
        guide_ade = float(np.linalg.norm(guide[1:] - gt[1:], axis=1).mean())
        expected = sample_at(route['gt_world'], md['ego_route_s_m'] + np.arange(len(gt)))
        if expected is None:
            raise ValueError('GT route too short')
        expected = transform(expected, md['ego_world_xy'], md['heading_rad'])
        closure = float(np.linalg.norm(expected - gt, axis=1).max())
        if closure > .01:
            raise ValueError(f'{row["id"]}: GT/world mismatch {closure} m')
        xy = to_world(pred, md)
        outside_length = float(LineString(xy).difference(road.buffer(.08)).length)
        post = np.arange(len(gt)) >= route['exit_s'] - md['ego_route_s_m'] + 3
        lane_outside = float(LineString(xy[post]).difference(lanes[nav].buffer(.08)).length) if post.sum() > 1 else 0.
        source_ids = {obj.get('source_id') for kind, obj in _objects(row, dataset.max_lines)
                      if kind in ('lane_marking', 'road_boundary')}
        boundary_count = sum(bid in source_ids for bid in target_boundaries(world, route))
        rows.append({'index': i, 'id': row['id'], 'world_id': wid, 'navigation': nav,
                     'pose': md['pose_name'], 'quality': md['quality'], **result,
                     'ade_m': ade, 'guide_ade_m': guide_ade, 'regressed': ade > guide_ade + 1e-6,
                     'gt_closure_max_m': closure, 'target_boundary_sources_present': boundary_count,
                     'outside_road_length_m': outside_length, 'outside_road': outside_length > .1,
                     'outside_target_lane_length_m': lane_outside, 'outside_target_lane': lane_outside > .1})
    groups = {}
    for keys in (('navigation',), ('pose',), ('quality',), ('world_id',), ('navigation', 'pose'),
                 ('navigation', 'quality'), ('target_boundary_sources_present',)):
        name = '+'.join(keys)
        groups[name] = {}
        for values in sorted({tuple(r[k] for k in keys) for r in rows}):
            group = [r for r in rows if tuple(r[k] for k in keys) == values]
            groups[name]['/'.join(map(str, values))] = {'all': summarize(group),
                                                       'regressed': summarize([r for r in group if r['regressed']])}
    sensitivity = {}
    for tol in (.05, .1, .2):
        alternative = [{**r, **classify_correction(arrays['guide'][i], arrays['gt'][i], arrays['pred'][i], tol,
                                                  args.min_station_fraction)} for i, r in enumerate(rows)]
        sensitivity[str(tol)] = {'all': summarize(alternative),
                                 'regressed': summarize([r for r in alternative if r['regressed']])}
    report = {'thresholds': {'station_error_tolerance_m': args.tolerance_m,
                             'sample_flag_min_fraction_of_all_80_future_points': args.min_station_fraction,
                             'longitudinal_mse_share': .7, 'regression_tolerance_m': 1e-6,
                             'geometry_buffer_m': .08, 'geometry_outside_length_threshold_m': .1},
              'all': summarize(rows), 'regressed': summarize([r for r in rows if r['regressed']]),
              'groups': groups, 'threshold_sensitivity': sensitivity,
              'notes': ['Flags may overlap within a sample; station lateral categories are exclusive.',
                        'Undershoot may still improve guide; flags are not synonymous with regression.',
                        'Residuals use equal-index guide tangent/normal, not exact Frenet reprojection.',
                        'Dominant lateral category uses largest full-horizon absolute-error contribution >= tolerance.',
                        'Boundary-source count measures retained fragments, not full visibility.',
                        'Only 3 independent worlds; sample proportions are descriptive, not statistical confidence.',
                        'Geometry checks use reconstructed V4 centerline legality, not vehicle footprint or traffic rules.']}
    with (out / 'per_sample.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / 'grouped.csv').open('w', newline='') as f:
        fields = ['dimension', 'group', 'subset', 'samples', *KINDS, 'longitudinal_dominant', 'outside_road']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dimension, entries in groups.items():
            for name, subsets in entries.items():
                for subset, stats in subsets.items():
                    writer.writerow({'dimension': dimension, 'group': name, 'subset': subset,
                                     'samples': stats['samples'], **{key: stats.get('flags', {}).get(key, {}).get('count', 0)
                                                                  for key in fields[4:]}})
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('all', 'regressed')}, indent=2))
    print(f'Report: {out}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--worlds-dir', type=Path, required=True)
    parser.add_argument('--routes-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--tolerance-m', type=float, default=.1)
    parser.add_argument('--min-station-fraction', type=float, default=.25)
    analyze(parser.parse_args())


if __name__ == '__main__':
    main()
