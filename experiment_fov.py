"""Paired FOV supplementation: frozen checkpoint, original inputs, quality, 32 slots.

Only NEW regions outside the original FOV can supply added observations. Existing
encoded slots are bit-identical; supplementary objects use remaining padding.
This is a diagnostic intervention, not a complete large-FOV sensor simulation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import csv
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import LineString, Polygon, box

from Dataset.dataset import encode_sample, _objects, resample, point_features, attributes
from Dataset.audit_lane_visibility import transform, sample_at, target_boundaries, point_polyline_distance
from Dataset.audit_drivable import drivable_polygon
from Scenario.build_dataset_v4 import observe, PerceptionQuality
from Scenario.build_dataset import NAVS, pose_seed_index
from evaluate import EvaluationDataset, load_model
from diagnose_paths import to_world, target_lane_polygon
from analyze_corrections import classify_correction
from train import inspect_split


@dataclass(frozen=True)
class ExperimentConfig:
    forward_m: float = 100.
    left_m: float = 60.
    right_m: float = 60.
    seed: int = 724
    batch_size: int = 16
    device: str = 'auto'
    coverage_distance_m: float = 1.25
    coverage_min_station_gain: int = 5
    examples: int = 3

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in (self.forward_m, self.left_m, self.right_m, self.coverage_distance_m)):
            raise ValueError('FOV extents and coverage tolerance must be positive')
        if self.batch_size < 1 or self.coverage_min_station_gain < 1 or self.examples < 0:
            raise ValueError('Invalid counts')


def fov_polygon(fov):
    return box(-fov['rear'], -fov['right'], fov['forward'], fov['left'])


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'LineString':
        return [geometry]
    return [part for child in getattr(geometry, 'geoms', []) for part in line_parts(child)]


def supplement_candidates(row, world, quality, cfg):
    md = row['metadata']
    old = md['vcs_fov_m']
    expanded = {'forward': max(old['forward'], cfg.forward_m), 'rear': old['rear'],
                'left': max(old['left'], cfg.left_m), 'right': max(old['right'], cfg.right_m)}
    q = replace(quality, forward_m=expanded['forward'], rear_m=expanded['rear'],
                left_m=expanded['left'], right_m=expanded['right'])
    # Existing observations are not regenerated. Separate reproducible RNG only
    # governs added regions, with exactly the original quality's error parameters.
    seed = [cfg.seed, int(world['metadata']['seed']), NAVS.index(row['navigation']),
            pose_seed_index(md['pose_name']), *md['quality'].encode()]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    h = md['heading_rad']
    observed = observe(world, np.asarray(md['ego_world_xy']), np.array([np.cos(h), np.sin(h)]), q, rng, max_lines=None)
    old_poly, expanded_poly = fov_polygon(old), fov_polygon(expanded)
    candidates = []
    for kind, objects in zip(('lane_marking', 'road_boundary', 'stop_line', 'crosswalk'), observed):
        for obj in objects:
            xy = np.asarray(obj['points'])
            if kind in ('lane_marking', 'road_boundary'):
                # Geometrically clip noisy points as well; never refill an old-FOV miss.
                geom = LineString(xy).intersection(expanded_poly).difference(old_poly)
                for segment in line_parts(geom):
                    if segment.length > 1.3:
                        candidates.append((kind, {**obj, 'points': np.asarray(segment.coords).tolist()}))
            else:
                poly = Polygon(xy)
                # A full four-corner box must lie outside old FOV and within new
                # FOV; do not turn partially visible old boxes into new detections.
                if poly.is_valid and poly.intersection(old_poly).area < 1e-9 and expanded_poly.covers(poly):
                    candidates.append((kind, obj))
    # No GT/nav-target association is used in selection. All types rank by range.
    candidates.sort(key=lambda item: (float(np.min(np.linalg.norm(item[1]['points'], axis=1))),
                                      item[0], item[1].get('source_id', item[1].get('id', ''))))
    return candidates, expanded


def append_observations(base, candidates, max_points):
    result = {key: value.clone() for key, value in base.items()}
    free = (~base['polyline_mask']).nonzero().flatten().tolist()
    kept = candidates[:len(free)]
    for slot, (kind, obj) in zip(free, kept):
        closed = kind in ('stop_line', 'crosswalk')
        sampled = resample(obj['points'], max_points, closed=closed)
        n = len(sampled)
        result['bev_points'][slot, :n] = torch.from_numpy(point_features(sampled, closed=closed))
        result['point_mask'][slot, :n] = True
        result['polyline_attr'][slot] = torch.from_numpy(attributes(obj, kind))
    result['polyline_mask'] = result['point_mask'].any(-1)
    original_slots = base['polyline_mask']
    for key in ('bev_points', 'point_mask', 'polyline_attr'):
        if not torch.equal(base[key][original_slots], result[key][original_slots]):
            raise AssertionError('An original observation changed')
    for key in ('guide', 'gt_path', 'navigation'):
        if not torch.equal(base[key], result[key]):
            raise AssertionError('A non-BEV input changed')
    return result, kept


def exit_coverage(row, world, route, objects, distance):
    """Stations where BOTH target boundaries have observed geometric support.

    Uses evaluation-only identity and nearest boundary projection. No FOV-center
    mask: actual observation support is compared at identical world stations.
    """
    md = row['metadata']
    stations = sample_at(route['gt_world'], route['exit_s'] + np.arange(3., 26.))
    centers = transform(stations, md['ego_world_xy'], md['heading_rad'])
    physical = {obj['id']: obj for obj in world['physical']['elements']}
    sides = []
    for bid in target_boundaries(world, route):
        boundary = transform(physical[bid]['points'], md['ego_world_xy'], md['heading_rad'])
        a, v = boundary[:-1], np.diff(boundary, axis=0)
        u = np.clip(((centers[:, None] - a[None]) * v[None]).sum(-1) / np.maximum((v*v).sum(-1), 1e-12), 0, 1)
        projection = a[None] + u[..., None] * v[None]
        nearest = ((projection - centers[:, None]) ** 2).sum(-1).argmin(-1)
        expected = projection[np.arange(len(centers)), nearest]
        supported = np.zeros(len(centers), dtype=bool)
        for kind, obj in objects:
            if kind in ('lane_marking', 'road_boundary') and obj.get('source_id') == bid:
                # Measure exactly the same 32-point representation the model sees.
                observed = resample(obj['points'], 32)
                supported |= point_polyline_distance(expected, observed) <= distance
        sides.append(supported)
    return int((sides[0] & sides[1]).sum())


def metrics(row, path, gt, guide, road, lane, route):
    distance = np.linalg.norm(path - gt, axis=-1)
    correction = classify_correction(guide, gt, path)
    xy = to_world(path, row['metadata'])
    post_exit = np.arange(len(gt)) >= route['exit_s'] - row['metadata']['ego_route_s_m'] + 3
    outside = float(LineString(xy).difference(road.buffer(.08)).length)
    lane_outside = float(LineString(xy[post_exit]).difference(lane.buffer(.08)).length) if post_exit.sum() > 1 else 0.
    return {'ade_m': float(distance[1:].mean()), 'fde_m': float(distance[-1]),
            'wrong_direction': correction['wrong_direction'], 'overshoot': correction['overshoot'],
            'outside_road': outside > .1, 'outside_target_lane': lane_outside > .1}


def paired_summary(rows):
    if not rows:
        return {'samples': 0}
    result = {'samples': len(rows), 'worlds': len({r['world_id'] for r in rows})}
    for key in ('ade_m', 'fde_m', 'wrong_direction', 'overshoot', 'outside_road', 'outside_target_lane'):
        before = float(np.mean([r['baseline_' + key] for r in rows]))
        after = float(np.mean([r['expanded_' + key] for r in rows]))
        result[key] = {'before': before, 'after': after, 'delta_after_minus_before': after - before}
    gains = np.array([r['ade_gain_m'] for r in rows])
    result['ade_improved_samples'] = int((gains > .01).sum())
    result['ade_worsened_samples'] = int((gains < -.01).sum())
    result['ade_within_1cm_samples'] = int((np.abs(gains) <= .01).sum())
    return result


def plot_pairs(dataset, additions, records, arrays, cfg, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    changed = sorted([r for r in records if r['input_added']], key=lambda r: r['ade_gain_m'])
    chosen = {}
    for label, group in [('worsened', [r for r in changed if r['ade_gain_m'] < -.01][:cfg.examples]),
                         ('improved', [r for r in reversed(changed) if r['ade_gain_m'] > .01][:cfg.examples])]:
        for record in group:
            chosen[record['index']] = label
    (output / 'samples').mkdir(exist_ok=True)
    for index, label in chosen.items():
        record = records[index]
        encoded = dataset[index]
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        for ax, limit in zip(axes, (81, 31)):
            used = set()
            for slot in range(32):
                mask = encoded['point_mask'][slot]
                if not mask.any():
                    continue
                xy = encoded['bev_points'][slot, mask, :2].numpy()
                if encoded['polyline_attr'][slot, 9] > .5:
                    xy = np.vstack([xy, xy[:1]])
                ax.plot(-xy[:, 1], xy[:, 0], color='#555555', lw=1.,
                        label='Original observations' if 'original' not in used else None)
                used.add('original')
            for obj in additions[index]['added_objects']:
                xy = resample(obj['points'], 32, closed=obj['kind'] in ('stop_line', 'crosswalk'))
                if obj['kind'] in ('stop_line', 'crosswalk'):
                    xy = np.vstack([xy, xy[:1]])
                ax.plot(-xy[:, 1], xy[:, 0], color='#159fdb', lw=1.5,
                        label='Added observations' if 'added' not in used else None)
                used.add('added')
            for key, color, style in [('gt', '#159447', '-'), ('guide', '#ea9900', '--'),
                                       ('baseline', '#db3030', '-'), ('expanded', '#a050bf', '-')]:
                xy = arrays[key][index]
                ax.plot(-xy[:, 1], xy[:, 0], color=color, ls=style, lw=2., label=key)
            extent = np.concatenate([a[index, :limit] for a in arrays.values()])
            ax.set_xlim(-extent[:, 1].max() - 5, -extent[:, 1].min() + 5)
            ax.set_ylim(extent[:, 0].min() - 5, extent[:, 0].max() + 5)
            ax.set_aspect('equal', adjustable='box')
            ax.scatter([0], [0], marker='^', color='black')
            ax.set(xlabel='Right (-ego y), m', ylabel='Forward (ego x), m', title='Full horizon' if limit == 81 else 'First 30m extent')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{record["id"]} | {label}\nADE {record["baseline_ade_m"]:.3f} -> {record["expanded_ade_m"]:.3f}m; '
                     f'added tokens {record["added_tokens"]}; exit support {record["coverage_before"]} -> {record["coverage_after"]}/23')
        fig.tight_layout()
        fig.savefig(output / 'samples' / f'{label}_{index:06d}.png', dpi=140)
        plt.close(fig)


def run(args):
    cfg = ExperimentConfig(**json.loads(args.config.read_text()))
    summary = json.loads((args.evaluation_dir / 'summary.json').read_text())
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else cfg.device)
    model, saved = load_model(summary['checkpoint'], device)
    if saved['config']['max_lines'] != 32 or saved['config']['max_points'] != 32:
        raise ValueError('Experiment A fixes capacity at 32 objects x 32 points')
    dataset = EvaluationDataset(summary['input'], 32, 32)
    if inspect_split(summary['input']) != summary['data_manifest']:
        raise ValueError('Source input changed')
    if set(summary['data_manifest']['worlds']) & set(saved['splits']['train']['worlds']):
        raise ValueError('Experiment must use held-out worlds')
    source_config = json.loads((args.worlds_dir.parent / 'config.json').read_text())
    qualities = {key: PerceptionQuality(**value) for key, value in source_config['qualities'].items()}
    output = args.output_dir or args.evaluation_dir / 'expanded_fov_A'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise ValueError('Report exists; choose a new output directory')
    cache, records, predictions, baseline_predictions, additions = {}, [], [], [], []
    pending_base, pending_aug, pending_context = [], [], []
    def flush():
        if not pending_base:
            return
        with torch.inference_mode():
            batches = [{k: torch.stack([b[k] for b in values]).to(device) for k in pending_base[0]}
                       for values in (pending_base, pending_aug)]
            before = model.forward_batch(batches[0]).cpu().numpy()
            after = model.forward_batch(batches[1]).cpu().numpy()
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise FloatingPointError('Nonfinite model output')
        for j, context in enumerate(pending_context):
            row, route, road, lane, fields = context
            gt, guide = np.asarray(row['gt_path']), np.asarray(row['reference_line'])
            b = metrics(row, before[j], gt, guide, road, lane, route)
            a = metrics(row, after[j], gt, guide, road, lane, route)
            records.append({**fields, **{'baseline_' + k: v for k, v in b.items()},
                            **{'expanded_' + k: v for k, v in a.items()}, 'ade_gain_m': b['ade_m'] - a['ade_m']})
            baseline_predictions.append(before[j])
            predictions.append(after[j])
        pending_base.clear(); pending_aug.clear(); pending_context.clear()
    for i in range(len(dataset)):
        row = dataset.raw_row(i)
        wid = row['world_id']
        if wid not in cache:
            world = json.loads((args.worlds_dir / f'{wid}.json').read_text())
            routes = json.loads((args.routes_dir / f'{wid}.json').read_text())
            cache[wid] = (world, routes, drivable_polygon(world),
                          {k: target_lane_polygon(world, r) for k, r in routes.items()})
        world, routes, road, lanes = cache[wid]
        route = routes[row['navigation']]
        base = encode_sample(row)
        candidates, fov = supplement_candidates(row, world, qualities[row['metadata']['quality']], cfg)
        augmented, kept = append_observations(base, candidates, 32)
        original_objects = _objects(row, 32)
        coverage_before = exit_coverage(row, world, route, original_objects, cfg.coverage_distance_m)
        coverage_after = exit_coverage(row, world, route, original_objects + kept, cfg.coverage_distance_m)
        fields = {'index': i, 'id': row['id'], 'world_id': wid, 'navigation': row['navigation'],
                  'pose': row['metadata']['pose_name'], 'quality': row['metadata']['quality'],
                  'original_tokens': int(base['polyline_mask'].sum()), 'candidate_fragments': len(candidates),
                  'added_tokens': len(kept), 'truncated_candidates': len(candidates) - len(kept),
                  'input_added': bool(kept), 'coverage_before': coverage_before, 'coverage_after': coverage_after,
                  'coverage_increased': coverage_after - coverage_before >= cfg.coverage_min_station_gain}
        additions.append({'id': row['id'], 'fov': fov,
                          'added_objects': [{'kind': k, **obj} for k, obj in kept]})
        pending_base.append(base); pending_aug.append(augmented)
        pending_context.append((row, route, road, lanes[row['navigation']], fields))
        if len(pending_base) >= cfg.batch_size:
            flush()
            if len(records) % 160 == 0:
                print(f'{len(records)}/{len(dataset)} samples evaluated', flush=True)
    flush()
    before, after = np.stack(baseline_predictions), np.stack(predictions)
    with np.load(args.evaluation_dir / 'predictions.npz', allow_pickle=False) as original:
        if original['ids'].tolist() != [r['id'] for r in records]:
            raise ValueError('Saved prediction IDs do not match')
        baseline_difference = float(np.max(np.abs(before - original['pred'])))
        if not np.allclose(before, original['pred'], atol=1e-3, rtol=1e-4):
            raise AssertionError(f'Baseline not reproduced: max delta {baseline_difference}')
        gt, guide = original['gt'].copy(), original['guide'].copy()
    subsets = {'all': records, 'input_added': [r for r in records if r['input_added']],
               'exit_coverage_increased': [r for r in records if r['coverage_increased']],
               'input_unchanged': [r for r in records if not r['input_added']]}
    result = {'config': cfg.__dict__, 'checkpoint': summary['checkpoint'], 'input': summary['input'],
              'data_manifest': summary['data_manifest'], 'baseline_max_coordinate_difference_m': baseline_difference,
              'original_slots_preserved': True, 'no_training': True,
              'subsets': {key: paired_summary(value) for key, value in subsets.items()},
              'groups': {key: {v: paired_summary([r for r in records if r[key] == v])
                               for v in sorted({r[key] for r in records})} for key in ('navigation', 'pose', 'quality')},
              'notes': ['Original encoded slots stay bit-identical; added observations only fill padding.',
                        'Same noise/miss/break parameters, separately sampled supplement outside old FOV.',
                        'Full boxes intersecting old FOV are excluded; no old misses are refilled.',
                        'Supplement selection uses range only, never GT/target identity.',
                        f'Coverage uses target IDs exclusively for evaluation; both sides within {cfg.coverage_distance_m}m at exit stations 3..25m.',
                        'Inference-time input shift; single supplement seed; not a retrained FOV comparison.']}
    with (output / 'per_sample.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (output / 'added_observations.jsonl').open('w') as f:
        for entry in additions:
            f.write(json.dumps(entry) + '\n')
    np.savez_compressed(output / 'predictions.npz', ids=np.array([r['id'] for r in records]),
                        baseline=before, expanded=after, gt=gt, guide=guide)
    if cfg.examples:
        plot_pairs(dataset, additions, records, {'baseline': before, 'expanded': after, 'gt': gt, 'guide': guide}, cfg, output)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['subsets'], indent=2))
    print(f'Results: {output}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--worlds-dir', type=Path, required=True)
    parser.add_argument('--routes-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/expanded_fov_A.json'))
    parser.add_argument('--output-dir', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
