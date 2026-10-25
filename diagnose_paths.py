"""Truth-overlay and residual diagnostics, exclusively after model inference.

python diagnose_paths.py --evaluation-dir outputs/training_v1/evaluation_best_val \
    --worlds-dir outputs/v4_world30_retry/worlds \
    --routes-dir outputs/v4_world30_retry/routes --indices 48 51
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import LineString, Polygon

from Dataset.dataset import _objects
from Dataset.audit_lane_visibility import transform, sample_at, target_boundaries, point_polyline_distance
from Dataset.audit_drivable import drivable_polygon
from World.intersection_generator_v4 import arm_xy
from World.road_warp import ArmWarp, deform, densify
from evaluate import EvaluationDataset
from train import inspect_split


def tangent(points):
    difference = np.gradient(points, axis=0)
    return difference / np.maximum(np.linalg.norm(difference, axis=-1, keepdims=True), 1e-9)


def components(delta, direction):
    return np.column_stack([(delta * direction).sum(-1),
                            direction[:, 0] * delta[:, 1] - direction[:, 1] * delta[:, 0]])


def to_world(points, metadata):
    h = metadata['heading_rad']
    rotation = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    return points @ rotation.T + metadata['ego_world_xy']


def target_lane_polygon(world, route):
    md = world['metadata']
    k, lane = route['exit_arm'], route['exit_lane_index']
    width, start, end = md['lane_width_m'], md['entry_distances_m'][k], md['arm_length_m']
    ring = arm_xy(k, np.array([start, end, end, start, start]),
                  -width * np.array([lane, lane, lane + 1, lane + 1, lane]))
    wm = md['arm_warp']
    warp = ArmWarp(tuple(wm['angle_deg']), tuple(wm['lateral_m']), wm['start_m'], wm['full_m'])
    polygon = Polygon(deform(densify(ring, .25), warp))
    if not polygon.is_valid:
        raise ValueError('Invalid reconstructed target lane polygon')
    return polygon


def diagnose_one(dataset, arrays, index, world, route):
    row = dataset.raw_row(index)
    md = row['metadata']
    paths = {key: value[index] for key, value in arrays.items()}
    guide, gt = paths['guide'], paths['gt']
    direction = tangent(guide)
    residuals = {key: components(value - guide, direction) for key, value in paths.items()}
    expected_gt = sample_at(route['gt_world'], md['ego_route_s_m'] + np.arange(len(gt)))
    if expected_gt is None:
        raise ValueError('GT route too short for closure check')
    expected_gt = transform(expected_gt, md['ego_world_xy'], md['heading_rad'])
    road = drivable_polygon(world)
    lane = target_lane_polygon(world, route)
    exit_station = route['exit_s'] - md['ego_route_s_m']
    post_exit = np.arange(len(gt)) >= exit_station + 3
    ids = target_boundaries(world, route)
    physical = {e['id']: e for e in world['physical']['elements']}
    encoded_objects = _objects(row, dataset.max_lines)
    source_ids = [obj.get('source_id') for kind, obj in encoded_objects if kind in ('lane_marking', 'road_boundary')]
    stats = {}
    for key, path in paths.items():
        xy_world = to_world(path, md)
        error = np.linalg.norm(path - gt, axis=-1)
        correction = np.linalg.norm(path - guide, axis=-1)
        heading = components(tangent(path), direction)
        heading_deg = np.degrees(np.arctan2(heading[:, 1], heading[:, 0]))
        eligible_path = xy_world[post_exit]
        stats[key] = {
            'ade_m': float(error[1:].mean()), 'fde_m': float(error[-1]),
            'mean_nearest_gt_distance_m': float(point_polyline_distance(path[1:], gt).mean()),
            'mean_abs_longitudinal_residual_m': float(np.abs(residuals[key][1:, 0]).mean()),
            'mean_abs_lateral_residual_m': float(np.abs(residuals[key][1:, 1]).mean()),
            'end_residual_tangent_normal_m': residuals[key][-1].tolist(),
            'mean_abs_heading_change_from_guide_deg': float(np.abs(heading_deg[1:]).mean()),
            'max_abs_heading_change_from_guide_deg': float(np.abs(heading_deg[1:]).max()),
            'first_station_correction_over_10cm': next((i for i in range(1, len(path)) if correction[i] > .1), None),
            'outside_road_length_m_tolerance_8cm': float(LineString(xy_world).difference(road.buffer(.08)).length),
            'outside_target_lane_length_m_post_exit_tolerance_8cm': (
                float(LineString(eligible_path).difference(lane.buffer(.08)).length) if len(eligible_path) > 1 else None),
        }
    fov = md['vcs_fov_m']
    exit_stations = route['exit_s'] + np.arange(3., 26.)
    exit_centers = transform(sample_at(route['gt_world'], exit_stations), md['ego_world_xy'], md['heading_rad'])
    in_fov = ((exit_centers[:, 0] >= -fov['rear']) & (exit_centers[:, 0] <= fov['forward'])
              & (exit_centers[:, 1] >= -fov['right']) & (exit_centers[:, 1] <= fov['left']))
    desired_lat, actual_lat = residuals['gt'][:, 1], residuals['pred'][:, 1]
    meaningful = np.abs(desired_lat) > .05
    meaningful[0] = False
    wrong = meaningful & (desired_lat * actual_lat < 0)
    boundary_stats = {}
    for bid in ids:
        boundary = transform(physical[bid]['points'], md['ego_world_xy'], md['heading_rad'])
        boundary_stats[bid] = {'present_in_encoded_observation': bid in source_ids,
                               'mean_gt_distance_post_exit_m': float(point_polyline_distance(gt[post_exit], boundary).mean()) if post_exit.any() else None,
                               'mean_pred_distance_post_exit_m': float(point_polyline_distance(paths['pred'][post_exit], boundary).mean()) if post_exit.any() else None}
    report = {'index': index, 'id': row['id'], 'world_id': row['world_id'],
              'gt_route_closure_max_m': float(np.linalg.norm(gt - expected_gt, axis=-1).max()),
              'saved_gt_vs_row_max_m': float(np.abs(gt - np.asarray(row['gt_path'])).max()),
              'saved_guide_vs_row_max_m': float(np.abs(guide - np.asarray(row['reference_line'])).max()),
              'exit_station_from_ego_m': float(exit_station),
              'target_exit_lane_index': route['exit_lane_index'], 'target_boundaries': boundary_stats,
              'exit_center_stations_3_to_25m_in_fov': int(in_fov.sum()),
              'exit_center_stations_checked': len(in_fov),
              'wrong_lateral_correction_stations': int(wrong.sum()),
              'meaningful_lateral_target_stations': int(meaningful.sum()), 'paths': stats,
              'limits': ['Source IDs indicate encoded fragment presence, not full boundary visibility.',
                         'Target lane is reconstructed from generator geometry; checks exclude first 3m after exit.',
                         'Residuals compare equal sample indices in guide frame, not exact Frenet station matching.',
                         'Road/lane checks apply to path centerlines, not vehicle footprints.']}
    return report, paths, residuals, direction


def plot_diagnostics(dataset, index, world, route, paths, residuals, direction, report, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    row = dataset.raw_row(index)
    md = row['metadata']
    ids = set(target_boundaries(world, route))
    transform_world = lambda p: transform(p, md['ego_world_xy'], md['heading_rad'])
    encoded = dataset[index]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    colors = {'gt': '#159447', 'guide': '#ea9900', 'pred': '#db3030', 'empty_bev': '#a050bf'}
    for ax, limit in zip(axes, (81, 31)):
        used = set()
        for e in world['physical']['elements']:
            p = transform_world(e['points'])
            color, label, width = ('#81b7d0', 'Truth target boundaries', 2) if e['id'] in ids else ('#cccccc', 'Truth geometry', .8)
            if e['type'] in ('crosswalk', 'stop_line'):
                p = np.vstack([p, p[:1]])
            ax.plot(-p[:, 1], p[:, 0], '--', color=color, lw=width, label=label if label not in used else None, zorder=1)
            used.add(label)
        for j in range(dataset.max_lines):
            mask = encoded['point_mask'][j]
            if not mask.any():
                continue
            p = encoded['bev_points'][j, mask, :2].numpy()
            kind = encoded['polyline_attr'][j, :4].argmax().item()
            if kind >= 2:
                p = np.vstack([p, p[:1]])
            label = 'Observed model input'
            ax.plot(-p[:, 1], p[:, 0], color='#353535', lw=1.3, label=label if label not in used else None)
            used.add(label)
        for key, p in paths.items():
            ax.plot(-p[:, 1], p[:, 0], color=colors[key], lw=2,
                    ls='--' if key == 'guide' else ':' if key == 'empty_bev' else '-', label=key)
        stations = np.arange(10, min(limit, len(paths['gt'])), 10)
        for i in stations:
            p = paths['gt'][i]
            ax.text(-p[1], p[0], f'{i}m', fontsize=8, color=colors['gt'])
        fov = md['vcs_fov_m']
        ax.add_patch(Rectangle((-fov['left'], -fov['rear']), fov['left'] + fov['right'], fov['forward'] + fov['rear'],
                               fill=False, ec='#999999', ls=':', label='FOV'))
        extent = np.concatenate([p[:limit] for p in paths.values()])
        ax.set_xlim(-extent[:, 1].max() - 6, -extent[:, 1].min() + 6)
        ax.set_ylim(extent[:, 0].min() - 6, extent[:, 0].max() + 6)
        ax.scatter([0], [0], marker='^', color='black')
        ax.set_aspect('equal', adjustable='box')
        ax.set(xlabel='Right (-ego y), m', ylabel='Forward (ego x), m', title='Full horizon' if limit == 81 else 'First 30m extent')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(row['id'] + '\nDashed pale lines = truth only; dark lines = model observations', fontsize=11)
    fig.tight_layout()
    fig.savefig(output / f'{index:06d}_truth.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for key in ('gt', 'pred', 'empty_bev'):
        if key not in paths:
            continue
        label = 'Required correction (GT-guide)' if key == 'gt' else key + ' - guide'
        for axis in (0, 1):
            axes[0, axis].plot(residuals[key][:, axis], color=colors[key], label=label)
        error = np.linalg.norm(paths[key] - paths['gt'], axis=-1)
        axes[1, 0].plot(error, color=colors[key], label=key)
        local_heading = components(tangent(paths[key]), direction)
        angle = np.degrees(np.arctan2(local_heading[:, 1], local_heading[:, 0]))
        axes[1, 1].plot(angle, color=colors[key], label=key)
    axes[1, 0].plot(np.linalg.norm(paths['guide'] - paths['gt'], axis=-1), '--', color=colors['guide'], label='guide')
    for ax, title, ylabel in zip(axes.flat,
                               ('Longitudinal residual', 'Lateral residual (+left)', 'Pointwise error to GT', 'Heading difference from guide'),
                               ('Metres', 'Metres', 'Metres', 'Degrees')):
        ax.axhline(0, color='#aaaaaa', lw=.7)
        if 0 <= report['exit_station_from_ego_m'] <= 80:
            ax.axvline(report['exit_station_from_ego_m'], color='black', ls=':', label='Exit')
        ax.set(title=title, xlabel='Sample index (GT station in metres)', ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(row['id'] + '\nResiduals use guide local tangent/left-normal at equal sample index', fontsize=11)
    fig.tight_layout()
    fig.savefig(output / f'{index:06d}_residuals.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--worlds-dir', type=Path, required=True)
    parser.add_argument('--routes-dir', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.evaluation_dir / 'summary.json').read_text())
    saved = torch.load(summary['checkpoint'], map_location='cpu', weights_only=False)
    dataset = EvaluationDataset(summary['input'], saved['config']['max_lines'], saved['config']['max_points'])
    if inspect_split(summary['input']) != summary['data_manifest']:
        raise ValueError('Source samples changed')
    with np.load(args.evaluation_dir / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in ('gt', 'guide', 'pred', 'empty_bev') if k in archive}
        ids = archive['ids'].tolist()
    if ids != [dataset.raw_row(i)['id'] for i in range(len(dataset))]:
        raise ValueError('Prediction alignment mismatch')
    out = args.output_dir or args.evaluation_dir / 'truth_diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'report.json').exists():
        raise ValueError('Report exists; choose a new output directory')
    import matplotlib
    matplotlib.use('Agg')
    reports = []
    for index in args.indices:
        if not 0 <= index < len(dataset):
            raise ValueError(f'Index out of range: {index}')
        row = dataset.raw_row(index)
        world = json.loads((args.worlds_dir / f'{row["world_id"]}.json').read_text())
        route = json.loads((args.routes_dir / f'{row["world_id"]}.json').read_text())[row['navigation']]
        report, paths, residuals, direction = diagnose_one(dataset, arrays, index, world, route)
        plot_diagnostics(dataset, index, world, route, paths, residuals, direction, report, out)
        np.savez_compressed(out / f'{index:06d}_residuals.npz', **residuals)
        reports.append(report)
        print(json.dumps(report, indent=2))
    (out / 'report.json').write_text(json.dumps(reports, indent=2, allow_nan=False) + '\n')
    print(f'Diagnostics: {out}')


if __name__ == '__main__':
    main()
