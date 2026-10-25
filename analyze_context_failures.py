"""Diagnose saved context-comparison predictions; no training or inference.

Ranks samples by mean ADE over fixed supplement seeds, not the worst seed.
Guide-frame labels describe errors and do not identify their causal mechanism.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
from itertools import cycle
import json
from pathlib import Path

import numpy as np

from analyze_corrections import classify_correction
from Dataset.audit_lane_visibility import point_polyline_distance
from diagnose_paths import components, tangent
from evaluate import EvaluationDataset
from evaluate_context_comparison import FixedInputs
from train import inspect_split


def ade(path, gt):
    return float(np.linalg.norm(path[1:] - gt[1:], axis=-1).mean())


def aggregate(rows):
    fields = ('ade_m', 'guide_ade_m', 'original_ade_m', 'regression_m',
              'mean_abs_longitudinal_error_m', 'mean_abs_lateral_error_m',
              'nearest_gt_distance_m', 'wrong_direction', 'overshoot', 'undershoot',
              'unnecessary', 'longitudinal_dominant', 'outside_road', 'outside_target_lane')
    return {'samples': len(rows), 'worlds': len({r['world_id'] for r in rows}),
            **{key: float(np.mean([r[key] for r in rows])) for key in fields},
            'regressed_fraction': float(np.mean([r['regression_m'] > .1 for r in rows])),
            'ade_over_1m_fraction': float(np.mean([r['ade_m'] > 1 for r in rows])),
            'dominant_counts': dict(Counter(r['error_dominant'] for r in rows))}


def plot_case(base, pool, arrays, index, seeds, shown_seed, destination):
    import matplotlib.pyplot as plt

    raw = base.raw_row(index)
    gt, guide = np.asarray(raw['gt_path']), np.asarray(raw['reference_line'])
    original = arrays['mixed_original'][index]
    expanded = arrays[f'mixed_expanded_seed_{shown_seed}'][index]
    before, after = base[index], FixedInputs(base, pool)[index]
    n_original = int(before['polyline_mask'].sum())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    styles = ((guide, 'Guide', '#e99b00', '--'), (gt, 'GT', '#159447', '-'),
              (original, 'Mixed / original', '#9455bd', ':'),
              (expanded, 'Mixed / expanded', '#db3030', '-'))
    for ax, title, limit in zip(axes[:2], ('Full horizon', 'First 30 m'), (81, 31)):
        used = set()
        for j in range(after['bev_points'].shape[0]):
            mask = after['point_mask'][j]
            if not mask.any():
                continue
            xy = after['bev_points'][j, mask, :2].numpy()
            kind = int(after['polyline_attr'][j, :4].argmax())
            color, label = (('#888888', 'Lane'), ('#222222', 'Curb'),
                            ('#2584ca', 'Stop line'), ('#42b8c4', 'Crosswalk'))[kind]
            added = j >= n_original
            label += ' added' if added else ' original'
            if after['polyline_attr'][j, 9] > .5:
                xy = np.concatenate([xy, xy[:1]])
            ax.plot(-xy[:, 1], xy[:, 0], color=color, ls='--' if added else '-',
                    alpha=.65, lw=1, label=label if label not in used else None)
            used.add(label)
        for xy, label, color, ls in styles:
            ax.plot(-xy[:, 1], xy[:, 0], color=color, ls=ls, lw=2, label=label)
        path = np.concatenate([xy[:limit] for xy, *_ in styles])
        ax.set(xlim=(-path[:, 1].max()-4, -path[:, 1].min()+4),
               ylim=(path[:, 0].min()-4, path[:, 0].max()+4), title=title,
               xlabel='Right (-ego y), m', ylabel='Forward (ego x), m')
        ax.scatter(0, 0, marker='^', c='black', s=35)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=6, loc='lower center')
    ax = axes[2]
    direction = tangent(guide)
    desired = components(gt - guide, direction)[:, 1]
    ax.plot(desired, color='#159447', lw=2, label='Required lateral correction')
    ax.plot(components(original-guide, direction)[:, 1], color='#9455bd', ls=':', label='Original correction')
    for seed, color in zip(seeds, cycle(('#287cb8', '#ee8b25', '#815a38', '#c75ca3'))):
        lateral = components(arrays[f'mixed_expanded_seed_{seed}'][index]-guide, direction)[:, 1]
        ax.plot(lateral, color=color, lw=1.5, alpha=.8, label=f'Expanded correction {seed}')
    ax.set(title='Signed lateral correction in guide frame', xlabel='Path station index (1 m spacing)',
           ylabel='Lateral correction (m)')
    ax.axhline(0, color='gray', lw=.7)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.suptitle(f'{raw["id"]}\nShown observation seed {shown_seed}; ADE: guide {ade(guide,gt):.2f}, '
                 f'original {ade(original,gt):.2f}, expanded {ade(expanded,gt):.2f} m', fontsize=11)
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def run(args):
    report = json.loads((args.evaluation_dir / 'summary.json').read_text())
    sources = [json.loads((d / 'summary.json').read_text()) for d in args.supplement_dirs]
    source = sources[0]
    manifest = inspect_split(source['input'])
    if manifest != report['data_manifest'] or any(s['data_manifest'] != manifest for s in sources):
        raise ValueError('Source fingerprints differ')
    seeds = [s['config']['seed'] for s in sources]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seeds')
    base = EvaluationDataset(source['input'], 32, 32)
    with np.load(args.evaluation_dir / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if arrays['ids'].tolist() != [base.raw_row(i)['id'] for i in range(len(base))]:
        raise ValueError('Prediction IDs differ')
    with (args.evaluation_dir / 'per_sample.csv').open() as f:
        saved_metrics = {(r['input'], int(r['index'])): r for r in csv.DictReader(f) if r['model'] == 'mixed'}
    rows = []
    for i in range(len(base)):
        raw = base.raw_row(i)
        gt, guide = np.asarray(raw['gt_path']), np.asarray(raw['reference_line'])
        paths = [arrays[f'mixed_expanded_seed_{seed}'][i] for seed in seeds]
        labels = [classify_correction(guide, gt, p) for p in paths]
        row = {'index': i, 'id': raw['id'], 'world_id': raw['world_id'], 'pose': raw['metadata']['pose_name'],
               'navigation': raw['navigation'], 'quality': raw['metadata']['quality'],
               'ade_m': float(np.mean([ade(p, gt) for p in paths])), 'guide_ade_m': ade(guide, gt),
               'original_ade_m': ade(arrays['mixed_original'][i], gt),
               'nearest_gt_distance_m': float(np.mean([point_polyline_distance(p[1:], gt).mean() for p in paths])),
               'error_dominant': Counter(c['error_dominant'] for c in labels).most_common(1)[0][0],
               **{k: float(np.mean([c[k] for c in labels])) for k in
                  ('mean_abs_longitudinal_error_m', 'mean_abs_lateral_error_m', 'wrong_direction',
                   'overshoot', 'undershoot', 'unnecessary', 'longitudinal_dominant')},
               **{k: float(np.mean([saved_metrics[(f'expanded_seed_{s}', i)][k] == 'True' for s in seeds]))
                  for k in ('outside_road', 'outside_target_lane')}}
        row['regression_m'] = row['ade_m'] - row['guide_ade_m']
        row['expansion_delta_m'] = row['ade_m'] - row['original_ade_m']
        rows.append(row)
    out = args.output_dir or args.evaluation_dir / 'failure_analysis'
    if out.exists():
        raise ValueError('Output exists; choose a new output directory')
    out.mkdir(parents=True)
    with (out / 'per_sample.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = {key: {v: aggregate([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
              for key in ('pose', 'navigation', 'quality')}
    focused = [r for r in rows if r['pose'] in args.poses]
    groups['focus_navigation'] = {v: aggregate([r for r in focused if r['navigation'] == v])
                                  for v in sorted({r['navigation'] for r in focused})}
    selections = []
    for pose in args.poses:
        subset = [r for r in rows if r['pose'] == pose]
        if not subset:
            raise ValueError(f'Unknown/empty pose: {pose}')
        selections.extend([(f'{pose}: worst ADE', max(subset, key=lambda r:r['ade_m'])),
                           (f'{pose}: worst vs guide', max(subset, key=lambda r:r['regression_m'])),
                           (f'{pose}: largest expansion degradation', max(subset, key=lambda r:r['expansion_delta_m'])),
                           (f'{pose}: median ADE', sorted(subset, key=lambda r:r['ade_m'])[len(subset)//2])])
    # Show one preselected observation seed in maps, all seeds in residual curves.
    # No per-sample seed cherry-picking; duplicate selected samples share one plot.
    shown_seed = seeds[0]
    pool = [json.loads(line) for line in (args.supplement_dirs[0] / 'added_observations.jsonl').open()]
    if len(pool) != len(base) or any(entry['id'] != base.raw_row(i)['id'] for i, entry in enumerate(pool)):
        raise ValueError('Supplement IDs differ')
    import matplotlib
    matplotlib.use('Agg')
    cards, rendered = [], set()
    for tag, row in selections:
        i = row['index']; filename = f'{i:06d}.png'
        if i not in rendered:
            plot_case(base, pool, arrays, i, seeds, shown_seed, out / filename)
            rendered.add(i)
        cards.append(f'<section><h2>{html.escape(tag)}</h2><p>{html.escape(row["id"])} | '
                     f'mean ADE {row["ade_m"]:.3f} m; guide {row["guide_ade_m"]:.3f} m; '
                     f'original {row["original_ade_m"]:.3f} m; {row["error_dominant"]}</p>'
                     f'<a href="{filename}"><img src="{filename}"></a></section>')
    summary = {'overall': aggregate(rows), 'groups': groups, 'supplement_seeds': seeds,
               'selected': [{'reason':tag, **row, 'plot':f'{row["index"]:06d}.png'} for tag,row in selections],
               'notes': ['One row per unique sample; numeric error/flag fields averaged over supplement seeds.',
                         'Flags can overlap; dominant label uses majority vote (ties follow seed order).',
                         'Regressed fraction uses mean ADE > guide ADE + 0.1 m.',
                         'Nearest-GT distance is directed prediction-to-GT distance, not an interchangeable ADE.',
                         'Error categories are descriptive thresholds, not causal conclusions.']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Context failures</title>'
        '<style>body{font:16px system-ui;margin:30px;background:#f7f7f9}section{background:white;padding:18px;margin:24px 0}'
        'img{width:100%}h2{font-size:20px}</style><h1>Mixed-model context diagnostics</h1>'
        '<p>Selection uses ADE averaged over fixed seeds. Maps show the first seed; curves show all seeds. '
        'Solid observations: original; dashed observations: added. Geometry shown is model input.</p>'
        + ''.join(cards), encoding='utf-8')
    print(json.dumps({'overall':summary['overall'], 'poses':groups['pose'],
                      'focus_navigation':groups['focus_navigation'], 'gallery':str(out/'index.html')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--supplement-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--poses', nargs='+', default=['entry', 'approach_10'])
    parser.add_argument('--output-dir', type=Path)
    run(parser.parse_args())
