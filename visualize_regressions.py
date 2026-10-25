"""Render samples where a saved prediction is worse than its guide; no inference.

python visualize_regressions.py --evaluation-dir outputs/training_v1/evaluation_best_val
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import EvaluationDataset, plot_sample, sample_records
from train import inspect_split


def rank_regressions(records, metric='ade', tolerance_m=1e-6):
    if metric not in ('ade', 'fde') or tolerance_m < 0 or not np.isfinite(tolerance_m):
        raise ValueError('Invalid ranking metric or tolerance')
    ranked = []
    for record in records:
        delta = record[f'{metric}_m'] - record[f'guide_{metric}_m']
        if delta > tolerance_m:
            ranked.append({**record, 'regression_m': delta})
    return sorted(ranked, key=lambda r: (-r['regression_m'], r['index']))


def render_regressions(evaluation_dir, output_dir=None, top_k=12, metric='ade', tolerance_m=1e-6):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    evaluation_dir = Path(evaluation_dir)
    summary = json.loads((evaluation_dir / 'summary.json').read_text())
    # The checkpoint restores the exact input encoding used by this evaluation.
    saved = torch.load(summary['checkpoint'], map_location='cpu', weights_only=False)
    dataset = EvaluationDataset(summary['input'], saved['config']['max_lines'], saved['config']['max_points'])
    if inspect_split(summary['input']) != summary['data_manifest']:
        raise ValueError('Evaluation input data has changed')
    with np.load(evaluation_dir / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {k: torch.from_numpy(archive[k].copy()) for k in ('pred', 'gt', 'guide', 'empty_bev') if k in archive}
        ids = archive['ids'].tolist()
    if ids != [dataset.raw_row(i)['id'] for i in range(len(dataset))]:
        raise ValueError('Saved predictions are not aligned with the input data')
    records = sample_records(dataset, arrays)
    ranked = rank_regressions(records, metric, tolerance_m)
    output = Path(output_dir) if output_dir else evaluation_dir / f'regressions_{metric}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'index.html').exists():
        raise ValueError('Report already exists; select a new --output-dir')
    with (output / 'ranked.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[*records[0], 'regression_m'])
        writer.writeheader()
        writer.writerows(ranked)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = output / 'samples'
    plots.mkdir(exist_ok=True)
    cards, selected = [], []
    for rank, record in enumerate(ranked[:top_k], 1):
        filename = f'samples/{rank:03d}_{record["index"]:06d}.png'
        plot_sample(dataset, arrays, record,
                    [f'{metric.upper()} regression +{record["regression_m"]:.3f}m'], output / filename)
        # Pointwise error makes it clear where the correction deteriorates.
        index = record['index']
        fig, ax = plt.subplots(figsize=(10, 3))
        for key, label, color in (('guide', 'Guide', '#ea9900'), ('pred', 'Prediction', '#db3030')):
            error = (arrays[key][index] - arrays['gt'][index]).norm(dim=-1).numpy()
            ax.plot(np.arange(len(error)), error, label=label, color=color)
        ax.set(xlabel='GT path station (m)', ylabel='Position error (m)', title=record['id'])
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        curve = f'samples/{rank:03d}_{index:06d}_error.png'
        fig.savefig(output / curve, dpi=130)
        plt.close(fig)
        cards.append(f'<section><h2>#{rank} {html.escape(record["id"])}</h2>'
                     f'<p>{metric.upper()} worsened by {record["regression_m"]:.3f} m | '
                     f'ADE: guide {record["guide_ade_m"]:.3f} → prediction {record["ade_m"]:.3f} m | '
                     f'FDE: guide {record["guide_fde_m"]:.3f} → prediction {record["fde_m"]:.3f} m</p>'
                     f'<a href="{filename}"><img src="{filename}" alt="Path comparison"></a>'
                     f'<img src="{curve}" alt="Position error by path station"></section>')
        selected.append({'rank': rank, 'id': record['id'], 'index': index,
                         'regression_m': record['regression_m'], 'plot': filename, 'error_plot': curve})
    report = {'metric': metric, 'tolerance_m': tolerance_m, 'samples': len(records),
              'regressed_samples': len(ranked), 'rendered': len(selected), 'selected': selected}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Guide correction regressions</title><style>'
        'body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;background:#f6f7f9}'
        'section{background:white;padding:20px;margin:24px 0;border-radius:10px}'
        'img{display:block;max-width:100%;height:auto}h2{font-size:18px;overflow-wrap:anywhere}'
        '</style><h1>Guide correction regressions</h1>'
        f'<p>{len(ranked)} / {len(records)} samples have {metric.upper()} regression greater than '
        f'{tolerance_m:g} m. Showing the largest {len(selected)}. '
        'Ranked by prediction error minus guide error. All distances in metres.</p>'
        '<p>Green: GT; orange dashed: guide; red: prediction; purple dotted: empty BEV. '
        'Plots show observed model inputs, not full truth-world geometry.</p>'
        + ''.join(cards) + '</html>', encoding='utf-8')
    print(f'{len(ranked)}/{len(records)} regressed; rendered {len(selected)} samples. Gallery: {output / "index.html"}')
    for row in ranked[:min(5, top_k)]:
        print(f'{row["id"]}: guide {row[f"guide_{metric}_m"]:.3f}m → '
              f'prediction {row[f"{metric}_m"]:.3f}m (+{row["regression_m"]:.3f}m)')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--top-k', type=int, default=12)
    parser.add_argument('--metric', choices=('ade', 'fde'), default='ade')
    parser.add_argument('--tolerance-m', type=float, default=1e-6)
    args = parser.parse_args()
    render_regressions(args.evaluation_dir, args.output_dir, args.top_k, args.metric, args.tolerance_m)


if __name__ == '__main__':
    main()
