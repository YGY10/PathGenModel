"""Compare repeated supplement seeds; seeds are robustness trials, not new worlds."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def aggregate(runs, output):
    reports = [json.loads((p / 'summary.json').read_text()) for p in runs]
    if len(reports) < 2 or len({r['supplement_seed'] for r in reports}) != len(reports):
        raise ValueError('Need at least two distinct supplement seeds')
    reference = reports[0]
    for report in reports[1:]:
        for key in ('checkpoint', 'input', 'budgets', 'selection_seed'):
            if report[key] != reference[key]:
                raise ValueError(f'Inconsistent {key}')
    variants = list(reference['all'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise ValueError('Output report exists')
    # Identity check, not just matching sample counts across the trials.
    ids = None
    for path in runs:
        with np.load(path / 'predictions.npz', allow_pickle=False) as data:
            current = data['ids'].tolist()
        if ids is not None and current != ids:
            raise ValueError('Sample identity mismatch across trials')
        ids = current
    rows, aggregates = [], {}
    for variant in variants:
        values = []
        for report in reports:
            s = report['all'][variant]
            row = {'supplement_seed': report['supplement_seed'], 'variant': variant,
                   'samples': s['samples'], 'mean_added_tokens': s['mean_added_tokens'],
                   'ade_m': s['ade_m']['after'], 'fde_m': s['fde_m']['after'],
                   'wrong_direction_rate': s['wrong_direction']['after'],
                   'target_lane_violation_rate': s['outside_target_lane']['after'],
                   'worlds_improved_over_1cm': s['worlds_improved_over_1cm'],
                   'worlds_worsened_over_1cm': s['worlds_worsened_over_1cm']}
            rows.append(row); values.append(row)
        aggregates[variant] = {key: {'mean': float(np.mean([v[key] for v in values])),
                                    'min': float(np.min([v[key] for v in values])),
                                    'max': float(np.max([v[key] for v in values]))}
                               for key in ('ade_m', 'fde_m', 'wrong_direction_rate', 'target_lane_violation_rate')}
    result = {'supplement_seeds': [r['supplement_seed'] for r in reports], 'runs': [str(p.resolve()) for p in runs],
              'samples_per_trial': len(ids), 'checkpoint': reference['checkpoint'], 'aggregate': aggregates,
              'notes': ['Same validation worlds across seeds; not independent dataset replications.',
                        'Min/max describe supplement randomness, not confidence intervals.',
                        'Selection seed fixed; Near/Far can differ in type/length/direction as well as distance.',
                        'Near/Far rank only within retained original-A pools; discarded candidates are not recovered.']}
    with (output / 'trials.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in zip(axes, ('ade_m', 'target_lane_violation_rate'), ('ADE (m)', 'Target-lane violation fraction')):
        for mode, color in [('near', '#158b67'), ('far', '#cc593d'), ('random', '#3e6dab')]:
            keys = [f'{mode}_{k}' for k in reference['budgets']]
            means = np.array([aggregates[k][metric]['mean'] for k in keys])
            lower = np.array([aggregates[k][metric]['min'] for k in keys])
            upper = np.array([aggregates[k][metric]['max'] for k in keys])
            ax.plot(reference['budgets'], means, '-o', color=color, label=mode)
            ax.fill_between(reference['budgets'], lower, upper, color=color, alpha=.15)
        for variant, style, color in [('baseline', '--', '#333333'), ('full', ':', '#999999')]:
            ax.axhline(aggregates[variant][metric]['mean'], ls=style, color=color, label=variant)
        ax.set(xlabel='Requested added K (actual K may be smaller)', ylabel=label, xticks=reference['budgets'])
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Frozen-model input selection; shade = min/max across supplement seeds')
    fig.tight_layout(); fig.savefig(output / 'comparison.png', dpi=150); plt.close(fig)
    print('variant / mean ADE / seed range')
    for variant, value in aggregates.items():
        v = value['ade_m']
        print(f'{variant}: {v["mean"]:.4f} [{v["min"]:.4f}, {v["max"]:.4f}]')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.runs, args.output_dir)
