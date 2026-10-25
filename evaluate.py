"""Evaluate a trusted training checkpoint and visualize observed BEV + paths.

Run from the repository root: python evaluate.py --checkpoint outputs/training_v1/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Dataset.dataset import PathJSONLDataset
from Model.pathgen_model_v1 import ModelConfig, PathGenModel
from train import inspect_split, path_metrics


class EvaluationDataset(PathJSONLDataset):
    """Retain a stable row index; metadata is read only for reports and plots."""
    def raw_row(self, index):
        with self.path.open('rb') as f:
            f.seek(self.offsets[index])
            return json.loads(f.readline())

    def __getitem__(self, index):
        return {**super().__getitem__(index), '_index': torch.tensor(index)}


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = PathGenModel(ModelConfig(**saved['config']['model'])).to(device)
    model.load_state_dict(saved['model'], strict=True)
    return model.eval(), saved


@torch.no_grad()
def predict(model, dataset, device, batch_size, workers, empty_bev):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=device.type == 'cuda')
    predictions, empty_predictions, targets, guides, indices = [], [], [], [], []
    for batch in loader:
        indices.extend(batch.pop('_index').tolist())
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pred = model.forward_batch(batch)
        if not torch.isfinite(pred).all():
            raise FloatingPointError('Non-finite predictions')
        predictions.append(pred.cpu())
        targets.append(batch['gt_path'].cpu())
        guides.append(batch['guide'].cpu())
        if empty_bev:
            empty = dict(batch)
            for key in ('bev_points', 'point_mask', 'polyline_mask', 'polyline_attr'):
                empty[key] = torch.zeros_like(batch[key])
            pred_empty = model.forward_batch(empty)
            if not torch.isfinite(pred_empty).all():
                raise FloatingPointError('Non-finite empty-BEV predictions')
            empty_predictions.append(pred_empty.cpu())
    if indices != list(range(len(dataset))):
        raise RuntimeError('Evaluation sample order mismatch')
    return {'pred': torch.cat(predictions), 'gt': torch.cat(targets),
            'guide': torch.cat(guides),
            **({'empty_bev': torch.cat(empty_predictions)} if empty_bev else {})}


def sample_records(dataset, arrays):
    metrics = {}
    for prefix, key in (('', 'pred'), ('guide_', 'guide'), ('empty_bev_', 'empty_bev')):
        if key in arrays:
            metrics.update({prefix + k: v.numpy() for k, v in path_metrics(arrays[key], arrays['gt']).items()})
    records = []
    for i in range(len(dataset)):
        row = dataset.raw_row(i)
        md = row.get('metadata', {})
        entry = {'index': i, 'id': row['id'], 'world_id': row['world_id'],
                 'navigation': row['navigation'], 'quality': md.get('quality', 'unknown'),
                 'pose': md.get('pose_name', 'unknown'),
                 **{k: float(v[i]) for k, v in metrics.items()}}
        entry['ade_gain_m'] = entry['guide_ade_m'] - entry['ade_m']
        if 'empty_bev' in arrays:
            entry['empty_bev_ade_delta_m'] = entry['empty_bev_ade_m'] - entry['ade_m']
        records.append(entry)
    return records


def summarize(records):
    numeric = [k for k, v in records[0].items() if k != 'index' and isinstance(v, (int, float))]
    means = {k: float(np.mean([r[k] for r in records])) for k in numeric}
    summary = {'samples': len(records), 'worlds': len({r['world_id'] for r in records}), **means}
    for key in ('ade_m', 'fde_m', 'guide_ade_m', 'empty_bev_ade_m'):
        if key in records[0]:
            values = [r[key] for r in records]
            summary[key + '_p50'] = float(np.percentile(values, 50))
            summary[key + '_p90'] = float(np.percentile(values, 90))
            summary[key + '_max'] = float(np.max(values))
    summary['ade_improvement_pct'] = (100 * (1 - means['ade_m'] / means['guide_ade_m'])
                                      if means['guide_ade_m'] > 0 else None)
    summary['fraction_better_than_guide'] = float(np.mean([r['ade_gain_m'] > 0 for r in records]))
    return summary


def grouped_summaries(records, keys=('navigation', 'quality', 'pose', 'world_id')):
    return {key: {value: summarize([r for r in records if r[key] == value])
                  for value in sorted({r[key] for r in records})} for key in keys}


def select_examples(records, count, seed):
    """Deduplicated best/worst/random examples within each navigation group."""
    rng = np.random.default_rng(seed)
    selected = {}
    for nav in sorted({r['navigation'] for r in records}):
        group = sorted([r for r in records if r['navigation'] == nav], key=lambda r: r['ade_m'])
        n = min(count, len(group))
        for label, subset in (('best', group[:n]), ('worst', list(reversed(group[-n:])) if n else []),
                              ('random', [group[i] for i in rng.choice(len(group), n, replace=False)])):
            for row in subset:
                selected.setdefault(row['index'], []).append(label)
    return selected


def plot_sample(dataset, arrays, record, tags, destination):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    index = record['index']
    encoded = dataset[index]
    raw = dataset.raw_row(index)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    styles = [('guide', '#ea9900', '--', 'Guide'), ('gt', '#159447', '-', 'GT'),
              ('pred', '#db3030', '-', 'Prediction'), ('empty_bev', '#a050bf', ':', 'Empty BEV')]
    for ax, title, limit in zip(axes, ('Full horizon', 'First 30 m'), (81, 31)):
        used = set()
        for j in range(encoded['bev_points'].shape[0]):
            mask = encoded['point_mask'][j]
            if not mask.any():
                continue
            xy = encoded['bev_points'][j, mask, :2].numpy()
            kind = int(encoded['polyline_attr'][j, :4].argmax())
            color, name = (('#888888', 'Lane marking'), ('#333333', 'Road boundary'),
                           ('#1e85ce', 'Stop line'), ('#42b8c4', 'Crosswalk'))[kind]
            if encoded['polyline_attr'][j, 9] > .5:
                xy = np.concatenate([xy, xy[:1]])
            ax.plot(-xy[:, 1], xy[:, 0], color=color, lw=1.1,
                    label=name if name not in used else None, alpha=.75)
            used.add(name)
        for key, color, style, name in styles:
            if key in arrays:
                xy = arrays[key][index].numpy()
                ax.plot(-xy[:, 1], xy[:, 0], color=color, ls=style, lw=2, label=name)
        fov = raw.get('metadata', {}).get('vcs_fov_m')
        if fov:
            ax.add_patch(Rectangle((-fov['left'], -fov['rear']), fov['left'] + fov['right'],
                                   fov['forward'] + fov['rear'], fill=False, ec='#bbbbbb',
                                   ls='--', label='FOV'))
        # Fit limits to paths, not distant unrelated observation fragments.
        path = np.concatenate([arrays[key][index, :limit].numpy()
                               for key, *_ in styles if key in arrays])
        ax.set_xlim(-path[:, 1].max() - 5, -path[:, 1].min() + 5)
        ax.set_ylim(path[:, 0].min() - 5, path[:, 0].max() + 5)
        ax.scatter([0], [0], marker='^', color='black', s=50, label='Ego')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Right (-ego y), m')
        ax.set_ylabel('Forward (ego x), m')
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle(f'{record["id"]}\n{record["navigation"]} / {record["quality"]} / {record["pose"]} | '
                 f'ADE {record["ade_m"]:.3f}m, FDE {record["fde_m"]:.3f}m, '
                 f'guide ADE {record["guide_ade_m"]:.3f}m | {", ".join(tags)}', fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_overview(arrays, destination):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in (('pred', 'Prediction'), ('guide', 'Guide'), ('empty_bev', 'Empty BEV')):
        if key not in arrays:
            continue
        distance = (arrays[key] - arrays['gt']).norm(dim=-1).numpy()
        axes[0].plot(np.arange(1, distance.shape[1]), distance[:, 1:].mean(0), label=label)
        ade = np.sort(distance[:, 1:].mean(1))
        axes[1].step(ade, np.arange(1, len(ade) + 1) / len(ade), where='post', label=label)
    axes[0].set(xlabel='GT path station (m, 1 m sampling)', ylabel='Mean error (m)')
    axes[1].set(xlabel='Per-sample ADE (m)', ylabel='Cumulative sample fraction', ylim=(0, 1.01))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def evaluate(args):
    if args.batch_size < 1 or args.num_workers < 0 or args.examples_per_group < 0:
        raise ValueError('Invalid batch/worker/example count')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    model, saved = load_model(args.checkpoint, device)
    path = args.input or Path(saved['config'][f'{args.split}_path'])
    manifest = inspect_split(path)
    train_worlds = set(saved['splits']['train']['worlds'])
    overlap = sorted(train_worlds.intersection(manifest['worlds']))
    if overlap and args.split != 'train':
        raise ValueError('Input overlaps checkpoint training worlds; use --split train for training-set diagnostics')
    if args.input is None and manifest['sha256'] != saved['splits'][args.split]['sha256']:
        raise ValueError('Checkpoint split data changed; pass --input explicitly to evaluate a new dataset')
    out = args.output_dir or args.checkpoint.parent / ('evaluation_' + args.checkpoint.stem + '_' + args.split)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise ValueError('Evaluation output already exists; select a new --output-dir')
    dataset = EvaluationDataset(path, saved['config']['max_lines'], saved['config']['max_points'])
    arrays = predict(model, dataset, device, args.batch_size, args.num_workers, args.empty_bev)
    records = sample_records(dataset, arrays)
    summary = {'checkpoint': str(args.checkpoint.resolve()), 'epoch': saved['epoch'],
               'input': str(Path(path).resolve()), 'device': str(device), 'data_manifest': manifest,
               'training_world_overlap': overlap, 'overall': summarize(records),
               'groups': grouped_summaries(records),
               'notes': ['ADE excludes fixed origin; metrics in metres; equal sample weighting.',
                         'Empty BEV is inference-time sensitivity, not a retrained guide-only baseline.',
                         'No drivable/lane legality metrics computed; aggregate quality/pose groups overlap within worlds.']}
    with (out / 'per_sample.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(out / 'predictions.npz', ids=np.array([r['id'] for r in records]),
                        **{k: v.numpy() for k, v in arrays.items()})
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        plots = out / 'samples'
        plots.mkdir(exist_ok=True)
        selected = select_examples(records, args.examples_per_group, args.seed)
        for index, tags in selected.items():
            plot_sample(dataset, arrays, records[index], tags, plots / f'{index:06d}.png')
        (out / 'plot_manifest.json').write_text(json.dumps([
            {'index': i, 'id': records[i]['id'], 'tags': tags, 'file': f'samples/{i:06d}.png'}
            for i, tags in selected.items()], indent=2) + '\n')
        plot_overview(arrays, out / 'overview.png')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary['overall'], indent=2))
    print(f'Results: {out}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted checkpoint only')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--input', type=Path, help='Explicit JSONL; required for test/custom data')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--empty-bev', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--examples-per-group', type=int, default=2, help='Best/worst/random count per navigation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.split == 'test' and args.input is None:
        parser.error('--split test requires --input test.jsonl')
    evaluate(args)


if __name__ == '__main__':
    main()
