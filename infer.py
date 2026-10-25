"""Single-sample inference using exactly the training-time vector encoding."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from Dataset.dataset import KINDS, encode_sample
from Model.pathgen_model_v1 import ModelConfig, PathGenModel


GEOMETRY_KEYS = ('lane_markings', 'road_boundaries', 'stop_lines', 'crosswalks')


def xy_array(value, name, minimum=1):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < minimum or not np.isfinite(a).all():
        raise ValueError(f'{name} must contain at least {minimum} finite [x,y] points')
    return a


def prepare_sample(raw, resample_guide=False):
    """Accept compact user JSON or a dataset row. GT is optional, never an input."""
    row = dict(raw)
    row['navigation'] = row.get('navigation', row.get('nav'))
    guide = xy_array(row.get('reference_line', row.get('guide')), 'guide', 2)
    if resample_guide:
        # Sample the first 80 metres; never stretch a short guide to 80 metres.
        guide = guide[np.r_[True, np.linalg.norm(np.diff(guide, axis=0), axis=1) > 1e-9]]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide, axis=0), axis=1))]
        if arc[-1] < 80 - 1e-6:
            raise ValueError('--resample-guide requires a guide at least 80 m long')
        guide = np.column_stack([np.interp(np.arange(81), arc, guide[:, k]) for k in range(2)])
    if guide.shape != (81, 2):
        raise ValueError('Guide must have 81 points; use --resample-guide for an arbitrary polyline >=80 m')
    row['reference_line'] = guide.tolist()
    for key in GEOMETRY_KEYS:
        objects = row.get(key, [])
        if not isinstance(objects, list):
            raise ValueError(f'{key} must be a list of objects with a points field')
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f'{key}[{i}] must be an object with a points field')
            a = xy_array(obj.get('points'), f'{key}[{i}].points', 3 if key in GEOMETRY_KEYS[2:] else 2)
            if key in GEOMETRY_KEYS[2:]:
                if len(np.unique(a, axis=0)) < 3:
                    raise ValueError(f'{key}[{i}] needs at least 3 distinct polygon vertices')
            confidence = float(obj.get('confidence', 1.))
            if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f'{key}[{i}].confidence must be in [0,1]')
        row[key] = objects
    if 'gt_path' in row:
        if xy_array(row['gt_path'], 'gt_path').shape != (81, 2):
            raise ValueError('Optional gt_path must have 81 points')
    return row


def encode_input(row, max_lines, max_points):
    # The existing dataset encoder validates GT. A local placeholder allows us to
    # reuse its feature calculation; remove it before calling the model.
    encoded = encode_sample({**row, 'gt_path': row.get('gt_path', row['reference_line'])},
                            max_lines=max_lines, max_points=max_points)
    encoded.pop('gt_path')
    return encoded


def read_sample(path, index):
    if path.suffix.lower() == '.jsonl':
        with path.open() as stream:
            for i, line in enumerate(stream):
                if i == index:
                    return json.loads(line)
        raise ValueError(f'No sample at index {index}')
    if index != 0:
        raise ValueError('--index is only supported with JSONL input')
    return json.loads(path.read_text())


def plot_prediction(encoded, row, prediction, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    colors = ('#888888', '#333333', '#2385cd', '#42b8c4')
    for ax, limit, title in zip(axes, (81, 31), ('Full horizon', 'First 30 stations')):
        used = set()
        for i in range(len(encoded['bev_points'])):
            mask = encoded['point_mask'][i]
            if not mask.any():
                continue
            xy = encoded['bev_points'][i, mask, :2].numpy()
            kind = int(encoded['polyline_attr'][i, :4].argmax())
            if encoded['polyline_attr'][i, 9] > .5:
                xy = np.vstack([xy, xy[:1]])
            name = KINDS[kind]
            ax.plot(-xy[:, 1], xy[:, 0], color=colors[kind], lw=1.2,
                    label=name if name not in used else None)
            used.add(name)
        paths = [(np.asarray(row['reference_line']), 'Guide', '#e99a00', '--'),
                 (prediction, 'Prediction', '#db3030', '-')]
        if 'gt_path' in row:
            paths.append((np.asarray(row['gt_path']), 'GT', '#159447', '-'))
        for xy, name, color, style in paths:
            ax.plot(-xy[:, 1], xy[:, 0], color=color, ls=style, lw=2, label=name)
        extent = np.concatenate([xy[:limit] for xy, *_ in paths] + [np.zeros((1, 2))])
        ax.set_xlim(-extent[:, 1].max() - 5, -extent[:, 1].min() + 5)
        ax.set_ylim(extent[:, 0].min() - 5, extent[:, 0].max() + 5)
        ax.scatter([0], [0], c='black', marker='^', label='Ego', zorder=5)
        ax.set(xlabel='Right (-ego y), m', ylabel='Forward (ego x), m', title=title)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle(f"{row.get('id', 'Custom input')} | nav={row['navigation']}")
    fig.tight_layout(rect=(0, .12, 1, .95))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Custom JSON or dataset JSONL')
    parser.add_argument('--index', type=int, default=0, help='Zero-based JSONL row index')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--resample-guide', action='store_true', help='Sample the first 80 m at 1 m spacing')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('outputs/custom_inference') / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    if args.index < 0:
        parser.error('--index must be nonnegative')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    # Load only checkpoints you trust, using the same format as train/evaluate.
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    model = PathGenModel(ModelConfig(**config['model'])).to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    row = prepare_sample(read_sample(args.input, args.index), args.resample_guide)
    encoded = encode_input(row, config.get('max_lines', 32), config.get('max_points', 32))
    with torch.inference_mode():
        prediction = model.forward_batch({k: v.unsqueeze(0).to(device) for k, v in encoded.items()})[0].cpu().numpy()
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Non-finite prediction')
    supplied = {k: len(row[k]) for k in GEOMETRY_KEYS}
    retained = {k: int((encoded['polyline_attr'][:, i] * encoded['polyline_mask']).sum())
                for i, k in enumerate(GEOMETRY_KEYS)}
    result = {'checkpoint': str(args.checkpoint.resolve()), 'input': str(args.input.resolve()),
              'index': args.index, 'device': str(device), 'coordinate_frame': 'ego_x_forward_y_left_m',
              'navigation': row['navigation'], 'supplied_objects': supplied, 'retained_objects': retained,
              'reference_line': row['reference_line'], 'prediction': prediction.tolist(),
              'residual': (prediction - np.asarray(row['reference_line'])).tolist()}
    if 'gt_path' in row:
        error = np.linalg.norm(prediction - np.asarray(row['gt_path']), axis=1)
        result.update(gt_path=row['gt_path'], ade_m=float(error[1:].mean()), fde_m=float(error[-1]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'prediction.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output_dir / 'input.json').write_text(json.dumps(row, indent=2) + '\n')
    plot_prediction(encoded, row, prediction, args.output_dir / 'prediction.png')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('reference_line', 'prediction', 'residual', 'gt_path')}, indent=2))
    if sum(supplied.values()) > sum(retained.values()):
        print('Token capacity exceeded: some objects were discarded by the training-time selection rule.')
    print(f'Results: {args.output_dir}')


if __name__ == '__main__':
    main()
