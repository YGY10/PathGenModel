"""Compare control/mixed checkpoints on identical fixed original + FOV-A inputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Dataset.dataset import encode_sample
from Dataset.context_augmentation import append_observations
from Dataset.audit_drivable import drivable_polygon
from diagnose_paths import target_lane_polygon
from evaluate import EvaluationDataset, load_model
from experiment_fov import metrics
from train import inspect_split


class FixedInputs(Dataset):
    def __init__(self, base, supplements=None):
        self.base = base
        self.supplements = supplements

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base.raw_row(index)
        result = encode_sample(row, self.base.max_lines, self.base.max_points)
        if self.supplements is not None:
            entry = self.supplements[index]
            if entry['id'] != row['id']:
                raise ValueError('Supplement sample identity mismatch')
            selected = [(obj['kind'], {k: v for k, v in obj.items() if k != 'kind'}) for obj in entry['added_objects']]
            result, kept = append_observations(result, selected, self.base.max_points)
            if len(kept) != len(selected):
                raise ValueError('Fixed input capacity mismatch')
        return result


@torch.inference_mode()
def inference(model, dataset, device, batch_size):
    outputs = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model.forward_batch(batch)
        if not torch.isfinite(pred).all():
            raise FloatingPointError('Nonfinite prediction')
        outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs)


def aggregate(rows):
    return {'samples': len(rows), 'worlds': len({r['world_id'] for r in rows}),
            **{key: float(np.mean([r[key] for r in rows])) for key in ('ade_m', 'fde_m', 'wrong_direction', 'overshoot', 'outside_road', 'outside_target_lane')}}


def run(args):
    if args.batch_size < 1:
        raise ValueError('batch_size must be positive')
    sources = [json.loads((path / 'summary.json').read_text()) for path in args.supplement_dirs]
    source = sources[0]
    info = inspect_split(source['input'])
    if info != source['data_manifest'] or any(s['data_manifest'] != info or s['input'] != source['input'] for s in sources):
        raise ValueError('Fixed validation source mismatch')
    seeds = [s['config']['seed'] for s in sources]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate supplement seeds')
    pools = [[json.loads(line) for line in (path / 'added_observations.jsonl').open()] for path in args.supplement_dirs]
    base = EvaluationDataset(source['input'], 32, 32)
    for pool in pools:
        if len(pool) != len(base):
            raise ValueError('Incomplete fixed supplement')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    checkpoints = {'control': args.control, 'mixed': args.mixed}
    saved_headers = {name: torch.load(path, map_location='cpu', weights_only=False) for name, path in checkpoints.items()}
    control, mixed = saved_headers['control'], saved_headers['mixed']
    for key in ('model', 'loss', 'seed', 'batch_size', 'learning_rate', 'weight_decay', 'grad_clip_norm',
                'lr_patience', 'lr_factor', 'min_lr', 'epochs', 'early_stop_patience'):
        if control['config'][key] != mixed['config'][key]:
            raise ValueError(f'Control/mixed {key} settings differ')
    if control['splits']['train'] != mixed['splits']['train']:
        raise ValueError('Control/mixed base training splits differ')
    if control['config'].get('augmentation') is not None or mixed['config'].get('augmentation') is None:
        raise ValueError('Expected unaugmented control and context-augmented mixed checkpoints')
    if not control.get('initial_model_sha256') or control['initial_model_sha256'] != mixed.get('initial_model_sha256'):
        raise ValueError('Missing/different initialization fingerprints; use the matched new training configs')
    for saved in saved_headers.values():
        if saved['splits']['val'] != info or set(info['worlds']) & set(saved['splits']['train']['worlds']):
            raise ValueError('Validation fingerprint or world isolation mismatch')
        if (saved['config']['max_lines'], saved['config']['max_points']) != (32, 32):
            raise ValueError('Expected unchanged 32x32 capacity')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise ValueError('Output exists; choose a new directory')
    contexts, cache = [], {}
    for i in range(len(base)):
        row = base.raw_row(i)
        wid, nav = row['world_id'], row['navigation']
        if wid not in cache:
            world = json.loads((args.worlds_dir / f'{wid}.json').read_text())
            routes = json.loads((args.routes_dir / f'{wid}.json').read_text())
            cache[wid] = (routes, drivable_polygon(world), {k: target_lane_polygon(world, rt) for k, rt in routes.items()})
        routes, road, lanes = cache[wid]
        contexts.append((row, routes[nav], road, lanes[nav]))
    all_rows, prediction_arrays, result = [], {}, {}
    for name, checkpoint in checkpoints.items():
        model, saved = load_model(checkpoint, device)
        result[name] = {'checkpoint': str(checkpoint.resolve()), 'epoch': saved['epoch'], 'inputs': {}}
        arms = [('original', None)] + [(f'expanded_seed_{seed}', pool) for seed, pool in zip(seeds, pools)]
        for arm, pool in arms:
            pred = inference(model, FixedInputs(base, pool), device, args.batch_size)
            rows = []
            for i, (row, route, road, lane) in enumerate(contexts):
                gt, guide = np.asarray(row['gt_path']), np.asarray(row['reference_line'])
                values = metrics(row, pred[i], gt, guide, road, lane, route)
                entry = {'model': name, 'input': arm, 'index': i, 'id': row['id'], 'world_id': row['world_id'],
                         'navigation': row['navigation'], 'pose': row['metadata']['pose_name'],
                         'quality': row['metadata']['quality'], **values}
                rows.append(entry)
            all_rows.extend(rows)
            prediction_arrays[f'{name}_{arm}'] = pred
            result[name]['inputs'][arm] = {'overall': aggregate(rows),
                'groups': {key: {value: aggregate([r for r in rows if r[key] == value]) for value in sorted({r[key] for r in rows})}
                           for key in ('navigation', 'pose', 'quality', 'world_id')}}
            print(name, arm, json.dumps(result[name]['inputs'][arm]['overall']), flush=True)
        del model
    comparisons = {}
    for arm in result['control']['inputs']:
        before, after = (result[name]['inputs'][arm]['overall'] for name in ('control', 'mixed'))
        comparisons[arm] = {key + '_mixed_minus_control': after[key] - before[key]
                            for key in ('ade_m', 'fde_m', 'wrong_direction', 'outside_road', 'outside_target_lane')}
    with (out / 'per_sample.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    np.savez_compressed(out / 'predictions.npz', ids=np.array([ctx[0]['id'] for ctx in contexts]), **prediction_arrays)
    report = {'models': result, 'paired_comparisons': comparisons, 'data_manifest': info,
              'initial_model_sha256': control['initial_model_sha256'],
              'notes': ['Model selection uses original validation ADE in both training runs.',
                        'Same fixed validation observations; expanded noise is not resampled.',
                        'Negative mixed-minus-control delta means improvement for each reported error/rate.',
                        'Supplement seeds repeat the same worlds; they are not independent dataset replications.']}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Comparison: {out}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--mixed', type=Path, required=True)
    parser.add_argument('--supplement-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--worlds-dir', type=Path, required=True)
    parser.add_argument('--routes-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    run(parser.parse_args())
