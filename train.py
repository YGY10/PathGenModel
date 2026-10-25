"""Single-device FP32 training: python train.py --config configs/train_v1.json.

Dataset/model/loss construction, epoch execution and checkpointing are separate
functions so future architectures or supervision can be added independently.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from Dataset.dataset import PathJSONLDataset
from Model.pathgen_model_v1 import ModelConfig, PathGenModel
from Model.losses import PathLossConfig, PathLoss
from Dataset.context_augmentation import (
    ContextAugmentationConfig, ContextAugmentedDataset, inspect_context_cache,
)


@dataclass
class TrainConfig:
    train_path: str = 'outputs/v4_world30_retry/audit/train.jsonl'
    val_path: str = 'outputs/v4_world30_retry/audit/val.jsonl'
    output_dir: str = 'outputs/training_v1'
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 0
    device: str = 'auto'
    seed: int = 42
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    lr_patience: int = 8
    lr_factor: float = 0.5
    min_lr: float = 0.000001
    early_stop_patience: int = 0  # 0 disables early stopping
    tensorboard: bool = False
    tensorboard_dir: str | None = None
    augmentation: dict | None = None
    max_lines: int = 32
    max_points: int = 32
    model: dict = field(default_factory=lambda: asdict(ModelConfig()))
    loss: dict = field(default_factory=lambda: asdict(PathLossConfig()))

    def validate(self):
        for key in ('epochs', 'batch_size', 'max_lines'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if self.max_points < 4 or min(self.num_workers, self.lr_patience, self.early_stop_patience) < 0:
            raise ValueError('Invalid point count or patience/worker count')
        for key in ('learning_rate', 'grad_clip_norm', 'min_lr'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        if not 0 < self.lr_factor < 1 or self.min_lr > self.learning_rate:
            raise ValueError('Invalid learning-rate scheduler settings')
        model = ModelConfig(**self.model)
        if (model.point_dim, model.attr_dim, model.num_nav) != (6, 10, 3):
            raise ValueError('Dataset requires point_dim=6, attr_dim=10, num_nav=3')
        PathLossConfig(**self.loss)
        if self.augmentation is not None:
            ContextAugmentationConfig(**self.augmentation)


def inspect_split(path):
    """Fingerprint all data and check split identity before training."""
    digest = hashlib.sha256()
    worlds, ids = set(), set()
    count = 0
    with Path(path).open('rb') as f:
        for line in f:
            digest.update(line)
            row = json.loads(line)
            if not row.get('world_id') or not row.get('id'):
                raise ValueError(f'{path}: every row needs world_id and id')
            if row['id'] in ids:
                raise ValueError(f'{path}: duplicate sample {row["id"]}')
            ids.add(row['id'])
            worlds.add(row['world_id'])
            count += 1
    if not count:
        raise ValueError(f'Empty split: {path}')
    return {'sha256': digest.hexdigest(), 'samples': count, 'worlds': sorted(worlds)}


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_loader(path, cfg, shuffle, generator, device):
    dataset = (ContextAugmentedDataset(path, ContextAugmentationConfig(**cfg.augmentation), cfg.max_lines, cfg.max_points)
               if shuffle and cfg.augmentation is not None else PathJSONLDataset(path, cfg.max_lines, cfg.max_points))
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=device.type == 'cuda',
                      multiprocessing_context='spawn' if cfg.num_workers else None,
                      worker_init_fn=seed_worker, generator=generator, drop_last=False)


def path_metrics(pred, gt):
    """Per-sample metre errors; exclude the fixed origin from ADE."""
    distance = (pred.float() - gt.float()).norm(dim=-1)
    return {'ade_m': distance[:, 1:].mean(1), 'fde_m': distance[:, -1],
            'ade_1_20m': distance[:, 1:21].mean(1),
            'ade_21_40m': distance[:, 21:41].mean(1),
            'ade_41_80m': distance[:, 41:81].mean(1)}


def run_epoch(model, criterion, loader, device, optimizer=None, grad_clip_norm=1.):
    training = optimizer is not None
    model.train(training)
    sums, nav_sums = {}, {}
    count = 0
    augmentation_profiles, added_tokens, actual_augmented = {}, 0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model.forward_batch(batch)
            result = criterion(pred, batch['gt_path'], batch.get('valid_mask'), batch.get('point_weight'))
            if not torch.isfinite(result['total']):
                raise FloatingPointError('Non-finite loss')
            if training:
                result['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=True)
                optimizer.step()
            n = pred.shape[0]
            count += n
            if '_augmentation_profile' in batch:
                for profile in batch['_augmentation_profile'].unique().tolist():
                    augmentation_profiles[str(profile)] = augmentation_profiles.get(str(profile), 0) + int((batch['_augmentation_profile'] == profile).sum())
                added_tokens += int(batch['_added_tokens'].sum())
                actual_augmented += int((batch['_added_tokens'] > 0).sum())
            scalars = {'loss': result['total'], **result['terms'],
                       **{f'weighted/{k}': v for k, v in result['weighted_terms'].items()}}
            for key, value in scalars.items():
                sums[key] = sums.get(key, 0.) + value.detach().item() * n
            with torch.no_grad():
                metrics = path_metrics(pred, batch['gt_path'])
                metrics.update({f'guide_{k}': v for k, v in path_metrics(batch['guide'], batch['gt_path']).items()})
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.) + value.sum().item()
                for nav, name in enumerate(('straight', 'left', 'right')):
                    mask = batch['navigation'] == nav
                    num = mask.sum().item()
                    if num:
                        entry = nav_sums.setdefault(name, {'samples': 0, 'ade_m': 0., 'fde_m': 0.})
                        entry['samples'] += num
                        for key in ('ade_m', 'fde_m'):
                            entry[key] += metrics[key][mask].sum().item()
    if not count:
        raise ValueError('Loader produced no samples')
    profile_names = {}
    if augmentation_profiles:
        for profile in augmentation_profiles:
            budget = loader.dataset.cfg.budgets[int(profile)]
            profile_names[profile] = 'original' if budget == 0 else 'full' if budget == -1 else f'k_{budget}'
    return {**{k: v / count for k, v in sums.items()}, 'samples': count,
            **({'augmentation': {'mean_added_tokens': added_tokens / count,
                                 'actual_augmented_fraction': actual_augmented / count,
                                 'profile_fraction': {profile_names[k]: v / count for k, v in augmentation_profiles.items()}}}
               if augmentation_profiles else {}),
            'by_navigation': {k: {**v, 'ade_m': v['ade_m'] / v['samples'],
                                      'fde_m': v['fde_m'] / v['samples']} for k, v in nav_sums.items()}}


def save_checkpoint(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def write_tensorboard(writer, record, stale, patience):
    def scalars(prefix, values):
        for key, value in values.items():
            if isinstance(value, dict):
                scalars(f'{prefix}/{key}', value)
            elif isinstance(value, (int, float)):
                writer.add_scalar(f'{prefix}/{key}', value, record['epoch'])
    scalars('train', record['train'])
    scalars('val_original', record['val'])
    for tag, value in [('optimizer/lr', record['lr']), ('time/epoch_seconds', record['seconds']),
                       ('selection/best_val_ade_m', record['best_val_ade_m']),
                       ('early_stop/stale_epochs', stale), ('early_stop/patience', patience)]:
        writer.add_scalar(tag, value, record['epoch'])
    writer.flush()


def train(cfg, resume=None):
    cfg.validate()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else cfg.device)
    splits = {'train': inspect_split(cfg.train_path), 'val': inspect_split(cfg.val_path)}
    overlap = set(splits['train']['worlds']) & set(splits['val']['worlds'])
    if overlap:
        raise ValueError(f'Train/val world overlap: {sorted(overlap)}')
    if cfg.augmentation is not None:
        splits['augmentation'] = inspect_context_cache(
            cfg.augmentation['cache_dir'], splits['train'], cfg.max_lines, cfg.max_points)
    # Fail before starting a run if logging was requested but is unavailable.
    writer_class = None
    if cfg.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer_class = SummaryWriter
        except ImportError as exc:
            raise RuntimeError('TensorBoard requested: install tensorboard in this Python environment or pass --no-tensorboard') from exc
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if resume is None and any((out / name).exists() for name in ('last.pt', 'best.pt', 'history.jsonl', 'config.json')):
        raise ValueError('Output already contains a run; use --resume or a new --output-dir')
    generator = torch.Generator().manual_seed(cfg.seed)
    val_generator = torch.Generator().manual_seed(cfg.seed + 1)
    train_loader = build_loader(cfg.train_path, cfg, True, generator, device)
    val_loader = build_loader(cfg.val_path, cfg, False, val_generator, device)
    model = PathGenModel(ModelConfig(**cfg.model)).to(device)
    initial_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial_digest.update(name.encode()); initial_digest.update(value.detach().cpu().numpy().tobytes())
    initial_model_sha256 = initial_digest.hexdigest()
    criterion = PathLoss(PathLossConfig(**cfg.loss)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    start, best, stale = 1, float('inf'), 0
    if resume:
        # Full optimizer/RNG state requires pickle: only load your own trusted checkpoints.
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        allowed_changes = {'epochs', 'device', 'num_workers', 'output_dir', 'tensorboard', 'tensorboard_dir', 'early_stop_patience'}
        current = json.loads(json.dumps(asdict(cfg)))
        previous = json.loads(json.dumps(asdict(TrainConfig(**checkpoint['config']))))
        if any(current[k] != previous[k] for k in current if k not in allowed_changes):
            raise ValueError('Resume config differs; only epochs/device/workers/output/logging/early-stop patience may change')
        if checkpoint['splits'] != splits:
            raise ValueError('Dataset changed since checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start, best, stale = checkpoint['epoch'] + 1, checkpoint['best_val_ade_m'], checkpoint['stale_epochs']
        initial_model_sha256 = checkpoint.get('initial_model_sha256', initial_model_sha256)
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        generator.set_state(checkpoint['loader_rng'])
        val_generator.set_state(checkpoint['val_loader_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    if start > cfg.epochs:
        raise ValueError(f'Checkpoint already reached epoch {start-1}; increase --epochs')
    if resume and cfg.early_stop_patience and stale >= cfg.early_stop_patience:
        raise ValueError('Checkpoint already meets early stopping; increase --early-stop-patience or set it to 0 to continue')
    (out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n')
    (out / 'data_manifest.json').write_text(json.dumps(splits, indent=2) + '\n')
    print(f'device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, '
          f'train={len(train_loader.dataset)}, val={len(val_loader.dataset)}', flush=True)
    writer = (writer_class(log_dir=cfg.tensorboard_dir or str(out / 'tensorboard'),
                           purge_step=start if resume else None) if writer_class else None)
    if writer:
        writer.add_text('configuration', json.dumps(asdict(cfg), indent=2), start-1)
    try:
        for epoch in range(start, cfg.epochs + 1):
            begin = time.monotonic()
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)
            train_metrics = run_epoch(model, criterion, train_loader, device, optimizer, cfg.grad_clip_norm)
            val_metrics = run_epoch(model, criterion, val_loader, device)
            score = val_metrics['ade_m']
            improved = score < best
            best, stale = (score, 0) if improved else (best, stale + 1)
            lr_used = optimizer.param_groups[0]['lr']
            scheduler.step(score)
            record = {'epoch': epoch, 'lr': lr_used, 'seconds': time.monotonic() - begin,
                      'train': train_metrics, 'val': val_metrics, 'best_val_ade_m': best}
            payload = {'epoch': epoch, 'config': asdict(cfg), 'splits': splits,
                       'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'best_val_ade_m': best,
                       'stale_epochs': stale, 'initial_model_sha256': initial_model_sha256, 'python_rng': random.getstate(),
                       'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                       'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                       'loader_rng': generator.get_state(), 'val_loader_rng': val_generator.get_state()}
            save_checkpoint(out / 'last.pt', payload)
            if improved:
                save_checkpoint(out / 'best.pt', payload)
            with (out / 'history.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            if writer:
                write_tensorboard(writer, record, stale, cfg.early_stop_patience)
            print(f'epoch {epoch}/{cfg.epochs} loss={train_metrics["loss"]:.5f} '
                  f'val_ADE={score:.4f}m val_FDE={val_metrics["fde_m"]:.4f}m '
                  f'guide_ADE={val_metrics["guide_ade_m"]:.4f}m lr={lr_used:.2g} '
                  f'time={record["seconds"]:.1f}s' + (' [best]' if improved else ''), flush=True)
            if cfg.early_stop_patience and stale >= cfg.early_stop_patience:
                print(f'Early stopping: no new best validation ADE for {stale} epochs.', flush=True)
                break
    finally:
        if writer:
            writer.close()
    print(f'Finished. Best validation ADE={best:.4f}m; checkpoints: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON config, paths relative to working directory')
    parser.add_argument('--resume', type=Path, help='Trusted checkpoint, normally last.pt')
    for name in ('train_path', 'val_path', 'output_dir', 'device', 'tensorboard_dir'):
        parser.add_argument('--' + name.replace('_', '-'))
    for name in ('epochs', 'batch_size', 'num_workers', 'early_stop_patience'):
        parser.add_argument('--' + name.replace('_', '-'), type=int)
    parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.config:
        raw = json.loads(args.config.read_text())
    elif args.resume:
        raw = torch.load(args.resume, map_location='cpu', weights_only=False)['config']
    else:
        raw = {}
    for key in ('model', 'loss'):
        if key in raw:
            defaults = asdict(ModelConfig() if key == 'model' else PathLossConfig())
            raw[key] = {**defaults, **raw[key]}
    for key in ('train_path', 'val_path', 'output_dir', 'device', 'epochs', 'batch_size', 'num_workers',
                'early_stop_patience', 'tensorboard', 'tensorboard_dir'):
        if getattr(args, key) is not None:
            raw[key] = getattr(args, key)
    train(TrainConfig(**raw), args.resume)


if __name__ == '__main__':
    main()
