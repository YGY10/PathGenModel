from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import train as training
from Dataset.dataset import encode_sample
from Dataset.context_augmentation import (
    ContextAugmentationConfig, ContextAugmentedDataset, inspect_context_cache,
)
from Scenario.build_dataset_v4 import DEFAULT_QUALITIES
from Scenario.build_fov_supplements import build_cache
from Scenario.fov_supplement import supplement_candidates as shared_supplement
from experiment_fov import supplement_candidates as original_supplement, ExperimentConfig


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'data'
    (root / 'worlds').mkdir(parents=True)
    q = replace(DEFAULT_QUALITIES['clean'], line_miss_rate=0., break_rate=0.)
    (root / 'config.json').write_text(json.dumps({'qualities': {'clean': asdict(q)}}))
    world = {'metadata': {'seed': 7}, 'physical': {'elements': [
        {'id': 'line', 'type': 'line', 'points': [[0., 2.], [80., 2.]]}]}}
    (root / 'worlds' / 'train_world.json').write_text(json.dumps(world))
    paths, train_rows = [], []
    for split in ('train', 'val'):
        path = root / f'{split}.jsonl'
        rows = []
        for i in range(4):
            rows.append({'id': f'{split}_{i}', 'world_id': f'{split}_world', 'navigation': 'straight',
                         'reference_line': [[float(j), 0.] for j in range(81)],
                         'gt_path': [[float(j), min(j / 10., 1.)] for j in range(81)],
                         'lane_markings': [{'points': [[0., 1.], [8., 1.]], 'source_id': 'old'}],
                         'metadata': {'ego_world_xy': [0., 0.], 'heading_rad': 0., 'pose_name': 'entry',
                                      'quality': 'clean', 'vcs_fov_m': {'forward': 10., 'rear': 5., 'left': 5., 'right': 5.}}})
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        paths.append(path)
        if split == 'train':
            train_rows = rows
    cache = tmp_path / 'cache'
    build_cache(paths[0], root / 'worlds', cache, seeds=[1031, 1032], workers=1)
    return paths, cache, train_rows, world, q


def test_cache_membership_shared_sampler_and_preservation(source):
    paths, cache, rows, world, q = source
    cfg = ContextAugmentationConfig(str(cache), budgets=(-1,), probabilities=(1.,))
    inspect_context_cache(cache, training.inspect_split(paths[0]), 32, 32)
    with pytest.raises(ValueError, match='match'):
        inspect_context_cache(cache, training.inspect_split(paths[1]), 32, 32)
    shared = shared_supplement(rows[0], world, q, ExperimentConfig())
    original = original_supplement(rows[0], world, q, ExperimentConfig())
    assert shared == original
    dataset = ContextAugmentedDataset(paths[0], cfg)
    dataset.set_epoch(7)
    base = encode_sample(rows[0])
    augmented = dataset[0]
    assert augmented['_added_tokens'] > 0
    for key in ('bev_points', 'point_mask', 'polyline_attr'):
        assert torch.equal(base[key][base['polyline_mask']], augmented[key][base['polyline_mask']])
    for key in ('guide', 'gt_path', 'navigation'):
        assert torch.equal(base[key], augmented[key])
    before = {key: value.clone() for key, value in augmented.items()}
    dataset.set_epoch(8); dataset.set_epoch(7)
    for key in before:
        assert torch.equal(before[key], dataset[0][key])
    for workers in (0, 2):
        batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=workers,
                                     multiprocessing_context='spawn' if workers else None)))
        for key in before:
            assert torch.equal(batch[key][0], before[key])


def test_augmented_training_exact_resume(source, tmp_path):
    paths, cache, *_ = source
    cfg = training.TrainConfig(train_path=str(paths[0]), val_path=str(paths[1]), output_dir=str(tmp_path / 'resumed'),
        epochs=1, batch_size=2, device='cpu', num_workers=2,
        augmentation={'cache_dir': str(cache)},
        model={'d_model': 16, 'nhead': 2, 'ff_dim': 32, 'encoder_layers': 1, 'decoder_layers': 1, 'dropout': .1})
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        training.train(cfg)
        training.train(replace(cfg, epochs=2), tmp_path / 'resumed' / 'last.pt')
        training.train(replace(cfg, epochs=2, output_dir=str(tmp_path / 'full')))
        a = torch.load(tmp_path / 'resumed' / 'last.pt', weights_only=False)
        b = torch.load(tmp_path / 'full' / 'last.pt', weights_only=False)
        for key in a['model']:
            torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
        logs = [json.loads(line) for line in (tmp_path / 'full' / 'history.jsonl').read_text().splitlines()]
        assert 'augmentation' in logs[0]['train'] and 'augmentation' not in logs[0]['val']
        assert 'augmentation' in a['splits']
    finally:
        torch.set_num_threads(threads)


def test_tensorboard_and_early_stopping(source, tmp_path, monkeypatch):
    pytest.importorskip('tensorboard')
    paths, _, *_ = source
    def constant_epoch(*args, **kwargs):
        return {'loss': 1., 'ade_m': .5, 'fde_m': .7, 'guide_ade_m': 1., 'samples': 4}
    monkeypatch.setattr(training, 'run_epoch', constant_epoch)
    out = tmp_path / 'early'
    cfg = training.TrainConfig(train_path=str(paths[0]), val_path=str(paths[1]), output_dir=str(out),
                              epochs=10, early_stop_patience=2, tensorboard=True,
                              model={'d_model': 16, 'nhead': 2, 'encoder_layers': 1, 'decoder_layers': 1})
    training.train(cfg)
    saved = torch.load(out / 'last.pt', weights_only=False)
    best = torch.load(out / 'best.pt', weights_only=False)
    assert saved['epoch'] == 3 and saved['stale_epochs'] == 2 and best['epoch'] == 1
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(out / 'tensorboard')); events.Reload()
    assert [e.step for e in events.Scalars('val_original/ade_m')] == [1, 2, 3]
    assert events.Scalars('early_stop/stale_epochs')[-1].value == 2
    with pytest.raises(ValueError, match='already meets'):
        training.train(replace(cfg, epochs=12), out / 'last.pt')
    training.train(replace(cfg, epochs=4, early_stop_patience=5), out / 'last.pt')
    events = EventAccumulator(str(out / 'tensorboard')); events.Reload()
    assert [e.step for e in events.Scalars('val_original/ade_m')] == [1, 2, 3, 4]
    resumed = torch.load(out / 'last.pt', weights_only=False)
    assert resumed['stale_epochs'] == 3


def test_cache_tamper_is_rejected(source):
    paths, cache, *_ = source
    with (cache / 'seed_1031.jsonl').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='fingerprint'):
        inspect_context_cache(cache, training.inspect_split(paths[0]), 32, 32)
