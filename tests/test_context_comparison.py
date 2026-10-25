from argparse import Namespace
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from Model.pathgen_model_v1 import ModelConfig, PathGenModel
from Scenario.build_dataset_v4 import build_one, DEFAULT_QUALITIES
from World.intersection_generator import IntersectionConfig
from train import TrainConfig, inspect_split
from evaluate_context_comparison import run


def test_fixed_input_comparison_and_initialization_guard(tmp_path):
    world, samples, routes = build_one(30000000, IntersectionConfig(), {'medium': DEFAULT_QUALITIES['medium']})
    samples = samples[:2]
    data = tmp_path / 'val.jsonl'
    data.write_text(''.join(json.dumps(r) + '\n' for r in samples))
    worlds, route_dir = tmp_path / 'worlds', tmp_path / 'routes'
    worlds.mkdir(); route_dir.mkdir()
    wid = world['world_id']
    (worlds / f'{wid}.json').write_text(json.dumps(world))
    serial = {k: {key: v.tolist() if isinstance(v, np.ndarray) else v for key, v in rt.items()} for k, rt in routes.items()}
    (route_dir / f'{wid}.json').write_text(json.dumps(serial))
    manifest = inspect_split(data)
    supplements = tmp_path / 'supplements'; supplements.mkdir()
    (supplements / 'summary.json').write_text(json.dumps({'input': str(data), 'data_manifest': manifest, 'config': {'seed': 724}}))
    additions = [{'id': r['id'], 'added_objects': [{'kind': 'lane_marking', 'points': [[100., 2.], [110., 2.]]}]} for r in samples]
    (supplements / 'added_observations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in additions))
    model_cfg = ModelConfig(d_model=16, nhead=2, ff_dim=32, encoder_layers=1, decoder_layers=1)
    model = PathGenModel(model_cfg)
    paths = []
    for name, augmentation in [('control', None), ('mixed', {'cache_dir': 'not_used_for_inference'})]:
        cfg = {**asdict(TrainConfig()), 'model': asdict(model_cfg), 'augmentation': augmentation}
        saved = {'config': cfg, 'model': model.state_dict(), 'epoch': 1,
                 'splits': {'train': {'worlds': ['other_train_world']}, 'val': manifest},
                 'initial_model_sha256': 'matched_test_initialization'}
        path = tmp_path / f'{name}.pt'; torch.save(saved, path); paths.append(path)
    args = Namespace(control=paths[0], mixed=paths[1], supplement_dirs=[supplements],
                     worlds_dir=worlds, routes_dir=route_dir, output_dir=tmp_path / 'evaluation',
                     device='cpu', batch_size=2)
    report = run(args)
    assert report['models']['control']['inputs']['original']['overall']['samples'] == 2
    assert all(value == 0 for values in report['paired_comparisons'].values() for value in values.values())
    saved = torch.load(paths[1], weights_only=False); saved['initial_model_sha256'] = 'different'; torch.save(saved, paths[1])
    with pytest.raises(ValueError, match='initialization'):
        run(args)
