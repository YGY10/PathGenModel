import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from dataclasses import asdict
from Model.pathgen_model_v1 import ModelConfig, PathGenModel
from evaluate import evaluate, select_examples
from train import inspect_split


def test_evaluation_export_and_plots(tmp_path):
    path = tmp_path / 'val.jsonl'
    rows = [{'id': f'v{i}', 'world_id': 'val_world', 'navigation': 'left',
             'reference_line': [[float(j), 0.] for j in range(81)],
             'gt_path': [[float(j), 0.] for j in range(81)],
             'stop_lines': [{'points': [[3., -1.], [3., 1.], [4., 1.], [4., -1.]]}],
             'metadata': {'quality': 'clean', 'pose_name': 'entry'}} for i in range(2)]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    cfg = ModelConfig(d_model=16, nhead=2, encoder_layers=1, decoder_layers=1, ff_dim=32)
    checkpoint = tmp_path / 'best.pt'
    torch.save({'model': PathGenModel(cfg).state_dict(), 'epoch': 1,
                'config': {'model': asdict(cfg), 'val_path': str(path), 'max_lines': 3, 'max_points': 4},
                'splits': {'train': {'worlds': ['train_world']}, 'val': inspect_split(path)}}, checkpoint)
    args = Namespace(checkpoint=checkpoint, split='val', input=None, output_dir=tmp_path/'eval',
                     device='cpu', batch_size=1, num_workers=0, empty_bev=True, plots=True,
                     examples_per_group=1, seed=42)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        summary = evaluate(args)
        assert summary['overall']['samples'] == 2
        assert summary['overall']['ade_m'] == 0
        assert summary['overall']['ade_improvement_pct'] is None
        with np.load(args.output_dir/'predictions.npz') as arrays:
            assert arrays['pred'].shape == (2, 81, 2)
            assert arrays['ids'].tolist() == ['v0', 'v1']
        assert (args.output_dir/'overview.png').stat().st_size > 0
        assert list((args.output_dir/'samples').glob('*.png'))
        assert len(select_examples([{'index': i, 'ade_m': i, 'navigation': 'left'} for i in range(3)], 0, 42)) == 0
        with pytest.raises(ValueError, match='already exists'):
            evaluate(args)
    finally:
        torch.set_num_threads(threads)
