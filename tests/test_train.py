from dataclasses import replace
import json

import pytest
import torch

from train import TrainConfig, path_metrics, train


def test_metrics():
    gt = torch.zeros(2, 81, 2)
    pred = gt.clone()
    pred[:, 1:, 0] = 3.
    metrics = path_metrics(pred, gt)
    for value in metrics.values():
        torch.testing.assert_close(value, torch.full((2,), 3.))


def test_training_resume_and_split_guard(tmp_path):
    paths = []
    for split in ('train', 'val'):
        path = tmp_path / f'{split}.jsonl'
        rows = []
        for i in range(3):
            rows.append({'world_id': split, 'id': f'{split}_{i}', 'navigation': 'left',
                         'reference_line': [[float(j), 0.] for j in range(81)],
                         'gt_path': [[float(j), min(j / 10., 1.)] for j in range(81)],
                         'lane_markings': [{'points': [[0., 2.], [80., 2.]]}]})
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        paths.append(str(path))
    cfg = TrainConfig(train_path=paths[0], val_path=paths[1], output_dir=str(tmp_path / 'resumed'),
                      epochs=1, batch_size=2, device='cpu', max_lines=3, max_points=4,
                      model={'d_model': 16, 'nhead': 2, 'ff_dim': 32,
                             'encoder_layers': 1, 'decoder_layers': 1, 'dropout': .1})
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train(cfg)
        checkpoint_path = tmp_path / 'resumed' / 'last.pt'
        train(replace(cfg, epochs=2), checkpoint_path)
        train(replace(cfg, epochs=2, output_dir=str(tmp_path / 'full')))
        resumed = torch.load(checkpoint_path, weights_only=False)
        full = torch.load(tmp_path / 'full' / 'last.pt', weights_only=False)
        for key in full['model']:
            torch.testing.assert_close(resumed['model'][key], full['model'][key], rtol=0, atol=0)
        assert (tmp_path / 'resumed' / 'best.pt').exists()
        records = [json.loads(s) for s in (tmp_path / 'resumed' / 'history.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in records] == [1, 2]
        assert records[0]['val']['samples'] == 3
        with pytest.raises(ValueError, match='overlap'):
            train(replace(cfg, val_path=paths[0]))
        with pytest.raises(ValueError, match='already contains'):
            train(cfg)
    finally:
        torch.set_num_threads(threads)
