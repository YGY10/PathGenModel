import copy
import json

import numpy as np
import pytest
import torch

from Scenario.build_dataset import at_s
from Scenario.build_dataset_v4 import (
    GUIDE_CONVENTION, DEFAULT_QUALITIES, build_one, sample_world_guide, verify,
)
from Scenario.reexport_guides import replace_guide, non_guide_payload, reexport
from World.intersection_generator import IntersectionConfig
from Model.pathgen_model_v1 import ModelConfig, PathGenModel
from Dataset.audit_extended import audit


def test_constant_offset_survives_multiple_ego_poses_and_headings():
    world_guide = np.column_stack([np.arange(201.), np.full(201, 2.)])
    for ego in (np.array([20., 0.]), np.array([40., 0.])):
        local, station = sample_world_guide(world_guide, ego, np.array([1., 0.]))
        np.testing.assert_allclose(local[:, 1], 2.)
        assert station == ego[0]
    local, _ = sample_world_guide(world_guide, np.array([20., 0.]), np.array([0., 1.]))
    np.testing.assert_allclose(local[0], [2., 0.], atol=1e-9)


def test_generated_guides_reconstruct_one_world_route_and_reexport(tmp_path):
    world, rows, routes = build_one(30000000, IntersectionConfig(), {'medium': DEFAULT_QUALITIES['medium']})
    verify(rows)
    for row in rows:
        md = row['metadata']
        assert md['guide_coordinate_convention'] == GUIDE_CONVENTION
        h = md['heading_rad']
        t = np.array([np.cos(h), np.sin(h)])
        local = np.asarray(row['reference_line'])
        recovered = local[:, :1] * t + local[:, 1:] * [-t[1], t[0]] + md['ego_world_xy']
        expected = at_s(routes[row['navigation']]['guide_world'], md['guide_start_s_m'] + np.arange(81))
        np.testing.assert_allclose(recovered, expected, atol=8e-5)
        fixed, error = replace_guide(row, routes[row['navigation']])
        assert error < 1e-8
        assert non_guide_payload(fixed) == non_guide_payload(row)
    assert all(np.linalg.norm(r['reference_line'][0]) > .5 for r in rows if r['metadata']['pose_name'] == 'exit_5')
    source = tmp_path / 'source'
    (source / 'worlds').mkdir(parents=True)
    (source / 'routes').mkdir()
    wid = world['world_id']
    (source / 'worlds' / f'{wid}.json').write_text(json.dumps(world))
    serial = {k: {field: value.tolist() if isinstance(value, np.ndarray) else value for field, value in rt.items()}
              for k, rt in routes.items()}
    (source / 'routes' / f'{wid}.json').write_text(json.dumps(serial))
    (source / 'config.json').write_text('{}')
    old_rows = copy.deepcopy(rows)
    for row in old_rows:
        ref = np.asarray(row['reference_line'])
        row['reference_line'] = (ref - ref[0]).tolist()
    (source / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in old_rows))
    report = reexport(source, tmp_path / 'fixed')
    assert report['samples'] == len(rows) and report['non_guide_fields_identical']
    check = audit([fixed], source / 'worlds', source / 'routes', tmp_path / 'audit_valid')
    assert check['issues'] == 0
    corrupted = copy.deepcopy(fixed)
    corrupted['reference_line'][20][1] += .5
    check = audit([corrupted], source / 'worlds', source / 'routes', tmp_path / 'audit_invalid')
    assert check['issues'] == 1
    assert 'fixed_guide_world_consistency' in (tmp_path / 'audit_invalid' / 'extended_issues.csv').read_text()
    with pytest.raises(ValueError):
        reexport(source, tmp_path / 'fixed')


def test_nonzero_guide_start_model_output_and_backward():
    model = PathGenModel(ModelConfig(d_model=16, nhead=2, encoder_layers=1, decoder_layers=1, ff_dim=32))
    points = torch.zeros(1, 2, 4, 6)
    mask = torch.zeros(1, 2, 4, dtype=torch.bool)
    guide = torch.stack([torch.arange(81.), torch.full((81,), 2.)], -1).unsqueeze(0)
    pred = model(points, mask, mask.any(-1), torch.zeros(1, 2, 10), guide, torch.tensor([0]))
    torch.testing.assert_close(pred[:, 1:], guide[:, 1:])
    assert (pred[:, 0] == 0).all()
    pred.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
