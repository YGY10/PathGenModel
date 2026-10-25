import copy

import numpy as np
import torch
from shapely.geometry import LineString

from Dataset.dataset import encode_sample
from Scenario.build_dataset_v4 import DEFAULT_QUALITIES
from experiment_fov import ExperimentConfig, append_observations, supplement_candidates, fov_polygon


def sample():
    return {'id': 's', 'navigation': 'straight', 'reference_line': [[float(i), 0.] for i in range(81)],
            'gt_path': [[float(i), 0.] for i in range(81)],
            'lane_markings': [{'points': [[0., 1.], [5., 1.]], 'source_id': 'old'}],
            'road_boundaries': [], 'crosswalks': [], 'stop_lines': [],
            'metadata': {'ego_world_xy': [0., 0.], 'heading_rad': 0., 'pose_name': 'entry',
                         'quality': 'clean', 'vcs_fov_m': {'forward': 10., 'rear': 5., 'left': 5., 'right': 5.}}}


def test_original_slots_preserved_and_capacity_unchanged():
    base = encode_sample(sample())
    candidates = [('lane_marking', {'points': [[20., 2.], [30., 2.]], 'source_id': 'new'})] * 40
    augmented, kept = append_observations(base, candidates, 32)
    assert len(kept) == 31 and augmented['polyline_mask'].sum() == 32
    for key in ('bev_points', 'point_mask', 'polyline_attr'):
        assert torch.equal(base[key][0], augmented[key][0])
    for key in ('guide', 'gt_path', 'navigation'):
        assert torch.equal(base[key], augmented[key])


def test_only_new_region_and_deterministic_supplement():
    row = sample()
    world = {'metadata': {'seed': 3}, 'physical': {'elements': [
        {'id': 'line', 'type': 'line', 'points': [[0., 2.], [80., 2.]]},
        {'id': 'oldbox', 'type': 'stop_line', 'points': [[2., -1.], [2., 1.], [3., 1.], [3., -1.]]},
    ]}}
    # Zero dropout/noise isolates clipping. Quality parameters themselves remain intact.
    from dataclasses import replace
    quality = replace(DEFAULT_QUALITIES['clean'], line_miss_rate=0., box_miss_rate=0., break_rate=0.,
                      offset_sigma_m=0., sigma_m=0., angle_sigma_deg=0.)
    cfg = ExperimentConfig()
    original = copy.deepcopy(row)
    candidates, expanded = supplement_candidates(row, world, quality, cfg)
    again, _ = supplement_candidates(row, world, quality, cfg)
    assert candidates == again and candidates
    assert row == original and expanded['rear'] == 5.
    old_fov = fov_polygon(row['metadata']['vcs_fov_m'])
    for kind, obj in candidates:
        assert kind == 'lane_marking'
        assert LineString(obj['points']).intersection(old_fov).length < 1e-7


def test_full_capacity_is_a_noop():
    base = encode_sample(sample())
    base['point_mask'][:] = True
    base['polyline_mask'][:] = True
    candidate = [('lane_marking', {'points': [[20., 2.], [30., 2.]]})]
    augmented, kept = append_observations(base, candidate, 32)
    assert not kept
    for key in base:
        assert torch.equal(base[key], augmented[key])
