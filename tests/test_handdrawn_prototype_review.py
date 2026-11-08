"""Check independent references, shared addition World and uncropped observations."""
import copy

import numpy as np
from shapely.geometry import LineString, Polygon

from scripts.build_handdrawn_prototype_review import SPECS, geometry, make_plan, simulate


def test_candidate_references_and_shared_addition_world():
    additions = []
    for _, _, kind, _, parameters in SPECS:
        world, gt = geometry(kind, parameters)
        assert gt.shape == (81, 2)
        np.testing.assert_allclose(gt[0], [0, 0], atol=1e-8)
        assert Polygon(world['road_polygon']).covers(LineString(gt))
        assert np.min(np.linalg.norm(np.diff(gt, axis=0), axis=1)) > .98
        if kind.startswith('addition'):
            additions.append((world, gt))
    assert all(w['elements'] == additions[0][0]['elements'] for w, _ in additions)
    assert len({round(float(gt[-1, 1]), 3) for _, gt in additions}) == 3


def test_simulation_keeps_guide_and_observations_beyond_old_fov():
    # A visible line at y=30 must not be removed by the former +/-22m FOV.
    points = [[0., 30.], [80., 30.]]
    observed = dict(reference_line=[[0., 0.], [80., 0.]], navigation='straight',
                    lane_markings=[dict(id='visible', points=points, line_style='solid')],
                    road_boundaries=[], stop_lines=[], crosswalks=[])
    original = copy.deepcopy(observed)
    world = dict(elements=[dict(id='physical', type='line', points=points)])
    plan = make_plan(observed, world)
    result, events = simulate(observed, world, plan, seed=5)
    assert result == simulate(observed, world, plan, seed=5)[0]
    assert observed == original
    assert result['reference_line'] == observed['reference_line']
    assert result['navigation'] == observed['navigation']
    assert result['lane_markings']
    assert all(np.min(np.asarray(line['points'])[:, 1]) > 22
               for line in result['lane_markings'])
    assert all(event['source_id'] == 'physical' for event in events)
