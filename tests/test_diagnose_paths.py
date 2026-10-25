import numpy as np
from shapely.geometry import Point

from diagnose_paths import components, tangent, to_world, target_lane_polygon
from Dataset.audit_lane_visibility import transform


def test_signed_residual_frame_and_world_roundtrip():
    guide = np.column_stack([np.arange(5.), np.zeros(5)])
    direction = tangent(guide)
    delta = np.tile([-0.2, 0.5], (5, 1))
    np.testing.assert_allclose(components(delta, direction), delta)
    md = {'heading_rad': .7, 'ego_world_xy': [12., -30.]}
    np.testing.assert_allclose(transform(to_world(guide, md), md['ego_world_xy'], md['heading_rad']), guide, atol=1e-12)


def test_target_lane_polygon_uses_lane_identity():
    world = {'metadata': {'lane_width_m': 3.5, 'entry_distances_m': [10.] * 4, 'arm_length_m': 180.,
                          'arm_warp': {'angle_deg': [0.] * 4, 'lateral_m': [0.] * 4,
                                       'start_m': 13., 'full_m': 65.}}}
    polygon = target_lane_polygon(world, {'exit_arm': 1, 'exit_lane_index': 1})
    assert polygon.covers(Point(20., -5.25))
    assert not polygon.covers(Point(20., -1.75))
    assert not polygon.covers(Point(5., -5.25))
