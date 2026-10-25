import numpy as np
import pytest

from diagnose_pose_progression import at_world_stations, observed_support


def test_same_world_station_and_longitudinal_error_preserved():
    stations=np.arange(140.,161.)
    for ego_s in (100.,108.,113.):
        metadata={'ego_route_s_m':ego_s,'ego_world_xy':[ego_s,0.],'heading_rad':0.}
        gt=np.column_stack((np.arange(81.),np.zeros(81)))
        pred=gt+np.array([5.,2.])
        expected=np.column_stack((stations,np.zeros(len(stations))))
        np.testing.assert_allclose(at_world_stations(gt,metadata,stations),expected)
        error=at_world_stations(pred,metadata,stations)-expected
        np.testing.assert_allclose(error,np.tile([5.,2.],(len(stations),1)))
    with pytest.raises(ValueError,match='outside'):
        at_world_stations(gt,metadata,np.array([100.,200.]))


def test_support_uses_actual_sources_and_encoded_geometry():
    x=np.arange(3.,31.)
    expected={'left':np.column_stack((x,np.full_like(x,1.75))),
              'right':np.column_stack((x,np.full_like(x,-1.75)))}
    md={'ego_world_xy':[0.,0.],'heading_rad':0.}
    left=('lane_marking',{'source_id':'left','points':[[0,1.75],[40,1.75]]})
    right=('road_boundary',{'source_id':'right','points':[[0,-1.75],[40,-1.75]]})
    mask=observed_support(expected,[left],md,1.25)
    assert mask[0].all() and not mask[1].any()
    assert observed_support(expected,[left,right],md,1.25).all()
    wrong=('lane_marking',{'source_id':'unrelated','points':left[1]['points']})
    assert not observed_support(expected,[wrong],md,1.25).any()
