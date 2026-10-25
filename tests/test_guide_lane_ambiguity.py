from argparse import Namespace

import numpy as np
from shapely.geometry import box

from audit_guide_lane_ambiguity import audit_guide, prediction_lane, road_distances
from World.road_warp import ArmWarp


def fixture():
    s=np.linspace(0,100,501)
    geo=dict(s=s,reference=np.column_stack((s,np.full_like(s,-1.75))),arc=s,
             warp=ArmWarp((0,0,0,0),(0,0,0,0)),width=3.5,arm=1,target=0,count=2,
             polygons=[box(0,-3.5,100,0),box(0,-7,100,-3.5)])
    args=Namespace(start_m=10.,end_m=40.,min_points=5,tolerance_m=.3,clear_fraction=.8)
    return geo,args


def test_known_lane_centers_boundary_and_horizon():
    geo,args=fixture()
    x=np.arange(10.,41.)
    for y,expected in [(-1.75,'target_clear'),(-3.458,'ambiguous_or_mixed'),(-5.25,'neighbor_clear')]:
        points=np.column_stack((x,np.full_like(x,y)))
        result,_=audit_guide(points,geo,args)
        assert result['group']==expected and result['checked_points']==31
        distances,arc=road_distances(points,geo)
        np.testing.assert_allclose(arc,x)
        np.testing.assert_allclose(distances[:,0],abs(y+1.75))
        np.testing.assert_allclose(distances[:,1],abs(y+5.25))
    result,_=audit_guide(np.column_stack((np.arange(5.),np.full(5,-1.75))),geo,args)
    assert result['group']=='insufficient_horizon'


def test_offroad_is_not_wrong_lane_and_single_lane_separate():
    geo,args=fixture();x=np.arange(10.,41.)
    points=np.column_stack((x,np.ones_like(x)))
    wrong,target,outside=prediction_lane(points,geo,args)
    assert wrong is False and target==0 and outside==1
    geo.update(count=1,polygons=geo['polygons'][:1])
    result,margin=audit_guide(points,geo,args)
    assert result['group']=='single_lane' and len(margin)==0


def test_incoming_arm_projection_is_excluded():
    geo,args=fixture()
    incoming=np.column_stack((np.arange(10.,41.),np.full(31,-100.)))
    outgoing=np.column_stack((np.arange(0.,61.),np.full(61,-1.75)))
    result,_=audit_guide(np.concatenate([incoming,outgoing]),geo,args)
    assert result['checked_points']==31
    assert result['group']=='target_clear'
    assert result['target_distance_mean_m']==0
