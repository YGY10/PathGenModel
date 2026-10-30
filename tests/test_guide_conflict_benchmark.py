"""Geometry and paired-input integrity checks for the stress benchmark."""
import copy
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, box
from scripts.evaluate_guide_conflicts import (
    perturb_canonical, path_geometry_metrics, non_guide_fingerprint,
    build_guides, variants,
)
from Scenario.build_dataset_v4 import build_one, DEFAULT_QUALITIES, sample_world_guide
from World.intersection_generator import IntersectionConfig


def test_segment_crossing_is_detected_even_when_endpoints_are_inside():
    # U-shaped road: endpoints inside opposite arms, connecting chord crosses gap.
    road=shapely.union_all([box(0,0,2,8),box(6,0,8,8),box(0,0,8,2)])
    shapely.prepare(road)
    path=np.array([[1.,6.],[7.,6.]])
    curb=LineString([[2,2],[2,8]])
    metrics=path_geometry_metrics(path,road,road.buffer(.08),[curb])
    assert metrics['curb_crossing'] and metrics['road_violation']
    assert metrics['max_penetration_m']==pytest.approx(2.,abs=.051)
    assert metrics['outside_road_length_m']>3.8


def test_tangent_is_not_a_crossing_and_small_numeric_exit_is_tolerated():
    road=box(-10,-1,10,1);shapely.prepare(road)
    path=np.array([[-5.,1.],[0.,1.],[5.,1.]])
    metrics=path_geometry_metrics(path,road,road.buffer(.08),[LineString([[-10,1],[10,1]])])
    assert not metrics['curb_crossing'] and not metrics['road_violation']
    assert metrics['max_penetration_m']==0
    path[:,1]=1.04
    metrics=path_geometry_metrics(path,road,road.buffer(.08),[])
    assert not metrics['road_violation']
    assert metrics['max_penetration_m']==pytest.approx(.04)


def test_guide_only_fingerprint_catches_geometry_or_gt_changes():
    row={'reference_line':[[0.,0.],[1.,0.]],'gt_path':[[0.,0.],[1.,1.]],
         'road_boundaries':[{'points':[[0,2],[10,2]]}],'metadata':{'pose_name':'entry'}}
    changed=copy.deepcopy(row);changed['reference_line'][1][1]=3
    assert non_guide_fingerprint(changed)==non_guide_fingerprint(row)
    changed['gt_path'][1][1]=2
    assert non_guide_fingerprint(changed)!=non_guide_fingerprint(row)


def test_zero_and_signed_spatial_perturbations():
    # Straight north segment: positive timing moves turn vicinity north;
    # positive lateral moves to its left. Far ends stay fixed for timing.
    p=np.column_stack([np.zeros(241),np.arange(241.)])
    np.testing.assert_array_equal(perturb_canonical(p,80.,120.,0.,0.),p)
    plus=perturb_canonical(p,80.,120.,6.,0.)
    minus=perturb_canonical(p,80.,120.,-6.,0.)
    assert plus[100,1]-p[100,1]==pytest.approx(6.)
    assert minus[100,1]-p[100,1]==pytest.approx(-6.)
    np.testing.assert_array_equal(plus[[0,-1]],p[[0,-1]])
    lateral=perturb_canonical(p,80.,120.,0.,2.)
    np.testing.assert_allclose(lateral[:,0],-2.)


def test_saved_world_guide_reconstruction_and_shared_pose_crop():
    world,rows,routes=build_one(30000000,IntersectionConfig(),{'clean':DEFAULT_QUALITIES['clean']})
    route=routes['right'];route={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in route.items()}
    guides,rejected,closure=build_guides(world,route,'right')
    assert closure<2e-5
    assert 'original' in guides
    assert len(guides)+len(rejected)==len(variants())
    for row in [r for r in rows if r['navigation']=='right']:
        md=row['metadata'];h=md['heading_rad'];t=np.array([np.cos(h),np.sin(h)])
        original,_=sample_world_guide(guides['original'],np.array(md['ego_world_xy']),t)
        np.testing.assert_allclose(original,row['reference_line'],atol=1e-4)
        for guide in guides.values():
            crop,_=sample_world_guide(guide,np.array(md['ego_world_xy']),t)
            assert crop.shape==(81,2) and np.isfinite(crop).all()
            assert np.linalg.norm(np.diff(crop,axis=0),axis=1).max()<1.001


def test_visible_curb_constraints_discount_fragments_intersecting_gt():
    import torch
    from scripts.evaluate_guide_conflicts import observed_curb_metrics
    encoded={'bev_points':torch.zeros(2,4,6),'point_mask':torch.zeros(2,4,dtype=torch.bool),
             'polyline_attr':torch.zeros(2,10)}
    encoded['point_mask'][:,:2]=True;encoded['polyline_attr'][:,1]=1
    encoded['bev_points'][0,:2,:2]=torch.tensor([[2.,-2.],[2.,2.]])
    encoded['bev_points'][1,:2,:2]=torch.tensor([[.5,-2.],[.5,2.]])
    gt=np.array([[0.,0.],[1.,0.]])
    guide=np.array([[0.,0.],[3.,0.]])
    corrected=np.array([[0.,0.],[1.,0.]])
    m=observed_curb_metrics(guide,corrected,gt,encoded)
    assert m['observed_curb_fragments']==2
    assert m['gt_clear_observed_curb_fragments']==1
    assert m['guide_observed_curb_conflict']
    assert not m['observed_curb_crossing']
    assert m['observed_curb_crossing_raw']
