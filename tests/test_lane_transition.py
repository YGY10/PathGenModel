import numpy as np
import pytest
from dataclasses import replace

from Dataset.dataset import encode_sample
from Scenario.build_dataset import arclength
from Scenario.build_dataset_v4 import DEFAULT_QUALITIES, sample_world_guide
from Scenario.build_lane_transition import (PRESETS, POSES, build_samples, make_world,
                                            linear_guide_anchors, connection_curve)


@pytest.mark.parametrize('preset', PRESETS)
def test_geometry_connections_and_model_inputs(preset):
    cfg, world, routes = make_world(42, preset)
    assert 1 <= cfg.lanes_in <= 4 and 1 <= cfg.lanes_out <= 4
    assert 0 < abs(cfg.lanes_in-cfg.lanes_out) <= 2
    assert 15 <= cfg.length_m <= 25
    assert len(routes) == max(preset[:2])
    rows = build_samples(cfg, world, routes, {'clean': DEFAULT_QUALITIES['clean']})
    assert len(rows) == len(routes)*len(POSES)
    for row in rows:
        encoded = encode_sample(row)
        assert encoded['guide'].shape == (81, 2)
        assert encoded['gt_path'][0].abs().max() == 0
        assert encoded['polyline_mask'].any()
        # A 1m arc step cannot have a chord longer than 1m (apart from rounding).
        assert np.linalg.norm(np.diff(row['gt_path'],axis=0),axis=1).max() < 1.001
    for rt in routes.values():
        gt=np.asarray(rt['gt_world'])
        assert arclength(gt)[-1] > 300
        side=1 if cfg.change_side=='left' else -1
        assert gt[0,1] == pytest.approx(side*(rt['entry_lane_index']+.5)*cfg.width_m)
        assert gt[-1,1] == pytest.approx(cfg.lateral_shift_m+side*(rt['exit_lane_index']+.5)*cfg.width_m)
    for obj in world['physical']['elements']:
        if obj['type']=='line' and obj.get('role')!='outer_edge_marking':
            x=np.asarray(obj['points'])[:,0]
            assert x.max() <= 0 or x.min() >= cfg.length_m
    assert sum(e['type']=='line' and e['id'].startswith('before') for e in world['physical']['elements']) == cfg.lanes_in-1
    assert sum(e['type']=='line' and e['id'].startswith('after') for e in world['physical']['elements']) == cfg.lanes_out-1
    elements={e['id']:e for e in world['physical']['elements']}
    for name in ('fixed_edge','changing_edge'):
        curb=elements[name];mark=elements[name+'_marking']
        assert curb['type']=='curb' and mark['type']=='line'
        assert mark['style']=='solid' and mark['color']=='white'
        np.testing.assert_allclose(np.abs(np.asarray(curb['points'])[:,1]-np.asarray(mark['points'])[:,1]),cfg.edge_line_inset_m)


def test_fixed_world_guide_and_quality_independence():
    cfg,world,routes=make_world(43,PRESETS[0])
    qualities={k:DEFAULT_QUALITIES[k] for k in ('clean','hard')}
    rows=build_samples(cfg,world,routes,qualities)
    again=build_samples(cfg,world,routes,qualities)
    assert rows == again
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['reference_line']==b['reference_line']
        assert a['gt_path']==b['gt_path']
        md=a['metadata']
        rt=routes[a['route_id'].split(world['world_id']+'_',1)[1]]
        heading=np.array([np.cos(md['heading_rad']),np.sin(md['heading_rad'])])
        expected,_=sample_world_guide(rt['guide_world'],np.asarray(md['ego_world_xy']),heading)
        np.testing.assert_allclose(a['reference_line'],expected,atol=5.1e-5)
    assert any(np.linalg.norm(r['reference_line'][0])>.1 for r in rows)


def test_piecewise_linear_guide_preserves_corners_and_gt():
    cfg,world,routes=make_world(41000000,PRESETS[0])
    old_cfg,old_world,old_routes=make_world(41000000,PRESETS[0],guide_shape='smooth')
    assert world['physical'] == old_world['physical']
    for rid,rt in routes.items():
        assert rt['gt_world'] == old_routes[rid]['gt_world']
        entry,exit_=rt['entry_lane_index'],rt['exit_lane_index']
        anchors=linear_guide_anchors(cfg,entry,exit_)
        guide=np.asarray(rt['guide_world'])
        for a,b in zip(anchors[:-1],anchors[1:]):
            p=guide[(guide[:,0]>=a[0])&(guide[:,0]<=b[0])]
            v=b-a
            cross=(p[:,0]-a[0])*v[1]-(p[:,1]-a[1])*v[0]
            np.testing.assert_allclose(cross,0,atol=1e-9)
        # Same function still constructs GT identically for either Guide mode.
        np.testing.assert_array_equal(connection_curve(cfg,entry,exit_),
                                      connection_curve(replace(cfg,guide_shape='smooth'),entry,exit_))
    anchors=linear_guide_anchors(cfg,0,2)
    angles=np.arctan2(np.diff(anchors,axis=0)[:,1],np.diff(anchors,axis=0)[:,0])
    assert np.max(np.abs(np.diff(angles)))>.1


def test_all_missed_observations_are_valid_for_bulk_generation():
    cfg,world,routes=make_world(42,PRESETS[0])
    missed=replace(DEFAULT_QUALITIES['hard'],line_miss_rate=1.,curb_miss_rate=1.,box_miss_rate=1.)
    rows=build_samples(cfg,world,routes,{'hard':missed})
    assert len(rows)==len(routes)*len(POSES)
    assert all(not encode_sample(r)['polyline_mask'].any() for r in rows)
