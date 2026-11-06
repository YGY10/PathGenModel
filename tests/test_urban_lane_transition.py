import numpy as np
import pytest
from Scenario.build_lane_transition import URBAN_PRESETS, make_urban_world, build_samples, smooth
from Scenario.build_dataset_v4 import DEFAULT_QUALITIES
from Dataset.dataset import encode_sample


@pytest.mark.parametrize('preset', URBAN_PRESETS)
def test_navigation_targets_reachable_and_geometrically_valid(preset):
    cfg,world,routes=make_urban_world(43000000,preset,marking_pattern='continuous')
    assert 35<=cfg.length_m<=50
    side=1 if cfg.change_side=='left' else -1
    for entry in set(r['entry_lane_index'] for r in routes.values()):
        by_nav={r['navigation']:r for r in routes.values() if r['entry_lane_index']==entry}
        assert set(by_nav)=={'left','straight','right'}
        targets={n:r['exit_lane_index'] for n,r in by_nav.items()}
        assert side*targets['left']>=side*targets['straight']>=side*targets['right']
        reachable=by_nav['straight']['reachable_exit_lanes']
        best=min(abs(cfg.lateral_shift_m+side*(j-entry)*cfg.width_m) for j in reachable)
        assert abs(cfg.lateral_shift_m+side*(targets['straight']-entry)*cfg.width_m)==pytest.approx(best)
    for route in routes.values():
        assert route['lane_adjustment_count']<=1
        assert route['exit_lane_index'] in route['reachable_exit_lanes']
        gt=np.asarray(route['gt_world']);base=cfg.lateral_shift_m*smooth(gt[:,0]/cfg.length_m)
        y=side*(gt[:,1]-base)
        width=cfg.width_m*(cfg.lanes_in+(cfg.lanes_out-cfg.lanes_in)*smooth(gt[:,0]/cfg.length_m))
        assert (y>=0).all() and (y<=width).all()
        assert np.abs(np.diff(gt[:,1])/np.diff(gt[:,0])).max()<=.3
        guide=np.asarray(route['guide_world'])
        assert abs(guide[-1,1]-gt[-1,1])<cfg.width_m/2
    for obj in world['physical']['elements']:
        if obj.get('role')=='continuing_separator':
            x=np.asarray(obj['points'])[:,0]
            assert x.min()<0 and x.max()>cfg.length_m
        if obj.get('role')=='added_separator':
            assert 0<obj['points'][0][0]<cfg.length_m
        if obj.get('role')=='ending_separator':
            assert 0<obj['points'][-1][0]<cfg.length_m


def test_urban_samples_keep_navigation_and_fixed_guides():
    cfg,w,routes=make_urban_world(43000000,URBAN_PRESETS[0])
    q={k:DEFAULT_QUALITIES[k] for k in ('clean','hard')}
    rows=build_samples(cfg,w,routes,q)
    assert set(r['navigation'] for r in rows)=={'left','straight','right'}
    assert rows==build_samples(cfg,w,routes,q)
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['reference_line']==b['reference_line'] and a['gt_path']==b['gt_path']
        assert a['navigation'] in a['route_id']
        e=encode_sample(a)
        assert e['guide'].shape==(81,2) and (e['gt_path'][0]==0).all()
    # Same entry before the transition: change nav/Guide, retain perception.
    before=[r for r in rows if r['metadata']['entry_lane_index']==0 and
            r['metadata']['pose_name']=='before_10' and r['metadata']['quality']=='clean']
    assert len(before)==3
    assert all(r['lane_markings']==before[0]['lane_markings'] and
               r['road_boundaries']==before[0]['road_boundaries'] for r in before)


@pytest.mark.parametrize('pattern', ['unmarked','partial','continuous'])
def test_physical_marking_gaps_do_not_change_targets_or_bridge_empty_region(pattern):
    cfg,base,base_routes=make_urban_world(43000000,URBAN_PRESETS[0],marking_pattern='continuous')
    _,world,routes=make_urban_world(43000000,URBAN_PRESETS[0],marking_pattern=pattern)
    assert routes==base_routes
    # Outer edge lines and curbs are identical across all marking patterns.
    outer=lambda w:[e for e in w['physical']['elements'] if e['type']=='curb' or e.get('role')=='outer_edge_marking']
    assert outer(world)==outer(base)
    md=world['metadata']
    assert md['marking_pattern']==pattern
    internal=[e for e in world['physical']['elements'] if e.get('role','').endswith('_separator')]
    if pattern=='unmarked':
        a,b=md['unmarked_core_world_x_m']
        assert 0<a<b<cfg.length_m and b-a>.45*cfg.length_m
        for e in internal:
            x=np.asarray(e['points'])[:,0]
            assert x.max()<=a or x.min()>=b
        # Actual model encoding must not rejoin disjoint pieces across the gap.
        rows=build_samples(cfg,world,routes,{'clean':DEFAULT_QUALITIES['clean']})
        row=next(r for r in rows if r['metadata']['pose_name']=='before_10')
        encoded=encode_sample(row)
        ego=np.asarray(row['metadata']['ego_world_xy'])
        h=row['metadata']['heading_rad'];rot=np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
        for i,attr in enumerate(encoded['polyline_attr']):
            if not attr[0] or not attr[8]:continue
            mask=encoded['point_mask'][i]
            p=encoded['bev_points'][i,mask,:2].numpy()@rot.T+ego
            assert p[:,0].max()<=a+.15 or p[:,0].min()>=b-.15
    elif pattern=='partial':
        assert md['continuous_separator_id']
        kept=next(e for e in internal if e['id']==md['continuous_separator_id'])
        assert kept['points'][0][0]<0 and kept['points'][-1][0]>cfg.length_m
        assert md['marking_gap_intervals']
    else:
        assert md['marking_gap_intervals']==[]
