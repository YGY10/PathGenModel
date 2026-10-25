import numpy as np

from Scenario import build_dataset_v4 as generator
from Scenario.build_dataset import POSES, V4_POSES, pose_seed_index, at_s
from Scenario.fov_supplement import SupplementConfig, supplement_candidates
from World.intersection_generator import IntersectionConfig


def test_inside13_strict_exit_condition_and_legacy_rng_ids():
    assert [pose_seed_index(p) for p in POSES]==list(range(6))
    assert pose_seed_index('inside_13')==6
    for exit_s in (112.,113.,113.+5e-7):
        assert 'inside_13' not in dict(generator.pose_stations(100.,exit_s))
    stations=generator.pose_stations(100.,114.)
    assert [p for p,_ in stations]==list(V4_POSES)
    assert dict(stations)['inside_13']==113.
    assert dict(stations)['exit_5']==119.


def test_new_pose_fixed_guide_and_existing_rows_unchanged(monkeypatch):
    q={'medium':generator.DEFAULT_QUALITIES['medium']}
    world,rows,routes=generator.build_one(30000000,IntersectionConfig(),q)
    generator.verify(rows)
    new=[r for r in rows if r['metadata']['pose_name']=='inside_13']
    expected=sum(rt['entry_s']+13<rt['exit_s']-1e-6 for rt in routes.values())
    assert len(new)==expected and expected>0
    for r in new:
        rt=routes[r['navigation']];md=r['metadata']
        assert md['ego_route_s_m']==rt['entry_s']+13
        assert md['ego_route_s_m']<rt['exit_s']
        np.testing.assert_allclose(md['ego_world_xy'],at_s(rt['gt_world'],[rt['entry_s']+13])[0],atol=1e-6)
        # New pose must also work with the mixed-FOV supplement pipeline.
        candidates,expanded=supplement_candidates(r,world,q['medium'],SupplementConfig())
        assert expanded['forward']>=100
    positions=generator.pose_stations
    monkeypatch.setattr(generator,'pose_stations',lambda entry,exit:[(p,s) for p,s in positions(entry,exit) if p!='inside_13'])
    _,legacy_rows,_=generator.build_one(30000000,IntersectionConfig(),q)
    assert legacy_rows==[r for r in rows if r['metadata']['pose_name']!='inside_13']
