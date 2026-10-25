import numpy as np

from experiment_exit_boundaries import ideal_boundaries


def test_exit_segments_preserve_physical_order_and_transform():
    # South arm has two outgoing lanes; incoming edge is deliberately excluded.
    world = {'metadata':{'num_out':[2,1,1,1]}, 'physical':{'elements':[
        {'id':'south_divider_+0','type':'line','points':[[100,0],[0,0]],'color':'yellow','style':'solid'},
        {'id':'south_divider_-1','type':'line','points':[[100,3.5],[0,3.5]],'color':'white','style':'dashed'},
        {'id':'south_curb_outgoing','type':'curb','points':[[100,7],[0,7]]},
        {'id':'south_curb_incoming','type':'curb','points':[[100,-7],[0,-7]]}]}}
    route = {'exit_arm':0,'exit_s':10.,'gt_world':[[0,1.75],[200,1.75]]}
    row = {'metadata':{'ego_world_xy':[5,0],'heading_rad':np.pi/2}}
    result = ideal_boundaries(row,world,route,40.)
    assert [o['source_id'] for _,o in result] == ['south_divider_+0','south_divider_-1','south_curb_outgoing']
    assert [k for k,_ in result] == ['lane_marking','lane_marking','road_boundary']
    # World endpoints x=50 then 13 become ego y=-45 then -8.
    np.testing.assert_allclose(result[0][1]['points'],[[0,-45],[0,-8]],atol=1e-10)
    np.testing.assert_allclose(result[1][1]['points'],[[3.5,-45],[3.5,-8]],atol=1e-10)
    assert result[1][1]['line_style']=='dashed'
