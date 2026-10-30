from dataclasses import replace

import numpy as np

from Scenario.build_dataset_v4 import DEFAULT_QUALITIES, observe, visible_line_sections


def test_break_is_inside_visible_run_and_independent_of_hidden_length():
    x=np.arange(-200.,201.,.2)
    points=np.column_stack([x,np.zeros_like(x)])
    mask=(x>=0)&(x<=40)
    visible=points[mask]
    a=visible_line_sections(points,mask,1.,np.random.default_rng(7))
    b=visible_line_sections(visible,np.ones(len(visible),dtype=bool),1.,np.random.default_rng(7))
    assert len(a)==len(b)==2
    for p,q in zip(a,b):np.testing.assert_array_equal(p,q)
    assert a[0][0,0] == visible[0,0] and a[-1][-1,0] == visible[-1,0]
    assert .8 <= a[1][0,0]-a[0][-1,0] <= 6
    unbroken=visible_line_sections(points,mask,0.,np.random.default_rng(7))
    assert len(unbroken)==1
    np.testing.assert_array_equal(unbroken[0],visible)


def test_no_polyline_connects_across_invisible_regions():
    x=np.arange(0.,51.,.2)
    points=np.column_stack([x,np.zeros_like(x)])
    mask=(x<15)|(x>30)
    parts=visible_line_sections(points,mask,0.,np.random.default_rng(3))
    assert len(parts)==2
    assert parts[0][-1,0]<15 and parts[1][0,0]>30
    short=visible_line_sections(points,x<3,1.,np.random.default_rng(3))
    assert len(short)==1


def test_observe_breaks_in_rotated_vehicle_fov():
    q=replace(DEFAULT_QUALITIES['clean'],line_miss_rate=0.,curb_miss_rate=0.,break_rate=1.,
              sigma_m=0.,offset_sigma_m=0.)
    angle=.7
    t=np.array([np.cos(angle),np.sin(angle)])
    normal=np.array([-t[1],t[0]])
    ego=np.array([100.,200.])
    x=np.arange(-200.,201.,.2)
    p=ego+x[:,None]*t+3*normal
    world={'physical':{'elements':[{'id':'edge','type':'line','style':'solid','color':'white','points':p.tolist()}]}}
    lines,curbs,_,_=observe(world,ego,t,q,np.random.default_rng(9))
    assert len(lines)==2 and not curbs
    for obj in lines:
        xy=np.asarray(obj['points'])
        assert xy[:,0].min()>=-q.rear_m-.001 and xy[:,0].max()<=q.forward_m+.001
        np.testing.assert_allclose(xy[:,1],3.,atol=.001)
        assert obj['line_style']=='solid'
