import numpy as np
import pytest
from scripts.diagnose_lateral_output import decompose


def test_straight_exact_pointwise_floor_and_free_origin():
    g=np.stack([np.arange(81.),np.full(81,1.)],axis=-1)[None]
    gt=g.copy();gt[:,:,0]+=.4;gt[:,:,1]-=2.;gt[:,0]=0
    r=decompose(g,gt)
    np.testing.assert_allclose(r['ds'][:,1:],.4,atol=4e-6)
    np.testing.assert_allclose(r['dl'][:,1:],-2.,atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(r['ideal_l_only'][:,1:]-r['gt'][:,1:],axis=-1),abs(r['ds'][:,1:]),atol=1e-10)
    np.testing.assert_array_equal(r['ideal_l_only'][:,0],[[0.,0.]])


def test_lateral_only_gt_has_zero_floor_even_with_nonzero_guide_start():
    g=np.stack([np.arange(81.),np.full(81,1.5)],axis=-1)[None]
    y=g.copy();y[:,:,1]-=.7;y[:,0]=0
    r=decompose(g,y)
    np.testing.assert_allclose(r['ds'][:,1:],0.,atol=1e-10)
    np.testing.assert_allclose(r['ideal_l_only'],r['gt'],atol=1e-7)


def test_rotation_invariance_and_curved_two_component_reconstruction():
    s=np.linspace(0,1.2,81);g=np.stack([10*np.sin(s),10*(1-np.cos(s))],axis=-1)[None]
    y=g+np.array([.8,-.3]);y[:,0]=0
    first=decompose(g,y)
    angle=.8;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    second=decompose(g@rot.T,y@rot.T)
    np.testing.assert_allclose(first['ds'],second['ds'],atol=5e-6)
    np.testing.assert_allclose(first['dl'],second['dl'],atol=5e-6)


def test_degenerate_normal_is_rejected():
    g=np.zeros((1,81,2))
    with pytest.raises(ValueError,match='Degenerate'):decompose(g,g)


def test_reparameterization_removes_phase_shift_but_requires_horizon_coverage():
    from scripts.diagnose_lateral_output import resampling_diagnostic
    gt=np.stack([np.arange(81.),np.zeros(81)],axis=-1)[None]
    g=gt.copy();g[:,:,0]+=.5;g[:,:,1]=1.
    result=decompose(g,gt)
    assert abs(result['ds'][:,1:]).mean()==pytest.approx(.5)
    diagnostic=resampling_diagnostic(result['ideal_l_only'],result['gt'])
    assert diagnostic['fraction']==1
    assert diagnostic['resampled_ade_in_covered_subset_m']==pytest.approx(0,abs=1e-8)
    g[:,:,0]-=1.
    result=decompose(g,gt)
    diagnostic=resampling_diagnostic(result['ideal_l_only'],result['gt'])
    assert diagnostic['fraction']==0
    assert diagnostic['insufficient_length_mean_deficit_m']==pytest.approx(.5)
