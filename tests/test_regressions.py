import pytest

from visualize_regressions import rank_regressions


def test_rank_by_deterioration_not_absolute_error():
    rows = [dict(index=0, ade_m=3., guide_ade_m=4., fde_m=5., guide_fde_m=3.),
            dict(index=1, ade_m=1., guide_ade_m=.2, fde_m=.5, guide_fde_m=1.),
            dict(index=2, ade_m=2., guide_ade_m=1.8, fde_m=1., guide_fde_m=1.),
            dict(index=3, ade_m=1., guide_ade_m=1., fde_m=1., guide_fde_m=1.)]
    assert [r['index'] for r in rank_regressions(rows)] == [1, 2]
    assert [r['index'] for r in rank_regressions(rows, 'fde')] == [0]
    assert rank_regressions(rows, tolerance_m=1.) == []
    with pytest.raises(ValueError):
        rank_regressions(rows, tolerance_m=-1.)
