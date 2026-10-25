import numpy as np

from analyze_corrections import classify_correction


def test_categories_and_longitudinal_bias():
    guide = np.column_stack([np.arange(81.), np.zeros(81)])
    gt = guide.copy()
    gt[1:, 1] = 1.
    for amount, expected in [(2., 'overshoot'), (.3, 'undershoot'), (-.3, 'wrong_direction')]:
        pred = guide.copy()
        pred[1:, 1] = amount
        result = classify_correction(guide, gt, pred)
        assert result[expected]
        assert result['error_dominant'] == expected
        assert sum(result[k] for k in ('overshoot', 'undershoot', 'wrong_direction', 'unnecessary')) == 1
    pred = guide.copy()
    pred[1:, 0] -= .2
    result = classify_correction(guide, guide, pred)
    assert result['longitudinal_dominant']
    assert result['error_dominant'] == 'longitudinal'
    pred[1:, 1] = .3
    assert classify_correction(guide, guide, pred)['unnecessary']


def test_minimum_extent_and_exact_match():
    guide = np.column_stack([np.arange(81.), np.zeros(81)])
    assert classify_correction(guide, guide, guide)['error_dominant'] == 'small_or_mixed_lateral_error'
    pred = guide.copy()
    pred[1:20, 1] = 1.
    assert not classify_correction(guide, guide, pred)['unnecessary']
    pred[20, 1] = 1.
    assert classify_correction(guide, guide, pred)['unnecessary']
