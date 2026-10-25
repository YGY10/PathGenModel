import pytest
import torch

from Model.losses import PathLoss, PathLossConfig


def test_exact_curved_path_has_zero_loss():
    x = torch.arange(5.)
    gt = torch.stack([x, x.square()], -1).unsqueeze(0)
    result = PathLoss()(gt.clone().requires_grad_(), gt)
    assert result['total'].item() == 0
    result['total'].backward()


def test_weighted_position_and_equal_sample_reduction():
    pred = torch.zeros(2, 4, 2, requires_grad=True)
    gt = torch.zeros_like(pred)
    gt[0, 1:, 0] = 2.  # Smooth L1: 1.5, mean XY: .75
    gt[1, 1, 0] = 1.  # Smooth L1: .5, mean XY: .25
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    result = PathLoss(PathLossConfig(second_difference_weight=0.))(pred, gt, mask)
    torch.testing.assert_close(result['total'], torch.tensor(.5))
    weights = torch.tensor([[0., 0., 0., 0.], [0., 1e-9, 0., 0.]])
    result = PathLoss()(pred, gt, mask, weights)
    torch.testing.assert_close(result['terms']['position'], torch.tensor(.25))


def test_empty_supervision_nonfinite_padding_and_short_path():
    for length in (1, 2, 5):
        pred = torch.ones(2, length, 2, requires_grad=True)
        gt = torch.full_like(pred, float('nan'))
        mask = torch.zeros(2, length, dtype=torch.bool)
        result = PathLoss()(pred, gt, mask)
        assert result['total'].item() == 0
        result['total'].backward()
        assert (pred.grad == 0).all()


def test_difference_validity_and_independence_from_position_weights():
    pred = torch.zeros(1, 5, 2, requires_grad=True)
    gt = torch.zeros_like(pred)
    gt[0, 2, 1] = 1.
    loss = PathLoss()
    result = loss(pred, gt, point_weight=torch.zeros(1, 5))
    assert result['terms']['position'] == 0
    assert result['terms']['second_difference'] > 0
    mask = torch.tensor([[True, True, False, True, True]])
    gt[0, 2] = float('nan')
    result = loss(pred, gt, mask)
    assert result['total'] == 0  # no valid triplet spans the missing point


def test_nonzero_loss_backward_and_input_validation():
    pred = torch.randn(2, 81, 2, requires_grad=True)
    gt = torch.zeros_like(pred)
    result = PathLoss()(pred, gt)
    result['total'].backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    assert (pred.grad[:, 0] != 0).any()  # start anchors second differences
    with pytest.raises(ValueError):
        PathLoss()(pred, gt, point_weight=-torch.ones(2, 81))
    with pytest.raises(ValueError):
        PathLoss()(pred, gt, valid_mask=torch.ones(2, 81))
    with pytest.raises(ValueError):
        PathLossConfig(position_beta_m=0.)
    with pytest.raises(ValueError):
        PathLossConfig(second_difference_weight=-1.)
