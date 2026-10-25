import torch

from Model.pathgen_model_v1 import (
    ModelConfig, PathGenModel, relative_polyline_geometry,
)


def test_projection_direction_and_padding():
    points = torch.zeros(1, 2, 4, 6)
    points[0, 0, :2, :2] = torch.tensor([[0., 2.], [10., 2.]])
    mask = torch.zeros(1, 2, 4, dtype=torch.bool)
    mask[0, 0, :2] = True
    guide = torch.tensor([[[5., 0.], [6., 0.]]])
    tangent = torch.tensor([[[1., 0.], [1., 0.]]])
    closed = torch.zeros(1, 2, dtype=torch.bool)
    geom = relative_polyline_geometry(points, mask, closed, guide, tangent)
    torch.testing.assert_close(geom[0, :, 0], torch.tensor([[0., 2., 2., 1., 0.]]).expand(2, -1))
    assert (geom[:, :, 1] == 0).all()
    reversed_points = points.clone()
    reversed_points[0, 0, :2] = points[0, 0, :2].flip(0)
    torch.testing.assert_close(
        geom, relative_polyline_geometry(reversed_points, mask, closed, guide, tangent))


def test_closed_edge_and_singleton():
    points = torch.zeros(1, 2, 4, 6)
    points[0, 0, :, :2] = torch.tensor([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
    points[0, 1, 0, :2] = torch.tensor([-2., 1.])
    mask = torch.tensor([[[True] * 4, [True, False, False, False]]])
    guide = torch.tensor([[[-1., 1.]]])
    tangent = torch.tensor([[[1., 0.]]])
    geom = relative_polyline_geometry(points, mask, torch.tensor([[True, False]]), guide, tangent)
    torch.testing.assert_close(geom[0, 0, 0, :3], torch.tensor([1., 0., 1.]))
    torch.testing.assert_close(geom[0, 0, 1], torch.tensor([-1., 0., 1., 0., 0.]))


def test_geometry_attention_mask_and_gradients():
    torch.manual_seed(7)
    model = PathGenModel(ModelConfig(d_model=32, nhead=4, ff_dim=64, dropout=0.))
    points = torch.zeros(2, 3, 4, 6)
    points[0, 0, :2, :2] = torch.tensor([[0., 2.], [10., 2.]])
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    mask[0, 0, :2] = True  # second scene is entirely empty
    attrs = torch.zeros(2, 3, 10)
    guide = torch.stack([torch.arange(8.), torch.zeros(8)], -1).repeat(2, 1, 1)
    nav = torch.tensor([0, 2])
    args = (points, mask, mask.any(-1), attrs, guide, nav)
    torch.testing.assert_close(model(*args), guide)
    # Open the initially zero output head to exercise upstream gradient flow.
    torch.nn.init.normal_(model.residual_head[-1].weight, std=.02)
    torch.nn.init.normal_(model.relative_bias[-1].weight, std=.02)
    pred = model(*args)
    polluted = points.clone()
    polluted[~mask] = 10000.
    torch.testing.assert_close(pred, model(polluted, *args[1:]))
    assert (pred[:, 0] == 0).all() and torch.isfinite(pred).all()
    pred.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.relative_bias[-1].weight.grad.abs().sum() > 0
