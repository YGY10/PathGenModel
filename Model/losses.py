"""Composable path supervision. Coordinates and Smooth L1 beta are in metres.

Usage:
    criterion = PathLoss()
    result = criterion(pred, batch['gt_path'])
    result['total'].backward()

Terms retain gradients; detach them when logging. Evaluation metrics belong
outside this module. Second differences assume the current uniform 1 m sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PathLossConfig:
    position_weight: float = 1.0
    second_difference_weight: float = 0.1
    position_beta_m: float = 1.0
    second_difference_beta_m: float = 0.1
    exclude_start: bool = True

    def __post_init__(self):
        for name in ('position_weight', 'second_difference_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('position_beta_m', 'second_difference_beta_m'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


def masked_sample_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean per sample, then mean across samples with supervision.

    Zero-weight samples are excluded. An entirely empty batch yields a
    differentiable zero. Tiny positive weights retain their relative meaning.
    """
    denominator = weights.sum(dim=1)
    present = denominator > 0
    safe_denominator = torch.where(present, denominator, torch.ones_like(denominator))
    sample_mean = (values * weights).sum(dim=1) / safe_denominator
    return sample_mean.sum() / present.sum().clamp_min(1)


def position_loss(pred: torch.Tensor, target: torch.Tensor,
                  weights: torch.Tensor, beta: float) -> torch.Tensor:
    per_point = F.smooth_l1_loss(pred, target, beta=beta, reduction='none').mean(-1)
    return masked_sample_mean(per_point, weights)


def second_difference_loss(pred: torch.Tensor, target: torch.Tensor,
                           valid_mask: torch.Tensor, beta: float) -> torch.Tensor:
    """Match GT bending rather than penalizing every turn toward a straight line.

    Validity is based on all three points. Position-specific weights deliberately
    do not affect this term. The fixed start remains a valid geometric anchor.
    """
    pred_d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    target_d2 = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    valid = valid_mask[:, 2:] & valid_mask[:, 1:-1] & valid_mask[:, :-2]
    values = F.smooth_l1_loss(pred_d2, target_d2, beta=beta, reduction='none').mean(-1)
    return masked_sample_mean(values, valid.to(values.dtype))


class PathLoss(nn.Module):
    """Position + second-difference matching, with explicit reduction semantics.

    pred_path / gt_path: [B,T,2], metres, same floating dtype and device.
    valid_mask: optional bool [B,T]. Invalid GT coordinates may be non-finite;
        they are replaced before arithmetic. Predictions must always be finite.
    point_weight: optional finite nonnegative [B,T], for POSITION ONLY.
        Zero weight does not invalidate a point for second-difference matching;
        use valid_mask when a point has no trustworthy target.
    """

    def __init__(self, cfg: PathLossConfig | None = None):
        super().__init__()
        self.cfg = PathLossConfig() if cfg is None else cfg

    def forward(self, pred_path: torch.Tensor, gt_path: torch.Tensor,
                valid_mask: torch.Tensor | None = None,
                point_weight: torch.Tensor | None = None) -> dict:
        if pred_path.ndim != 3 or pred_path.shape[-1] != 2 or pred_path.shape[1] < 1:
            raise ValueError('pred_path must be [B,T>=1,2]')
        if gt_path.shape != pred_path.shape or pred_path.shape[0] < 1:
            raise ValueError('GT and prediction must have the same nonempty batch shape')
        if (not pred_path.is_floating_point() or gt_path.dtype != pred_path.dtype
                or gt_path.device != pred_path.device):
            raise ValueError('GT and prediction must share floating dtype and device')
        if not torch.isfinite(pred_path).all():
            raise ValueError('Non-finite prediction')
        shape = pred_path.shape[:2]
        if valid_mask is None:
            valid_mask = torch.ones(shape, dtype=torch.bool, device=pred_path.device)
        elif (valid_mask.shape != shape or valid_mask.dtype != torch.bool
              or valid_mask.device != pred_path.device):
            raise ValueError('valid_mask must be bool [B,T] on the prediction device')
        if not torch.isfinite(gt_path[valid_mask]).all():
            raise ValueError('Non-finite valid GT')
        if point_weight is None:
            point_weight = torch.ones(shape, device=pred_path.device, dtype=pred_path.dtype)
        elif (point_weight.shape != shape or point_weight.device != pred_path.device
              or not torch.isfinite(point_weight).all() or (point_weight < 0).any()):
            raise ValueError('point_weight must be finite nonnegative [B,T] on the prediction device')

        # Compute reductions in float32 for half/bfloat16 training; preserve float64.
        dtype = torch.float64 if pred_path.dtype == torch.float64 else torch.float32
        pred = pred_path.to(dtype)
        target = torch.where(valid_mask.unsqueeze(-1), gt_path, torch.zeros_like(gt_path)).to(dtype)
        # Mask predictions too, so invalid coordinates cannot leak through differences.
        pred = torch.where(valid_mask.unsqueeze(-1), pred, torch.zeros_like(pred))
        position_valid = valid_mask.clone()
        if self.cfg.exclude_start:
            position_valid[:, 0] = False
        weights = point_weight.to(dtype) * position_valid.to(dtype)
        terms = {
            'position': position_loss(pred, target, weights, self.cfg.position_beta_m),
            'second_difference': second_difference_loss(
                pred, target, valid_mask, self.cfg.second_difference_beta_m),
        }
        weighted = {
            'position': self.cfg.position_weight * terms['position'],
            'second_difference': self.cfg.second_difference_weight * terms['second_difference'],
        }
        return {'total': sum(weighted.values()), 'terms': terms, 'weighted_terms': weighted}
