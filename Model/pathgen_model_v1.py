"""Guide-conditioned vector-path Transformer (V1).

Consumes Dataset.dataset.PathJSONLDataset batches, without world-truth metadata.
- PointNet/VectorNet-style masked local polyline encoding
- Transformer spatial interaction across polyline tokens
- Guide XY+tangent+station query encoding; nav embedding
- Non-autoregressive decoder and guide-residual path prediction

No training loop or loss is included in this module.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    point_dim: int = 6
    attr_dim: int = 10
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    num_nav: int = 3


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Return zero for fully masked sets, not -inf / NaN."""
    if x.shape[:-1] != mask.shape:
        raise ValueError(f"mask mismatch {tuple(mask.shape)} vs {tuple(x.shape)}")
    v = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
    out = v.max(dim=dim).values
    present = mask.any(dim=dim)
    return torch.where(present.unsqueeze(-1), out, torch.zeros_like(out))


class PolylineEncoder(nn.Module):
    """PointNet-style two-stage subgraph plus polyline semantic attributes.

    Explicit masks prevent invalid point or line tokens from contributing.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.point1 = _mlp(cfg.point_dim, d, d)
        self.point2 = _mlp(2*d, d, d)
        self.attr = _mlp(cfg.attr_dim, d // 2, d)
        self.geom = _mlp(5, d // 2, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, points: torch.Tensor, point_mask: torch.Tensor,
                attrs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 4 or point_mask.shape != points.shape[:3]:
            raise ValueError('Expected points [B,L,P,F], mask [B,L,P]')
        if attrs.shape != (*points.shape[:2], self.attr[0].in_features):
            raise ValueError('polyline_attr shape mismatch')
        valid_line = point_mask.any(dim=-1)
        x = self.point1(points)
        pooled = masked_max(x, point_mask, dim=2)
        global_x = pooled.unsqueeze(2).expand_as(x)
        x = self.point2(torch.cat([x, global_x], dim=-1))
        pooled = masked_max(x, point_mask, dim=2)

        # Geometry: centroid + (undirected) chord magnitudes + total observed length.
        m = point_mask.to(points.dtype)
        xy = points[..., :2]
        centroid = (xy * m.unsqueeze(-1)).sum(2) / m.sum(2).clamp_min(1).unsqueeze(-1)
        first_index = point_mask.float().argmax(2)
        last_index = (point_mask.long() * torch.arange(points.shape[2], device=points.device)).max(2).values
        first = xy.gather(2, first_index[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        last = xy.gather(2, last_index[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        chord = (last - first).abs()
        # point feature index 5 is incoming-segment length (meters)
        total_length = (points[..., 5] * m).sum(2).unsqueeze(-1)
        geometry = torch.cat([centroid, chord, total_length], dim=-1)
        tokens = self.norm(pooled + self.attr(attrs) + self.geom(geometry))
        tokens = tokens.masked_fill(~valid_line.unsqueeze(-1), 0.)
        return tokens, valid_line


def relative_polyline_geometry(points: torch.Tensor, point_mask: torch.Tensor,
                               closed: torch.Tensor, guide: torch.Tensor,
                               tangent: torch.Tensor) -> torch.Tensor:
    """[B,Q,L,5]: longitudinal/lateral offset (m), distance (m), cos2/sin2.

    Project each guide point onto observed segments, including a closed box's
    last-to-first edge. Doubled-angle direction is invariant to line reversal.
    Invalid lines return zeros; a single valid point is a point candidate.
    """
    xy = points[..., :2]
    mask = point_mask.bool()
    b, l, p = mask.shape
    idx = torch.arange(p, device=points.device).view(1, 1, p).expand(b, l, -1)
    first_idx = mask.long().argmax(-1)
    last_idx = idx.masked_fill(~mask, -1).max(-1).values
    first = xy.gather(2, first_idx[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    next_xy = torch.cat([xy[:, :, 1:], first.unsqueeze(2)], dim=2)
    next_valid = torch.cat([mask[:, :, 1:], torch.zeros_like(mask[:, :, :1])], dim=2)
    closing = closed.bool().unsqueeze(-1) & (idx == last_idx.unsqueeze(-1))
    end = torch.where(closing.unsqueeze(-1), first.unsqueeze(2), next_xy)
    segment_valid = mask & (next_valid | closing)
    # Degenerate point candidates also handle singleton observations and gaps.
    v = torch.where(segment_valid.unsqueeze(-1), end - xy, torch.zeros_like(xy))
    length2 = v.square().sum(-1)
    offset = guide[:, :, None, None, :] - xy[:, None, :, :, :]
    fraction = (offset * v[:, None]).sum(-1) / length2[:, None].clamp_min(1e-12)
    projection = xy[:, None] + fraction.clamp(0, 1).unsqueeze(-1) * v[:, None]
    delta = projection - guide[:, :, None, None, :]
    distance2 = delta.square().sum(-1).masked_fill(~mask[:, None], float('inf'))
    nearest = distance2.argmin(-1)
    delta = delta.gather(3, nearest[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(3)
    direction = F.normalize(v, dim=-1, eps=1e-6)[:, None].expand(-1, guide.shape[1], -1, -1, -1)
    direction = direction.gather(3, nearest[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(3)
    t = tangent[:, :, None, :]
    longitudinal = (delta * t).sum(-1)
    lateral = t[..., 0] * delta[..., 1] - t[..., 1] * delta[..., 0]
    cosine = (direction * t).sum(-1)
    sine = t[..., 0] * direction[..., 1] - t[..., 1] * direction[..., 0]
    features = torch.stack([longitudinal, lateral, delta.norm(dim=-1),
                            cosine.square() - sine.square(), 2 * cosine * sine], dim=-1)
    return features.masked_fill(~mask.any(-1)[:, None, :, None], 0.)


class PathGenModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = ModelConfig() if cfg is None else cfg
        if cfg.d_model % cfg.nhead:
            raise ValueError('d_model must be divisible by nhead')
        self.cfg = cfg
        d = cfg.d_model
        self.polyline_encoder = PolylineEncoder(cfg)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout, batch_first=True, activation='gelu', norm_first=True)
        self.spatial_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.encoder_layers,
                                                     enable_nested_tensor=False)
        self.empty_scene = nn.Parameter(torch.zeros(1, 1, d))
        self.guide_embed = _mlp(5, d, d)
        self.nav_embed = nn.Embedding(cfg.num_nav, d)
        self.query_norm = nn.LayerNorm(d)
        # Type-conditioned relative geometry -> additive logits for each head.
        # Zero initialization starts with the original content-only attention.
        self.relative_bias = _mlp(5 + cfg.attr_dim, d // 2, cfg.nhead)
        nn.init.zeros_(self.relative_bias[-1].weight)
        nn.init.zeros_(self.relative_bias[-1].bias)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout, batch_first=True, activation='gelu', norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.decoder_layers)
        self.residual_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                           nn.GELU(), nn.Linear(d, 2))
        # Initialization means model initially copies guide, except first point fixed at origin.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def guide_features(guide: torch.Tensor) -> torch.Tensor:
        if guide.ndim != 3 or guide.shape[-1] != 2 or guide.shape[1] < 2:
            raise ValueError('guide must have shape [B,S>=2,2]')
        # Centered finite differences, computed exclusively from the input guide.
        tangent = torch.empty_like(guide)
        tangent[:, 0] = guide[:, 1] - guide[:, 0]
        tangent[:, -1] = guide[:, -1] - guide[:, -2]
        tangent[:, 1:-1] = guide[:, 2:] - guide[:, :-2]
        tangent = F.normalize(tangent, dim=-1, eps=1e-6)
        seglen = (guide[:, 1:] - guide[:, :-1]).norm(dim=-1)
        station = torch.cat([torch.zeros_like(seglen[:, :1]), seglen.cumsum(dim=1)], dim=1)
        station = station / station[:, -1:].clamp_min(1e-6)
        return torch.cat([guide, tangent, station.unsqueeze(-1)], dim=-1)

    def forward(self, bev_points: torch.Tensor, point_mask: torch.Tensor,
                polyline_mask: torch.Tensor, polyline_attr: torch.Tensor,
                guide: torch.Tensor, navigation: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(bev_points).all() or not torch.isfinite(guide).all():
            raise ValueError('Non-finite model input')
        if polyline_mask.shape != point_mask.shape[:2]:
            raise ValueError('polyline_mask must be [B,L]')
        if not torch.equal(polyline_mask.bool(), point_mask.any(-1)):
            raise ValueError('polyline_mask must equal point_mask.any(-1)')
        if navigation.ndim != 1 or navigation.shape[0] != guide.shape[0]:
            raise ValueError('navigation must be [B]')
        tokens, valid = self.polyline_encoder(bev_points, point_mask.bool(), polyline_attr)
        # Add a never-masked learned sentinel. This makes all-empty BEV batches safe.
        sentinel = self.empty_scene.expand(tokens.shape[0], -1, -1)
        memory = torch.cat([tokens, sentinel], dim=1)
        key_mask = torch.cat([~valid, torch.zeros(valid.shape[0], 1,
                          device=valid.device, dtype=torch.bool)], dim=1)
        memory = self.spatial_encoder(memory, src_key_padding_mask=key_mask)
        guide_features = self.guide_features(guide)
        query = self.query_norm(self.guide_embed(guide_features) +
                                self.nav_embed(navigation.long()).unsqueeze(1))
        geometry = relative_polyline_geometry(
            bev_points, point_mask, polyline_attr[..., 9] > .5,
            guide, guide_features[..., 2:4])
        attrs = polyline_attr[:, None].expand(-1, guide.shape[1], -1, -1)
        bias = self.relative_bias(torch.cat([geometry, attrs], dim=-1))
        # [B,Q,L,H] -> [B*H,Q,L+1]. Sentinel has no geometric bias.
        bias = bias.permute(0, 3, 1, 2)
        bias = F.pad(bias, (0, 1), value=0.)
        bias = bias.masked_fill(key_mask[:, None, None, :], float('-inf'))
        bias = bias.reshape(-1, guide.shape[1], memory.shape[1])
        decoded = self.decoder(query, memory, memory_mask=bias)
        residual = self.residual_head(decoded)
        out = guide + residual
        # No in-place mutation; first path station has exactly zero XY.
        return torch.cat([torch.zeros_like(out[:, :1]), out[:, 1:]], dim=1)

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self(batch['bev_points'], batch['point_mask'], batch['polyline_mask'],
                    batch['polyline_attr'], batch['guide'], batch['navigation'])


if __name__ == '__main__':
    torch.manual_seed(7)
    b = 2
    model = PathGenModel()
    points = torch.randn(b, 32, 32, 6)
    pm = torch.zeros(b,32,32,dtype=torch.bool)
    pm[0, 0, :12] = True  # second item has completely empty BEV
    attr = torch.zeros(b,32,10)
    guide = torch.stack([torch.linspace(0,80,81), torch.zeros(81)],dim=-1).repeat(b,1,1)
    pred = model(points, pm, pm.any(-1), attr, guide, torch.tensor([0,2]))
    assert pred.shape == (b,81,2) and (pred[:,0] == 0).all() and torch.isfinite(pred).all()
    loss = (pred[:,1:]**2).mean()
    loss.backward()
    print('smoke test OK:', tuple(pred.shape), 'finite gradients:', all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
