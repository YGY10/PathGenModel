"""V4 perceived-vector -> model tensors. No truth-world information enters features.

Point features: (x,y, dx,dy, station_m, segment_length_m).
Attributes: one-hot [lane_marking, road_boundary, stop_line, crosswalk],
            confidence, yellow, white, solid, dashed, is_closed. (10 floats)
For a closed rectangle, 32 sampled points run around its *closed perimeter*.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

KINDS = ("lane_marking", "road_boundary", "stop_line", "crosswalk")
MAX_LINES = 32
MAX_POINTS = 32
NUM_POINT_FEATURES = 6
NUM_ATTRIBUTES = 10
NAV = {"straight": 0, "left": 1, "right": 2}


def _as_xy(points):
    a = np.asarray(points, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError('Polyline needs finite points shaped [N,2]')
    return a


def orient_near_to_far(points):
    """Orient open curve by endpoint distance; not monotonic-distance sorting."""
    a = _as_xy(points)
    if len(a) > 1 and np.linalg.norm(a[0]) > np.linalg.norm(a[-1]):
        return a[::-1].copy()
    return a


def resample(points, max_points=MAX_POINTS, closed=False):
    a = _as_xy(points)
    if closed:
        if len(a) < 3:
            raise ValueError('Closed polygon requires 3+ vertices')
        # canonicalize ring start by nearest vertex, preserve cyclic adjacency.
        j = int(np.argmin(np.linalg.norm(a, axis=1)))
        a = np.roll(a, -j, axis=0)
        a = np.vstack([a, a[0]])
    else:
        a = orient_near_to_far(a)
    dist = np.linalg.norm(np.diff(a, axis=0), axis=1)
    valid = np.concatenate([[True], dist > 1e-9])
    a = a[valid]
    if len(a) == 0:
        raise ValueError('Empty geometry')
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))]
    if arc[-1] < 1e-9:
        return a[:1].astype(np.float32)
    # Always use fixed-length arclength sampling for non-degenerate lines;
    # closed ring uses endpoint=False to avoid duplicating start and expose final edge.
    stations = np.linspace(0.0, arc[-1], max_points, endpoint=not closed)
    out = np.stack([np.interp(stations, arc, a[:,k]) for k in range(2)], axis=-1)
    return out.astype(np.float32)


def point_features(points, closed=False):
    a = _as_xy(points).astype(np.float32)
    n = len(a)
    delta = np.zeros_like(a)
    if n > 1:
        delta[1:] = a[1:] - a[:-1]
    seg = np.linalg.norm(delta, axis=1).astype(np.float32)
    station = np.cumsum(seg, dtype=np.float32)
    return np.column_stack([a, delta, station, seg]).astype(np.float32)


def attributes(obj, kind):
    v = np.zeros(NUM_ATTRIBUTES, dtype=np.float32)
    v[KINDS.index(kind)] = 1
    v[4] = float(np.clip(obj.get('confidence', 1.0), 0, 1))
    v[5] = float(obj.get('color') == 'yellow')
    v[6] = float(obj.get('color') == 'white')
    v[7] = float(obj.get('line_style') == 'solid')
    v[8] = float(obj.get('line_style') == 'dashed')
    v[9] = float(kind in ('stop_line', 'crosswalk'))
    return v


def _objects(row, max_lines):
    ls = [(k, obj) for key,k in [('lane_markings','lane_marking'),('road_boundaries','road_boundary')]
          for obj in row.get(key, [])]
    boxes = [(k, obj) for key,k in [('stop_lines','stop_line'),('crosswalks','crosswalk')]
             for obj in row.get(key, [])]
    def near(item):
        a = _as_xy(item[1]['points'])
        return float(np.min(np.linalg.norm(a, axis=1)))
    # Reserve capacity for all visible box objects (up to max_lines); prioritize nearer
    # visible boxes when capacity is exhausted. Then keep nearest open curves.
    boxes.sort(key=lambda x: (near(x), x[0], x[1].get('id','')))
    ls.sort(key=lambda x: (near(x), x[0], x[1].get('source_id','')))
    return (boxes[:max_lines] + ls[:max(0, max_lines - len(boxes))])


def encode_sample(row, max_lines=MAX_LINES, max_points=MAX_POINTS):
    if row.get('navigation') not in NAV:
        raise ValueError(f"Unexpected navigation {row.get('navigation')}")
    if max_lines < 1 or max_points < 4:
        raise ValueError('max_lines>=1 and max_points>=4 required')
    points = torch.zeros((max_lines, max_points, NUM_POINT_FEATURES), dtype=torch.float32)
    point_mask = torch.zeros((max_lines, max_points), dtype=torch.bool)
    attrs = torch.zeros((max_lines, NUM_ATTRIBUTES), dtype=torch.float32)
    for i, (kind,obj) in enumerate(_objects(row, max_lines)):
        closed = kind in ('stop_line', 'crosswalk')
        sampled = resample(obj['points'], max_points, closed=closed)
        n = len(sampled)
        points[i, :n] = torch.from_numpy(point_features(sampled, closed=closed))
        point_mask[i, :n] = True
        attrs[i] = torch.from_numpy(attributes(obj, kind))
    guide = torch.as_tensor(row['reference_line'], dtype=torch.float32)
    gt = torch.as_tensor(row['gt_path'], dtype=torch.float32)
    if guide.shape != (81,2) or gt.shape != (81,2):
        raise ValueError(f'Guide and GT must have shape [81,2]; got {guide.shape}, {gt.shape}')
    if not torch.isfinite(guide).all() or not torch.isfinite(gt).all():
        raise ValueError('Non-finite Guide/GT')
    return {
        'bev_points': points,
        'point_mask': point_mask,
        'polyline_mask': point_mask.any(-1),
        'polyline_attr': attrs,
        'guide': guide,
        'navigation': torch.tensor(NAV[row['navigation']], dtype=torch.long),
        'gt_path': gt,
    }


class PathJSONLDataset(Dataset):
    """Load world-disjoint train/val/test .jsonl files. Lazy decode, indexed offsets."""
    def __init__(self, path, max_lines=MAX_LINES, max_points=MAX_POINTS):
        self.path = Path(path)
        self.max_lines = max_lines
        self.max_points = max_points
        self.offsets = []
        with self.path.open('rb') as f:
            while True:
                offset = f.tell()
                if not f.readline():
                    break
                self.offsets.append(offset)
    def __len__(self):
        return len(self.offsets)
    def __getitem__(self, index):
        with self.path.open('rb') as f:
            f.seek(self.offsets[index])
            row = json.loads(f.readline())
        return encode_sample(row, self.max_lines, self.max_points)
