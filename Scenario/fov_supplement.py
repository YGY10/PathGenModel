"""Shared new-region observation sampling; no model, labels or token encoding."""
from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry import LineString, Polygon, box

from Scenario.build_dataset import NAVS, pose_seed_index
from Scenario.build_dataset_v4 import observe


@dataclass(frozen=True)
class SupplementConfig:
    forward_m: float = 100.
    left_m: float = 60.
    right_m: float = 60.
    seed: int = 1031


def fov_polygon(fov):
    return box(-fov['rear'], -fov['right'], fov['forward'], fov['left'])


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'LineString':
        return [geometry]
    return [part for child in getattr(geometry, 'geoms', []) for part in line_parts(child)]


def supplement_candidates(row, world, quality, cfg):
    md = row['metadata']
    old = md['vcs_fov_m']
    expanded = {'forward': max(old['forward'], cfg.forward_m), 'rear': old['rear'],
                'left': max(old['left'], cfg.left_m), 'right': max(old['right'], cfg.right_m)}
    q = replace(quality, forward_m=expanded['forward'], rear_m=expanded['rear'],
                left_m=expanded['left'], right_m=expanded['right'])
    seed = [cfg.seed, int(world['metadata']['seed']), NAVS.index(row['navigation']),
            pose_seed_index(md['pose_name']), *md['quality'].encode()]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    h = md['heading_rad']
    observed = observe(world, np.asarray(md['ego_world_xy']), np.array([np.cos(h), np.sin(h)]), q, rng, max_lines=None)
    old_poly, expanded_poly = fov_polygon(old), fov_polygon(expanded)
    candidates = []
    for kind, objects in zip(('lane_marking', 'road_boundary', 'stop_line', 'crosswalk'), observed):
        for obj in objects:
            xy = np.asarray(obj['points'])
            if kind in ('lane_marking', 'road_boundary'):
                geom = LineString(xy).intersection(expanded_poly).difference(old_poly)
                for segment in line_parts(geom):
                    if segment.length > 1.3:
                        candidates.append((kind, {**obj, 'points': np.asarray(segment.coords).tolist()}))
            else:
                poly = Polygon(xy)
                if poly.is_valid and poly.intersection(old_poly).area < 1e-9 and expanded_poly.covers(poly):
                    candidates.append((kind, obj))
    candidates.sort(key=lambda item: (float(np.min(np.linalg.norm(item[1]['points'], axis=1))),
                                      item[0], item[1].get('source_id', item[1].get('id', ''))))
    return candidates, expanded
