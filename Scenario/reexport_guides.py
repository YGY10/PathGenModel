"""Repair V4 guide anchoring from saved world guides without resampling truth.

python -m Scenario.reexport_guides --source outputs/v4_world300 \
    --output-dir outputs/v4_world300_fixedguide

Creates a new directory. Run Dataset.audit / audit_extended for fresh statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from Scenario.build_dataset import at_s, plot, V4_POSES
from Scenario.build_dataset_v4 import GUIDE_CONVENTION, sample_world_guide, verify


def replace_guide(row, route):
    md = row['metadata']
    ego = np.asarray(md['ego_world_xy'], dtype=float)
    heading = np.array([np.cos(md['heading_rad']), np.sin(md['heading_rad'])])
    ref, station = sample_world_guide(route['guide_world'], ego, heading)
    reconstructed = ref[:, :1] * heading + ref[:, 1:] * np.array([-heading[1], heading[0]]) + ego
    expected = at_s(np.asarray(route['guide_world']), station + np.arange(len(ref)))
    error = float(np.linalg.norm(reconstructed - expected, axis=1).max())
    if error > 1e-8:
        raise AssertionError('Guide world-frame closure failed')
    return {**row, 'reference_line': np.round(ref, 4).tolist(),
            'metadata': {**md, 'guide_coordinate_convention': GUIDE_CONVENTION,
                         'guide_start_s_m': station}}, error


def non_guide_payload(row):
    """Canonical payload for checking unchanged GT/perception/identity fields."""
    md = {k: v for k, v in row['metadata'].items()
          if k not in ('guide_coordinate_convention', 'guide_start_s_m')}
    data = {**row, 'metadata': md}
    data.pop('reference_line')
    return json.dumps(data, sort_keys=True, separators=(',', ':')).encode()


def reexport(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    for name in ('worlds', 'routes'):
        shutil.copytree(source / name, output / name)
    for name in ('world.json', 'accepted_world_seeds.json', 'rejected_worlds.json'):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    config = json.loads((source / 'config.json').read_text())
    config.update(guide_coordinate_convention=GUIDE_CONVENTION, reexport_source=str(source.resolve()))
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    current_world, routes, group, first_group = None, None, [], None
    original_digest, fixed_digest = hashlib.sha256(), hashlib.sha256()
    counts, original_ade, fixed_ade = Counter(), Counter(), Counter()
    maximum_closure = 0.
    ids, seen_worlds = set(), set()
    with (source / 'samples.jsonl').open() as src, (output / 'samples.jsonl').open('w') as dst:
        for line in src:
            row = json.loads(line)
            if row['id'] in ids:
                raise ValueError('Duplicate sample ID')
            ids.add(row['id'])
            if row['world_id'] != current_world:
                if group:
                    verify(group)
                    if first_group is None:
                        first_group = group
                group = []
                current_world = row['world_id']
                seen_worlds.add(current_world)
                routes = json.loads((source / 'routes' / f'{current_world}.json').read_text())
            fixed, closure = replace_guide(row, routes[row['navigation']])
            maximum_closure = max(maximum_closure, closure)
            original_digest.update(non_guide_payload(row))
            fixed_digest.update(non_guide_payload(fixed))
            gt = np.asarray(row['gt_path'])
            key = row['metadata']['pose_name']
            counts[key] += 1
            original_ade[key] += float(np.linalg.norm(np.asarray(row['reference_line'])[1:] - gt[1:], axis=1).mean())
            fixed_ade[key] += float(np.linalg.norm(np.asarray(fixed['reference_line'])[1:] - gt[1:], axis=1).mean())
            dst.write(json.dumps(fixed) + '\n')
            group.append(fixed)
    if group:
        verify(group)
    if first_group is None:
        first_group = group
    if not ids or original_digest.hexdigest() != fixed_digest.hexdigest():
        raise AssertionError('Empty data or non-guide fields changed')
    world = json.loads((source / 'worlds' / f'{first_group[0]["world_id"]}.json').read_text())
    poses=V4_POSES if any(r['metadata']['pose_name']=='inside_13' for r in first_group) else None
    plot(world, first_group, output / 'preview.png',pose_order=poses)
    report = {'source': str(source.resolve()), 'guide_coordinate_convention': GUIDE_CONVENTION,
              'samples': len(ids), 'worlds': len(seen_worlds),
              'non_guide_fields_identical': True, 'non_guide_sha256': original_digest.hexdigest(),
              'guide_world_closure_max_m_before_rounding': maximum_closure,
              'guide_ade_excluding_origin_by_pose': {
                  key: {'samples': counts[key], 'before_m': original_ade[key] / counts[key],
                        'after_m': fixed_ade[key] / counts[key]} for key in counts}}
    (output / 'guide_reexport_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    reexport(args.source, args.output_dir)


if __name__ == '__main__':
    main()
