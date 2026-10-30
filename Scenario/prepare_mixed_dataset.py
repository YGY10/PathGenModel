"""Combine pre-split datasets without changing World membership or GT/Guide.

Optionally regenerate a source's perception using its saved World and ego pose.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from Scenario.build_dataset_v4 import PERCEPTION_CONVENTION, observe, read_qualities


SPLITS = ('train', 'val', 'test')


def source_arg(text):
    name, sep, folder = text.partition('=')
    if not sep or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name) or not folder:
        raise argparse.ArgumentTypeError('Expected a unique name=dataset_directory')
    return name, Path(folder)


def inspect_sources(sources):
    """Check all splits before creating output; never reassign a sample's split."""
    worlds, ids, manifests = {}, set(), {}
    for name, root in sources.items():
        manifests[name] = {}
        for split in SPLITS:
            path = root/'audit'/f'{split}.jsonl'
            digest = hashlib.sha256()
            count, split_worlds = 0, set()
            with path.open('rb') as stream:
                for line in stream:
                    digest.update(line)
                    row = json.loads(line)
                    wid, sid = row.get('world_id'), row.get('id')
                    if not isinstance(wid,str) or not wid or not isinstance(sid,str) or not sid:
                        raise ValueError(f'{path}: missing World/sample ID')
                    if sid in ids:
                        raise ValueError(f'Duplicate sample ID: {sid}')
                    ids.add(sid)
                    owner = (name, split)
                    if wid in worlds and worlds[wid] != owner:
                        raise ValueError(f'World overlap: {wid}: {worlds[wid]} vs {owner}')
                    worlds[wid] = owner
                    split_worlds.add(wid)
                    existing_type = row.get('metadata', {}).get('scene_type')
                    if existing_type is not None and existing_type != name:
                        raise ValueError(f'{sid}: scene_type={existing_type} conflicts with source name {name}')
                    for key in ('gt_path', 'reference_line'):
                        xy = np.asarray(row[key], dtype=float)
                        if xy.shape != (81,2) or not np.isfinite(xy).all():
                            raise ValueError(f'{sid}: invalid {key}')
                    count += 1
            if not count:
                raise ValueError(f'Empty source split: {path}; use a larger generated batch')
            manifests[name][split] = {'path':str(path.resolve()), 'sha256':digest.hexdigest(),
                                     'samples':count, 'worlds':sorted(split_worlds)}
    return manifests


def refresh_observation(row, world, qualities, seed):
    """Preserve labels/pose exactly; no new World, routes or split sampling."""
    md = row['metadata']
    q = qualities[md['quality']]
    identity = hashlib.sha256(row['id'].encode()).digest()
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, *[int.from_bytes(identity[i:i+4], 'little') for i in range(0,16,4)]]))
    h = md['heading_rad']
    lines, curbs, stops, crosswalks = observe(
        world, np.asarray(md['ego_world_xy']), np.array([np.cos(h),np.sin(h)]), q, rng)
    return {**row, 'lane_markings':lines, 'road_boundaries':curbs,
            'stop_lines':stops, 'crosswalks':crosswalks,
            # Existing support diagnostics describe old inputs and must be removed.
            **({'evaluation':{}} if 'evaluation' in row else {}),
            'metadata':{**md, 'perception_convention':PERCEPTION_CONVENTION,
                        'perception_resampled_seed':seed,
                        'vcs_fov_m':{'forward':q.forward_m,'rear':q.rear_m,
                                     'left':q.left_m,'right':q.right_m}}}


def prepare(sources, destination, refresh=(), perception_config=None, seed=7301):
    refresh = set(refresh)
    if not refresh <= sources.keys():
        raise ValueError(f'Unknown refresh sources: {sorted(refresh-sources.keys())}')
    manifests = inspect_sources(sources)
    qualities = read_qualities(perception_config)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'audit').mkdir()
    subsets = destination/'subsets'
    for name in sources:
        (subsets/name).mkdir(parents=True)
    counts, output_hashes, diagnostics = {}, {}, {}
    for split in SPLITS:
        counts[split] = {}
        output_hashes[split] = {}
        combined_hash = hashlib.sha256()
        with (destination/'audit'/f'{split}.jsonl').open('w') as combined:
            for name, root in sources.items():
                count, subset_hash = 0, hashlib.sha256()
                conventions, navs, poses = Counter(), Counter(), Counter()
                last_world_id, world = None, None
                with (root/'audit'/f'{split}.jsonl').open() as incoming, (subsets/name/f'{split}.jsonl').open('w') as subset:
                    for line in incoming:
                        row = json.loads(line)
                        if name in refresh:
                            if row['world_id'] != last_world_id:
                                wid = row['world_id']
                                if Path(wid).name != wid or wid in ('.','..'):
                                    raise ValueError('Unsafe world_id')
                                paths = (root/'worlds'/f'{wid}.json', root/'worlds'/wid/'world.json')
                                path = next((p for p in paths if p.is_file()), None)
                                if path is None:
                                    raise FileNotFoundError(f'Missing physical World for {wid}')
                                world = json.loads(path.read_text())
                                last_world_id = wid
                            row = refresh_observation(row, world, qualities, seed)
                        row['metadata'] = {**row.get('metadata', {}), 'scene_type':name,
                                           'source_dataset':str(root.resolve())}
                        text = json.dumps(row, allow_nan=False)+'\n'
                        subset.write(text)
                        combined.write(text)
                        subset_hash.update(text.encode())
                        combined_hash.update(text.encode())
                        md = row['metadata']
                        conventions[md.get('perception_convention', 'legacy_unspecified')] += 1
                        navs[row['navigation']] += 1
                        poses[md.get('pose_name', 'unknown')] += 1
                        count += 1
                        if count % 1000 == 0:
                            print(f'{split}/{name}: {count} samples', flush=True)
                counts[split][name] = count
                output_hashes[split][name] = subset_hash.hexdigest()
                diagnostics[f'{split}/{name}'] = {'perception_conventions':dict(conventions),
                                                 'navigation':dict(navs),'poses':dict(poses)}
                print(f'{split}/{name}: saved {count} samples', flush=True)
        output_hashes[split]['combined'] = combined_hash.hexdigest()
    report = {'sources':manifests, 'counts':counts, 'output_sha256':output_hashes,
              'diagnostics':diagnostics, 'refresh_sources':sorted(refresh),'refresh_seed':seed,
              'perception_config':str(Path(perception_config).resolve()) if perception_config else None,
              'refresh_quality_parameters':{name:asdict(q) for name,q in qualities.items()},
              'notes':['Source World split membership preserved; no oversampling.',
                       'Subsets are exact rows of combined data; use for separate evaluation.',
                       'Worlds/routes remain in source datasets; keep those directories.',
                       'No expanded-FOV augmentation. Refresh removes stale evaluation support annotations.']}
    (destination/'mix_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Mixed dataset ready: {destination}',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=source_arg,action='append',required=True,
                        help='Repeat: intersection=... lane_transition=...')
    parser.add_argument('--refresh-perception',action='append',default=[],help='Source name to reobserve')
    parser.add_argument('--perception-config',type=Path)
    parser.add_argument('--seed',type=int,default=7301)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    if len(dict(args.source)) != len(args.source):
        parser.error('Source names must be unique')
    prepare(dict(args.source),args.output_dir,args.refresh_perception,args.perception_config,args.seed)


if __name__ == '__main__':
    main()
