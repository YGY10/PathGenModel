"""Build training-only, world-aligned FOV supplements without changing base JSONL.

python -m Scenario.build_fov_supplements --input outputs/v4_world300_fixedguide/audit/train.jsonl \
    --worlds-dir outputs/v4_world300_fixedguide/worlds --output-dir outputs/context_train300
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

from Scenario.fov_supplement import SupplementConfig, supplement_candidates
from Scenario.build_dataset_v4 import PerceptionQuality


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_source(path):
    worlds, ids = set(), set()
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            if not row.get('id') or not row.get('world_id') or row['id'] in ids:
                raise ValueError('Missing/duplicate sample identity')
            worlds.add(row['world_id']); ids.add(row['id'])
    if not ids:
        raise ValueError('Empty source')
    return {'sha256': sha256(path), 'samples': len(ids), 'worlds': sorted(worlds)}


def world_groups(path):
    group, current, seen = [], None, set()
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            if row['world_id'] != current:
                if group:
                    yield group
                current = row['world_id']
                if current in seen:
                    raise ValueError('Input must group each world contiguously')
                seen.add(current); group = []
            group.append(row)
    if group:
        yield group


def generate_world(task):
    rows, worlds_dir, qualities, ranges, seeds, max_lines = task
    world = json.loads((Path(worlds_dir) / f'{rows[0]["world_id"]}.json').read_text())
    q = {name: PerceptionQuality(**value) for name, value in qualities.items()}
    lines = {seed: [] for seed in seeds}
    kept_count = candidate_count = 0
    for seed in seeds:
        cfg = SupplementConfig(**ranges, seed=seed)
        for row in rows:
            candidates, fov = supplement_candidates(row, world, q[row['metadata']['quality']], cfg)
            original_n = min(max_lines, sum(len(row.get(key, [])) for key in ('lane_markings', 'road_boundaries', 'stop_lines', 'crosswalks')))
            kept = candidates[:max_lines-original_n]
            entry = {'id': row['id'], 'world_id': row['world_id'], 'fov': fov,
                     'original_tokens': original_n, 'candidate_count': len(candidates),
                     'added_objects': [{'kind': kind, **obj} for kind, obj in kept]}
            lines[seed].append(json.dumps(entry) + '\n')
            kept_count += len(kept); candidate_count += len(candidates)
    return lines, len(rows), kept_count, candidate_count


def build_cache(input_path, worlds_dir, output_dir, seeds=(1031, 1032), workers=4,
                forward_m=100., left_m=60., right_m=60., max_lines=32, max_points=32):
    input_path, worlds_dir, output_dir = Path(input_path), Path(worlds_dir), Path(output_dir)
    if workers < 1 or not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError('Invalid workers/seeds')
    if min(forward_m, left_m, right_m) <= 0 or max_lines < 1 or max_points < 4:
        raise ValueError('Invalid FOV/capacity')
    if output_dir.exists():
        raise ValueError('Use a new output directory')
    source = inspect_source(input_path)
    generation_config = json.loads((worlds_dir.parent / 'config.json').read_text())
    qualities = generation_config['qualities']
    ranges = {'forward_m': forward_m, 'left_m': left_m, 'right_m': right_m}
    output_dir.mkdir(parents=True)
    handles = {seed: (output_dir / f'seed_{seed}.jsonl.tmp').open('w') for seed in seeds}
    count = kept = candidates = worlds = 0
    tasks = ((group, str(worlds_dir), qualities, ranges, tuple(seeds), max_lines) for group in world_groups(input_path))
    def consume(result):
        nonlocal count, kept, candidates, worlds
        lines, n, retained, offered = result
        for seed, content in lines.items():
            handles[seed].writelines(content)
        count += n; kept += retained; candidates += offered; worlds += 1
        if worlds % 10 == 0:
            print(f'cached {worlds}/{len(source["worlds"])} worlds, {count}/{source["samples"]} samples x {len(seeds)} seeds', flush=True)
    try:
        if workers == 1:
            for task in tasks:
                consume(generate_world(task))
        else:
            # Bounded submission prevents retaining the entire parsed source in RAM.
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
                pending = deque()
                for task in tasks:
                    pending.append(pool.submit(generate_world, task))
                    if len(pending) >= 2 * workers:
                        consume(pending.popleft().result())
                while pending:
                    consume(pending.popleft().result())
    finally:
        for handle in handles.values():
            handle.close()
    if count != source['samples'] or worlds != len(source['worlds']):
        raise AssertionError('Incomplete cache')
    files = []
    for seed in seeds:
        path = output_dir / f'seed_{seed}.jsonl'
        path.with_suffix('.jsonl.tmp').replace(path)
        files.append({'seed': seed, 'path': path.name, 'sha256': sha256(path)})
    manifest = {'version': 1, 'source_path': str(input_path.resolve()), 'source': source,
                'max_lines': max_lines, 'max_points': max_points, 'ranges': ranges,
                'qualities': qualities, 'files': files,
                'mean_retained_additions': kept / (count * len(seeds)),
                'mean_candidates': candidates / (count * len(seeds)),
                'notes': ['Only worlds present in source are sampled. Original JSONL is unchanged.',
                          'Same quality parameters, only new-region FOV sampling; never GT-based selection.']}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Cache ready: {output_dir}; samples={count}; seeds={list(seeds)}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--worlds-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1031, 1032])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--forward-m', type=float, default=100.)
    parser.add_argument('--left-m', type=float, default=60.)
    parser.add_argument('--right-m', type=float, default=60.)
    args = parser.parse_args()
    build_cache(args.input, args.worlds_dir, args.output_dir, args.seeds, args.workers,
                args.forward_m, args.left_m, args.right_m)


if __name__ == '__main__':
    main()
