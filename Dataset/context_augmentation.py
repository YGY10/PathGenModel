"""Epoch-deterministic context augmentation, frozen base observations and labels."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from Dataset.dataset import PathJSONLDataset, encode_sample, resample, point_features, attributes


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ContextAugmentationConfig:
    cache_dir: str
    budgets: tuple = (0, 4, 8, 12, -1)  # -1 = all retained supplements
    probabilities: tuple = (.5, .125, .125, .125, .125)
    seed: int = 314159

    def __post_init__(self):
        if (not self.cache_dir or len(self.budgets) != len(self.probabilities)
                or not self.budgets or len(set(self.budgets)) != len(self.budgets)
                or any(not isinstance(k, int) or k < -1 for k in self.budgets)):
            raise ValueError('Invalid augmentation budgets/cache')
        if (any(not np.isfinite(p) or p < 0 for p in self.probabilities)
                or not np.isclose(sum(self.probabilities), 1., atol=1e-8, rtol=0)):
            raise ValueError('Augmentation probabilities must be nonnegative and sum to 1')


def inspect_context_cache(cache_dir, base_info, max_lines, max_points):
    directory = Path(cache_dir)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('version') != 1 or manifest['source']['sha256'] != base_info['sha256']:
        raise ValueError('Supplement cache does not match base dataset')
    if manifest['source']['worlds'] != base_info['worlds'] or manifest['source']['samples'] != base_info['samples']:
        raise ValueError('Cache world/sample membership mismatch')
    if (manifest['max_lines'], manifest['max_points']) != (max_lines, max_points):
        raise ValueError('Cache/model encoding capacity mismatch')
    if not manifest['files']:
        raise ValueError('Empty supplement cache')
    for entry in manifest['files']:
        if file_sha256(directory / entry['path']) != entry['sha256']:
            raise ValueError(f'Cache fingerprint mismatch: {entry["path"]}')
    return {'manifest_sha256': file_sha256(directory / 'manifest.json'),
            'files': {entry['path']: entry['sha256'] for entry in manifest['files']}}


def append_observations(base, candidates, max_points):
    result = {key: value.clone() for key, value in base.items()}
    free = (~base['polyline_mask']).nonzero().flatten().tolist()
    kept = candidates[:len(free)]
    for slot, (kind, obj) in zip(free, kept):
        sampled = resample(obj['points'], max_points, closed=kind in ('stop_line', 'crosswalk'))
        n = len(sampled)
        result['bev_points'][slot, :n] = torch.from_numpy(point_features(sampled))
        result['point_mask'][slot, :n] = True
        result['polyline_attr'][slot] = torch.from_numpy(attributes(obj, kind))
    result['polyline_mask'] = result['point_mask'].any(-1)
    valid = base['polyline_mask']
    for key in ('bev_points', 'point_mask', 'polyline_attr'):
        if not torch.equal(base[key][valid], result[key][valid]):
            raise AssertionError('Original observations changed')
    for key in ('guide', 'gt_path', 'navigation'):
        if not torch.equal(base[key], result[key]):
            raise AssertionError('Non-BEV fields changed')
    return result, kept


class ContextAugmentedDataset(PathJSONLDataset):
    """Only training uses this wrapper. No global RNG or worker-order dependence.

    Workers are recreated each epoch by the training loader (persistent_workers
    remains false) so set_epoch is propagated with num_workers>0.
    """
    def __init__(self, path, cfg, max_lines=32, max_points=32):
        super().__init__(path, max_lines, max_points)
        self.cfg = cfg
        self.epoch = 0
        directory = Path(cfg.cache_dir)
        self.manifest = json.loads((directory / 'manifest.json').read_text())
        self.cache_paths, self.cache_offsets = [], []
        with self.path.open() as f:
            expected_ids = [json.loads(line)['id'] for line in f]
        for entry in self.manifest['files']:
            cache = directory / entry['path']
            offsets, ids = [], []
            with cache.open('rb') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    offsets.append(offset); ids.append(json.loads(line)['id'])
            if ids != expected_ids:
                raise ValueError('Supplement cache sample order mismatch')
            self.cache_paths.append(cache); self.cache_offsets.append(offsets)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        with self.path.open('rb') as f:
            f.seek(self.offsets[index]); row = json.loads(f.readline())
        base = encode_sample(row, self.max_lines, self.max_points)
        digest = hashlib.sha256(f'{self.cfg.seed}|{self.epoch}|{row["id"]}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        profile = int(rng.choice(len(self.cfg.budgets), p=self.cfg.probabilities))
        budget = self.cfg.budgets[profile]
        added, seed = 0, -1
        if budget:
            variant = int(rng.integers(len(self.cache_paths)))
            with self.cache_paths[variant].open('rb') as f:
                f.seek(self.cache_offsets[variant][index]); cached = json.loads(f.readline())
            if cached['id'] != row['id']:
                raise ValueError('Augmentation sample mismatch')
            pool = cached['added_objects']
            seed = self.manifest['files'][variant]['seed']
            n = len(pool) if budget == -1 else min(budget, len(pool))
            indices = sorted(rng.permutation(len(pool))[:n].tolist())
            selected = [(pool[i]['kind'], {k: v for k, v in pool[i].items() if k != 'kind'}) for i in indices]
            base, kept = append_observations(base, selected, self.max_points)
            if len(kept) != n:
                raise ValueError('Cache contains more objects than available padding')
            added = len(kept)
        return {**base, '_augmentation_profile': torch.tensor(profile),
                '_added_tokens': torch.tensor(added), '_supplement_seed': torch.tensor(seed)}
