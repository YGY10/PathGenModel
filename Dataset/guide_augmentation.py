"""Training-only deterministic multi-Guide sampling, fixed BEV / GT / nav."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from Dataset.dataset import PathJSONLDataset
from Dataset.context_augmentation import file_sha256


@dataclass(frozen=True)
class GuideAugmentationConfig:
    cache_dir: str
    probability: float = .5
    seed: int = 271828

    def __post_init__(self):
        if not self.cache_dir or not np.isfinite(self.probability) or not 0<=self.probability<=1:
            raise ValueError('Guide cache and probability in [0,1] required')
        if not isinstance(self.seed,int) or self.seed<0:
            raise ValueError('Guide seed must be a nonnegative integer')


def inspect_guide_cache(directory, source):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('version')!=1 or manifest['source']!=source:
        raise ValueError('Guide cache does not match training split')
    for file in ('guides.npy','keys.json'):
        if file_sha256(directory/file)!=manifest['files'][file]:
            raise ValueError(f'Guide cache fingerprint mismatch: {file}')
    guides=np.load(directory/'guides.npy',mmap_mode='r')
    keys=json.loads((directory/'keys.json').read_text())
    if guides.shape!=(len(keys),manifest['variants_per_route'],81,2) or guides.dtype!=np.float32:
        raise ValueError('Invalid Guide cache shape/dtype')
    tuples=[tuple(k) for k in keys]
    if len(set(tuples))!=len(tuples) or set(k[0] for k in tuples)!=set(source['worlds']):
        raise ValueError('Invalid Guide key/world membership')
    return {'manifest_sha256':file_sha256(directory/'manifest.json'),'files':manifest['files']}


class GuideAugmentedDataset(PathJSONLDataset):
    """Epoch/index RNG is independent of worker ordering; one sample per epoch.

    Cache holds common world-Guide variants cropped at each pose. Straight
    samples remain unchanged. Eligible left/right samples keep original Guide
    with probability 1-p and choose one cached variant with probability p.
    """
    def __init__(self,path,cfg,max_lines=32,max_points=32):
        super().__init__(path,max_lines,max_points)
        self.cfg=cfg;self.epoch=0;self.directory=Path(cfg.cache_dir);self._guides=None
        keys=json.loads((self.directory/'keys.json').read_text())
        self.key_index={tuple(k):i for i,k in enumerate(keys)}
        manifest=json.loads((self.directory/'manifest.json').read_text())
        self.variant_count=manifest['variants_per_route']
        self.row_keys=[]
        with self.path.open() as stream:
            for line in stream:
                row=json.loads(line)
                key=(row['world_id'],row['navigation'],row['metadata']['pose_name'])
                if row['navigation'] in ('left','right') and key not in self.key_index:
                    raise ValueError(f'Missing Guide variants for {key}')
                self.row_keys.append(key)

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def __getstate__(self):
        result=dict(self.__dict__);result['_guides']=None
        return result

    def __getitem__(self,index):
        base=super().__getitem__(index)
        rng=np.random.default_rng(np.random.SeedSequence([self.cfg.seed,self.epoch,index]))
        key=self.row_keys[index]
        changed=key in self.key_index and rng.random()<self.cfg.probability
        if changed:
            if self._guides is None:self._guides=np.load(self.directory/'guides.npy',mmap_mode='r')
            variant=int(rng.integers(self.variant_count))
            base['guide']=torch.from_numpy(np.array(self._guides[self.key_index[key],variant],copy=True))
        return {**base,'_guide_augmented':torch.tensor(changed,dtype=torch.bool)}
