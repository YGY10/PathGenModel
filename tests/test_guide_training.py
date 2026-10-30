import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from Dataset.dataset import PathJSONLDataset
from Dataset.guide_augmentation import GuideAugmentationConfig,GuideAugmentedDataset,inspect_guide_cache
from Scenario.build_dataset_v4 import build_one,DEFAULT_QUALITIES
from World.intersection_generator import IntersectionConfig
from scripts.build_training_guide_cache import build
from train import TrainConfig,train,inspect_split,build_loader


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root=tmp_path_factory.mktemp('guide_training');source=root/'source'
    for d in ['audit','worlds','routes']:(source/d).mkdir(parents=True,exist_ok=True)
    for split,seed in [('train',30000000),('val',30007919)]:
        world,rows,routes=build_one(seed,IntersectionConfig(),{'clean':DEFAULT_QUALITIES['clean']})
        (source/'audit'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (source/'worlds'/f'{world["world_id"]}.json').write_text(json.dumps(world))
        serial={nav:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in route.items()} for nav,route in routes.items()}
        (source/'routes'/f'{world["world_id"]}.json').write_text(json.dumps(serial))
    cache=root/'cache';build(source,cache)
    return root,source,cache


def test_cache_is_training_only_and_verified(prepared):
    root,source,cache=prepared
    info=inspect_split(source/'audit/train.jsonl')
    assert inspect_guide_cache(cache,info)['manifest_sha256']
    wrong={**info,'sha256':'changed'}
    with pytest.raises(ValueError,match='match training split'):inspect_guide_cache(cache,wrong)
    manifest=json.loads((cache/'manifest.json').read_text())
    assert not set(manifest['source']['worlds'])&set(manifest['validation_worlds_excluded'])
    assert manifest['routes']==2 and manifest['variants_per_route']==8


def test_only_guide_changes_and_sampling_is_epoch_deterministic(prepared):
    _,source,cache=prepared;path=source/'audit/train.jsonl'
    cfg=GuideAugmentationConfig(str(cache),probability=1.)
    augmented=GuideAugmentedDataset(path,cfg);original=PathJSONLDataset(path)
    raw=[json.loads(line) for line in path.open()]
    for i,row in enumerate(raw):
        a,b=augmented[i],original[i]
        for key in b:
            if key!='guide':torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        assert bool(a['_guide_augmented'])==(row['navigation']!='straight')
        if row['navigation']=='straight':torch.testing.assert_close(a['guide'],b['guide'])
    idx=next(i for i,r in enumerate(raw) if r['navigation']=='right')
    augmented.set_epoch(7);first=augmented[idx]['guide'];torch.testing.assert_close(first,augmented[idx]['guide'],rtol=0,atol=0)
    choices=[]
    for e in range(10):augmented.set_epoch(e);choices.append(augmented[idx]['guide'].numpy().tobytes())
    assert len(set(choices))>1
    with pytest.raises(ValueError):GuideAugmentationConfig(str(cache),probability=1.1)


def test_validation_never_augmented_and_spawn_workers_agree(prepared):
    _,source,cache=prepared
    cfg=TrainConfig(train_path=str(source/'audit/train.jsonl'),val_path=str(source/'audit/val.jsonl'),
        batch_size=7,num_workers=0,guide_augmentation={'cache_dir':str(cache),'probability':.5})
    val=build_loader(cfg.val_path,cfg,False,torch.Generator().manual_seed(9),torch.device('cpu'))
    assert type(val.dataset) is PathJSONLDataset
    def collect(workers):
        loader=build_loader(cfg.train_path,replace(cfg,num_workers=workers),True,torch.Generator().manual_seed(9),torch.device('cpu'))
        loader.timeout = 30 if workers else 0
        loader.dataset.set_epoch(3)
        return torch.cat([b['guide'] for b in loader])
    torch.testing.assert_close(collect(0),collect(2),rtol=0,atol=0)


def test_guide_training_resume_is_identical(prepared,tmp_path):
    _,source,cache=prepared
    cfg=TrainConfig(train_path=str(source/'audit/train.jsonl'),val_path=str(source/'audit/val.jsonl'),
        output_dir=str(tmp_path/'resumed'),epochs=1,batch_size=8,device='cpu',max_lines=3,max_points=4,
        guide_augmentation={'cache_dir':str(cache),'probability':.5},
        model={'d_model':16,'nhead':2,'ff_dim':32,'encoder_layers':1,'decoder_layers':1,'dropout':.1})
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        train(cfg);train(replace(cfg,epochs=2),tmp_path/'resumed/last.pt')
        train(replace(cfg,epochs=2,output_dir=str(tmp_path/'full')))
        a=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
        b=torch.load(tmp_path/'full/last.pt',weights_only=False)
        for name in a['model']:torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
        assert 'guide_augmentation' in a['splits']
        history=[json.loads(line) for line in (tmp_path/'full/history.jsonl').open()]
        assert 'guide_augmentation' in history[0]['train']
        assert 'guide_augmentation' not in history[0]['val']
    finally:torch.set_num_threads(threads)
