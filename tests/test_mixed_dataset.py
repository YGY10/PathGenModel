from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from Scenario.prepare_mixed_dataset import prepare, refresh_observation
from Scenario.build_dataset_v4 import DEFAULT_QUALITIES, PERCEPTION_CONVENTION
from train import TrainConfig, train


def make_source(root, name):
    (root/'audit').mkdir(parents=True)
    (root/'worlds').mkdir()
    for split in ('train','val','test'):
        wid=f'{name}_{split}'
        row={'id':f'{wid}_0','world_id':wid,'navigation':'straight',
             'reference_line':[[float(i),.5] for i in range(81)],
             'gt_path':[[float(i),0.] for i in range(81)],
             'lane_markings':[],'road_boundaries':[],'crosswalks':[],'stop_lines':[],
             'metadata':{'quality':'clean','pose_name':'entry','ego_world_xy':[0.,0.],
                         'heading_rad':0.,'scene_type':name}}
        (root/'audit'/f'{split}.jsonl').write_text(json.dumps(row)+'\n')
        world={'physical':{'elements':[{'id':'line','type':'line','points':[[0.,2.],[85.,2.]]}]}}
        (root/'worlds'/f'{wid}.json').write_text(json.dumps(world))
    return root


def test_mixing_preserves_labels_splits_and_subsets(tmp_path):
    sources={name:make_source(tmp_path/name,name) for name in ('intersection','lane_transition')}
    out=tmp_path/'mixed'
    report=prepare(sources,out,refresh=('intersection',),seed=11)
    assert report['counts']['train']=={'intersection':1,'lane_transition':1}
    for split in ('train','val','test'):
        combined=[json.loads(s) for s in (out/'audit'/f'{split}.jsonl').open()]
        for name in sources:
            raw=json.loads((sources[name]/'audit'/f'{split}.jsonl').read_text())
            subset=json.loads((out/'subsets'/name/f'{split}.jsonl').read_text())
            assert subset in combined
            for key in ('id','world_id','gt_path','reference_line'):
                assert subset[key]==raw[key]
            if name=='intersection':
                assert subset['metadata']['perception_convention']==PERCEPTION_CONVENTION
    with pytest.raises(FileExistsError):prepare(sources,out)


def test_world_overlap_is_rejected_before_output(tmp_path):
    root=make_source(tmp_path/'source','intersection')
    path=root/'audit'/'val.jsonl'
    row=json.loads(path.read_text());row['world_id']='intersection_train'
    path.write_text(json.dumps(row)+'\n')
    out=tmp_path/'mixed'
    with pytest.raises(ValueError,match='World overlap'):prepare({'intersection':root},out)
    assert not out.exists()


def test_refresh_is_deterministic_and_drops_stale_support():
    row={'id':'sample','reference_line':[[0,0]],'gt_path':[[0,0]],'evaluation':{'support_heuristic':[True]},
         'metadata':{'quality':'clean','heading_rad':.3,'ego_world_xy':[10.,20.]}}
    world={'physical':{'elements':[{'id':'line','type':'line','points':[[0.,2.],[85.,2.]]}]}}
    a=refresh_observation(row,world,DEFAULT_QUALITIES,42)
    b=refresh_observation(row,world,DEFAULT_QUALITIES,42)
    assert a==b and a['evaluation']=={}
    assert a['gt_path']==row['gt_path'] and a['reference_line']==row['reference_line']
    assert row['evaluation']  # Input untouched.


def test_training_logs_separate_validation_groups(tmp_path):
    sources={name:make_source(tmp_path/name,name) for name in ('intersection','lane_transition')}
    out=tmp_path/'data'
    prepare(sources,out)
    cfg=TrainConfig(train_path=str(out/'audit'/'train.jsonl'),val_path=str(out/'audit'/'val.jsonl'),
                    validation_groups={name:str(out/'subsets'/name/'val.jsonl') for name in sources},
                    output_dir=str(tmp_path/'run'),epochs=1,batch_size=2,num_workers=0,device='cpu',
                    max_lines=2,max_points=4,tensorboard=True,
                    model={'d_model':16,'nhead':2,'encoder_layers':1,'decoder_layers':1,'ff_dim':32})
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:train(cfg)
    finally:torch.set_num_threads(previous)
    record=json.loads((tmp_path/'run'/'history.jsonl').read_text())
    assert set(record['validation_groups'])==set(sources)
    np.testing.assert_allclose(record['val']['ade_m'],
        np.mean([v['ade_m'] for v in record['validation_groups'].values()]),rtol=1e-6)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(tmp_path/'run'/'tensorboard'));events.Reload()
    assert events.Scalars('val_groups/intersection/ade_m')
    assert events.Scalars('val_groups/lane_transition/ade_m')
    invalid=replace(cfg,validation_groups={'bad':str(out/'subsets'/'intersection'/'train.jsonl')})
    with pytest.raises(ValueError,match='validation Worlds'):train(invalid)
