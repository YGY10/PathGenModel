import copy
import json
from pathlib import Path
import numpy as np
import pytest
from Dataset.dataset import encode_sample, orient_near_to_far, resample, PathJSONLDataset


def row():
    return dict(navigation='left', reference_line=[[float(i),0.] for i in range(81)],
                gt_path=[[float(i),0.] for i in range(81)],
                lane_markings=[dict(points=[[12.,1.],[8.,1.],[3.,1.]],confidence=0.9,color='yellow',line_style='dashed')],
                road_boundaries=[], stop_lines=[dict(points=[[1,2],[1,3],[2,3],[2,2]],confidence=.8)], crosswalks=[])


def test_orientation_and_sampling():
    p = resample([[12.,1.],[8.,1.],[3.,1.]])
    assert np.linalg.norm(p[0]) < np.linalg.norm(p[-1])
    assert len(p)==32
    assert np.all(np.diff(p[:,0])>=0)


def test_masks_and_features():
    t=encode_sample(row())
    assert t['bev_points'].shape==(32,32,6)
    assert t['polyline_attr'].shape==(32,10)
    assert t['point_mask'].shape==(32,32)
    assert t['polyline_mask'].sum()==2
    assert t['point_mask'].sum()==64
    assert not t['point_mask'][2:].any()
    assert t['bev_points'][2:].abs().sum()==0
    assert t['polyline_attr'][0,2]==1  # box first
    assert t['polyline_attr'][1,0]==1
    assert t['bev_points'][1,0,4]==0  # first station


def test_near_origin_is_valid():
    r=row(); r['lane_markings'][0]['points']=[[0.,0.],[1.,0.]]
    t=encode_sample(r)
    assert t['point_mask'][1,0].item()


def test_closed_box_perimeter():
    r=row(); t=encode_sample(r)
    xy=t['bev_points'][0,:,:2].numpy()
    assert xy.min()>=.99 and xy.max()<=3.01
    assert np.max(np.linalg.norm(np.diff(xy,axis=0),axis=1))<.2


def test_jsonl(tmp_path):
    p=tmp_path/'train.jsonl';p.write_text(json.dumps(row())+'\n')
    t=PathJSONLDataset(p)
    assert len(t)==1 and t[0]['navigation'].item()==1


def test_invalid_shape():
    r=row();r['gt_path']=r['gt_path'][:-1]
    with pytest.raises(ValueError):encode_sample(r)
