import json
from pathlib import Path
import numpy as np
from Dataset.audit import stable_split, box_ok, run


def test_disjoint_deterministic():
    worlds=[f'w{i:03d}' for i in range(101)]
    a=stable_split(worlds,123)
    assert a==stable_split(list(reversed(worlds)),123)
    assert len(a)==101
    assert {s:list(a.values()).count(s) for s in ('train','val','test')} == {'train':81,'val':10,'test':10}


def test_boxes():
    assert box_ok({'points':[[0,0],[2,0],[2,1],[0,1]]})
    assert not box_ok({'points':[[0,0],[2,0],[1,1],[0,1]]})


def test_audit_small(tmp_path):
    rows=[]
    for w in ('w1','w2','w3'):
        for q in ('clean','hard'):
            rows.append({'id':w+'_'+q,'world_id':w,'route_id':w+'_r','ego_pose_id':'p0',
             'navigation':'left','gt_path':[[float(i),0] for i in range(81)],
             'reference_line':[[float(i),.5] for i in range(81)],
             'lane_markings':[],'road_boundaries':[],'stop_lines':[], 'crosswalks':[],
             'metadata':{'quality':q,'pose_name':'approach_30'}})
    path=tmp_path/'in.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    summary=run(path,tmp_path/'out')
    assert summary['samples']==6
    assert summary['geometry_issues']==0
    assert sum(summary['split_world_counts'].values())==3
    manifests=json.loads((tmp_path/'out'/'split_manifest.json').read_text())
    sets=[set(x) for x in manifests['worlds'].values()]
    assert all(not (sets[i]&sets[j]) for i in range(3) for j in range(i+1,3))
