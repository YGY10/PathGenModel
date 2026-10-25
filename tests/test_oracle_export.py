import json

import numpy as np

from Dataset.dataset import encode_sample
from Scenario.build_dataset_v4 import build_one, DEFAULT_QUALITIES
from Scenario.export_oracle_lanes import oracle_row, export, entry_boundaries
from World.intersection_generator import IntersectionConfig


def test_oracle_geometry_independent_of_gt_and_correct_ids():
    world,rows,routes=build_one(30000000,IntersectionConfig(),{'clean':DEFAULT_QUALITIES['clean']})
    row=rows[0];route=routes[row['navigation']]
    result,_=oracle_row(row,world,route)
    assert set(entry_boundaries(world,route))<=set(result['metadata']['oracle_selected_source_ids'])
    changed={**row,'gt_path':(np.asarray(row['gt_path'])+100).tolist()}
    altered,_=oracle_row(changed,world,route)
    a,b=encode_sample(result),encode_sample(altered)
    for key in ('bev_points','point_mask','polyline_attr','guide','navigation'):
        assert (a[key]==b[key]).all()
    assert a['polyline_mask'].sum()==4
    assert result['reference_line']==row['reference_line']
    assert result['gt_path']==row['gt_path']


def test_export_keeps_world_splits_and_deduplicates_quality(tmp_path):
    source=tmp_path/'source'
    for d in ('worlds','routes','audit'):(source/d).mkdir(parents=True)
    (source/'config.json').write_text('{}')
    total=0
    for split,seed in zip(('train','val','test'),(30000000,30007919,30015838)):
        world,rows,routes=build_one(seed,IntersectionConfig(),{q:DEFAULT_QUALITIES[q] for q in ('clean','hard')})
        wid=world['world_id']
        (source/'worlds'/f'{wid}.json').write_text(json.dumps(world))
        (source/'routes'/f'{wid}.json').write_text(json.dumps(routes,default=lambda x:x.tolist()))
        (source/'audit'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        total+=len(rows)
    report=export(source,tmp_path/'oracle',fit_worlds=1,fit_val_worlds=1)
    assert report['output_samples']*2==total
    assert set(report['fit_train']['worlds']).isdisjoint(report['fit_val']['worlds'])
    for split in ('train','val','test'):
        assert report['source_splits'][split]['worlds']==report['output_splits'][split]['worlds']
