"""Add observed-curb metrics to a completed Guide benchmark without new inference.

Useful for the initial run that began before the visible-curb metric was added.
Future evaluate_guide_conflicts.py runs already compute these metrics directly.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Dataset.dataset import encode_sample
from scripts.evaluate_guide_conflicts import observed_curb_metrics, summarize, write_csv, non_guide_fingerprint


def parse_value(v):
    if v=='':return None
    if v=='True':return True
    if v=='False':return False
    try:return float(v)
    except ValueError:return v


def run(output):
    summary=json.loads((output/'summary.json').read_text())
    records=[{k:parse_value(v) for k,v in r.items()} for r in csv.DictReader((output/'per_case.csv').open())]
    arrays=np.load(output/'predictions.npz',allow_pickle=False)
    assert [r['case_id'] for r in records]==arrays['ids'].tolist()
    raw={r['id']:r for r in (json.loads(line) for line in Path(summary['source']).open())}
    saved=torch.load(summary['checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config']
    enc={};checked=set()
    with (output/'cases.jsonl').open() as stream:
        for i,(rec,line) in enumerate(zip(records,stream)):
            recipe=json.loads(line);assert recipe['case_id']==rec['case_id']
            row=raw[rec['base_id']]
            if row['id'] not in checked:
                assert non_guide_fingerprint(row)==recipe['non_guide_sha256']
                enc[row['id']]=encode_sample(row,cfg.get('max_lines',32),cfg.get('max_points',32));checked.add(row['id'])
            rec.update(observed_curb_metrics(np.asarray(recipe['reference_line']),arrays['pred'][i],np.asarray(row['gt_path']),enc[row['id']]))
            if (i+1)%1000==0:print('Observed metrics',i+1,flush=True)
    assert i+1==len(records)
    summary['overall']=summarize(records)
    for field in ['variant','severity','navigation','quality','pose']:
        summary['by_'+field]={str(v):summarize([r for r in records if r[field]==v]) for v in dict.fromkeys(r[field] for r in records)}
    summary['by_observed_guide_conflict']={str(v):summarize([r for r in records if r['guide_observed_curb_conflict']==v]) for v in [False,True]}
    summary['checks']['observed_geometry_cases_checked']=len(records)
    summary['limits'].append('Observed curb crossings use retained encoded fragments; GT-intersecting noisy fragments excluded from GT-clear metric and included separately in raw metric.')
    write_csv(output/'per_case.csv',records)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['by_observed_guide_conflict'],indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--benchmark',type=Path,required=True)
    run(ap.parse_args().benchmark)
