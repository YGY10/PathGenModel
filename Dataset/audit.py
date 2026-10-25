"""World-disjoint splits and honest diagnostics for synthetic intersection data.

No topology or ground truth is copied into the model feature fields.
Exit visibility is an explicitly named geometrical PROXY (not confirmed target-lane visibility).
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SPLITS = ('train', 'val', 'test')


def stable_split(world_ids, seed=42, ratios=(.8, .1, .1)):
    if any(x < 0 for x in ratios) or abs(sum(ratios)-1) > 1e-8:
        raise ValueError('Split ratios must be nonnegative and sum to 1')
    ordered = sorted(world_ids, key=lambda w: hashlib.sha256(f'{seed}|{w}'.encode()).hexdigest())
    n = len(ordered)
    # Largest remainder allocation gives well-defined counts even for small datasets.
    desired = np.asarray(ratios)*n
    counts = np.floor(desired).astype(int)
    for i in np.argsort(-(desired-counts), kind='stable')[:n-int(counts.sum())]:
        counts[i] += 1
    result = {}
    pos = 0
    for name, count in zip(SPLITS, counts):
        for w in ordered[pos:pos+count]:
            result[w] = name
        pos += count
    return result


def obj_ids(items):
    return {str(x.get('source_id', x.get('id', ''))) for x in items if x.get('source_id', x.get('id')) is not None}


def box_ok(obj):
    p = np.asarray(obj.get('points', []), dtype=float)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        return False
    a, b, c, d = p
    u=b-a; v=d-a
    return bool(abs(np.dot(u, v)) < 0.015*max(np.linalg.norm(u)*np.linalg.norm(v), 1e-9)
                and np.linalg.norm(u) > .03 and np.linalg.norm(v) > .03
                and np.linalg.norm(a+c-b-d)<.06)


def visible_support(row):
    """GT neighborhood support proxy, NOT ground-truth exit lane boundary matching."""
    p = np.asarray(row['gt_path'], float)
    if p.shape != (81, 2):
        return 'invalid', 0
    tang = np.gradient(p, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-8)
    normal = np.column_stack((-tang[:, 1], tang[:, 0]))
    segments = []
    for obj in row.get('lane_markings', []) + row.get('road_boundaries', []):
        q = np.asarray(obj.get('points', []), float)
        if q.ndim == 2 and q.shape[1] == 2 and len(q) >= 2:
            segments.append((q[:-1], np.diff(q, axis=0)))
    if not segments:
        return 'none_proxy', 0
    # Only the future half of the path is measured; true exit station is not in V4 rows.
    # Do not label this as strict target exit visibility.
    two = 0; one = 0
    for i in range(40,81):
        left = right = False
        for a, d in segments:
            v = p[i] - a
            t=np.clip(np.einsum('ij,ij->i', v, d)/np.maximum(np.einsum('ij,ij->i',d,d),1e-9),0,1)
            diff = (a+t[:,None]*d)-p[i]
            k=int(np.argmin(np.linalg.norm(diff,axis=1)))
            lateral=float(diff[k] @ normal[i]); longitudinal=float(diff[k] @ tang[i])
            if .3 <= lateral <= 5.0 and abs(longitudinal)<=1.5: left=True
            if -5.0 <= lateral <= -.3 and abs(longitudinal)<=1.5: right=True
        one += left or right
        two += left and right
    return ('two_sided_proxy' if two>=5 else 'partial_proxy' if one>=5 else 'none_proxy'), two


def run(input_path, output_dir, seed=42, ratios=(.8,.1,.1), worlds_dir=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]; by_world=defaultdict(list)
    with Path(input_path).open(encoding='utf8') as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            r=json.loads(line)
            world_id=r.get('world_id')
            if not world_id: raise ValueError(f'Line {n} has no world_id: cannot prevent leakage')
            rows.append(r);by_world[world_id].append(r)
    if not rows:raise ValueError('No samples')
    assignments=stable_split(by_world, seed, ratios)
    manifest={'seed':seed,'ratios':dict(zip(SPLITS,ratios)),
              'worlds':{s:sorted(w for w,v in assignments.items() if v==s) for s in SPLITS}}
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    handles={s:(out/f'{s}.jsonl').open('w',encoding='utf8') for s in SPLITS}
    records=[]; seen_gt={}; failed=[]; group_count=Counter()
    try:
        for r in rows:
            split=assignments[r['world_id']]
            handles[split].write(json.dumps(r,ensure_ascii=False)+'\n')
            gt=np.asarray(r.get('gt_path',[]),float)
            ref=np.asarray(r.get('reference_line',[]),float)
            key=(r.get('route_id'),r.get('ego_pose_id'))
            issues=[]
            if gt.shape!=(81,2) or ref.shape!=(81,2): issues.append('path_shape')
            else:
                if np.max(np.abs(np.linalg.norm(np.diff(gt,axis=0),axis=1)-1))>.035:issues.append('gt_spacing')
                if np.linalg.norm(gt[0])>.025:issues.append('gt_origin')
            if key in seen_gt and (gt.shape!=seen_gt[key].shape or not np.array_equal(gt,seen_gt[key])):
                issues.append('gt_changed_with_perception')
            else:seen_gt[key]=gt.copy()
            for item in r.get('crosswalks',[]):
                if not box_ok(item):issues.append('crosswalk_not_box')
            for item in r.get('stop_lines',[]):
                if not box_ok(item):issues.append('stopline_not_box')
            if len(r.get('lane_markings',[]))+len(r.get('road_boundaries',[]))>32:issues.append('over_capacity')
            md=r.get('metadata',{}); qual=md.get('quality','unknown'); pose=md.get('pose_name','unknown');nav=r.get('navigation','unknown')
            cat,two=visible_support(r) if gt.shape==(81,2) else ('invalid',0)
            guide_ade=float(np.linalg.norm(ref-gt,axis=1).mean()) if gt.shape==ref.shape==(81,2) else None
            guide_fde=float(np.linalg.norm(ref[-1]-gt[-1])) if guide_ade is not None else None
            rec={'id':r.get('id'),'split':split,'world_id':r['world_id'],'route_id':r.get('route_id'),
                 'pose':pose,'navigation':nav,'quality':qual,'exit_visibility_proxy':cat,
                 'two_sided_future_points':two,
                 'observed_line_segments':len(r.get('lane_markings',[])),
                 'observed_curb_segments':len(r.get('road_boundaries',[])),
                 'observed_stop_boxes':len(r.get('stop_lines',[])),
                 'observed_crosswalk_boxes':len(r.get('crosswalks',[])),
                 'observed_unique_lines':len(obj_ids(r.get('lane_markings',[]))),
                 'observed_unique_curbs':len(obj_ids(r.get('road_boundaries',[]))),
                 'guide_ade_m':guide_ade,'guide_fde_m':guide_fde,
                 'issues':';'.join(sorted(set(issues)))}
            records.append(rec)
            group_count[(split,nav,pose,qual,cat)]+=1
            for issue in issues:failed.append({'sample_id':r.get('id'),'issue':issue})
    finally:
        for f in handles.values():f.close()
    def csvout(name, data, fields):
        with (out/name).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
    csvout('per_sample.csv',records,list(records[0]))
    csvout('issues.csv',failed,['sample_id','issue'])
    groups=[{'split':s,'navigation':nav,'pose':pose,'quality':q,'visibility_proxy':cat,'count':cnt}
            for (s,nav,pose,q,cat),cnt in sorted(group_count.items())]
    csvout('visibility_by_group.csv',groups,['split','navigation','pose','quality','visibility_proxy','count'])
    # Optional truth-object count comparison; source IDs are withheld from model features.
    truth_counts={}
    if worlds_dir:
        for wid in by_world:
            file=Path(worlds_dir)/f'{wid}.json'
            if file.exists():
                data=json.loads(file.read_text());truth_counts[wid]=dict(Counter(o['type'] for o in data.get('physical',{}).get('elements',[])))
    summary={'samples':len(rows),'worlds':len(by_world),'split_sample_counts':dict(Counter(assignments[r['world_id']] for r in rows)),
             'split_world_counts':dict(Counter(assignments.values())),'geometry_issues':len(failed),
             'quality_counts':dict(Counter(x['quality'] for x in records)),
             'detected_box_counts':{'stop_line':sum(x['observed_stop_boxes'] for x in records),
                                    'crosswalk':sum(x['observed_crosswalk_boxes'] for x in records)},
             'visibility_proxy_counts':dict(Counter(x['exit_visibility_proxy'] for x in records)),
             'truth_elements_by_world':truth_counts,
             'limitations':['visibility is future-GT-neighborhood proxy, not target exit lane identity',
                            'actual box missed-rate requires matching every truth box to observations by ego pose/FOV',
                            'GT lane polygon legality must be checked against world truth by a separate geometry validator']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--worlds-dir')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-ratio',type=float,default=.8)
    p.add_argument('--val-ratio',type=float,default=.1)
    args=p.parse_args()
    run(args.input,args.output_dir,args.seed,(args.train_ratio,args.val_ratio,1-args.train_ratio-args.val_ratio),args.worlds_dir)

if __name__=='__main__':main()
