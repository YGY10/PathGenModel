"""Prepare small Guide-only training cache; does not create/modify Worlds or GT."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from shapely.geometry import LineString
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Dataset.context_augmentation import file_sha256
from Scenario.build_dataset_v4 import route_base,sample_world_guide
from World.road_warp import ArmWarp,deform
from scripts.evaluate_guide_conflicts import perturb_canonical
from train import inspect_split


def sample_parameters(slot,rng):
    if slot==0:return -float(rng.uniform(1,6)),0.
    if slot==1:return float(rng.uniform(1,6)),0.
    if slot==2:return 0.,-float(rng.uniform(.5,3))
    if slot==3:return 0.,float(rng.uniform(.5,3))
    signs=[(-1,-1),(-1,1),(1,-1),(1,1)][slot-4]
    return signs[0]*float(rng.uniform(1,6)),signs[1]*float(rng.uniform(.5,3))


def build(source,output,seed=88421):
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Use a new cache directory')
    train_info=inspect_split(source/'audit/train.jsonl');val_info=inspect_split(source/'audit/val.jsonl')
    if set(train_info['worlds'])&set(val_info['worlds']):raise ValueError('Train/val World overlap')
    poses={}
    with (source/'audit/train.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['navigation'] not in ('left','right'):continue
            md=row['metadata'];key=(row['world_id'],row['navigation'],md['pose_name'])
            descriptor={'ego_world_xy':md['ego_world_xy'],'heading_rad':md['heading_rad']}
            if key in poses and poses[key]!=descriptor:raise ValueError('Quality copies have different poses')
            poses[key]=descriptor
    if not poses:raise ValueError('No turning samples')
    keys=sorted(poses);index={k:i for i,k in enumerate(keys)};grouped=defaultdict(list)
    for key in keys:grouped[key[:2]].append(key)
    output.mkdir(parents=True)
    arrays=np.lib.format.open_memmap(output/'guides.npy',mode='w+',dtype=np.float32,shape=(len(keys),8,81,2))
    parameters=[];rejected=0;closure=0.;current=None;world=None;routes=None
    for count,((wid,nav),pose_keys) in enumerate(sorted(grouped.items()),1):
        if current!=wid:
            world=json.loads((source/'worlds'/f'{wid}.json').read_text())
            routes=json.loads((source/'routes'/f'{wid}.json').read_text());current=wid
        route=routes[nav];cfg=SimpleNamespace(**world['metadata'])
        canonical,se,sx,_=route_base(cfg,cfg,nav,route['entry_lane_index'],route['exit_lane_index'],True,route['guide_parameters'])
        wm=world['metadata']['arm_warp'];warp=ArmWarp(**{k:tuple(v) if isinstance(v,list) else v for k,v in wm.items()})
        error=float(np.max(np.abs(deform(canonical,warp)-np.asarray(route['guide_world']))));closure=max(closure,error)
        if error>2e-5:raise ValueError('Original Guide reconstruction mismatch')
        identity=int.from_bytes(hashlib.sha256(f'{wid}/{nav}'.encode()).digest()[:8],'little')
        rng=np.random.default_rng(np.random.SeedSequence([seed,identity]))
        for slot in range(8):
            for attempt in range(64):
                timing,lateral=sample_parameters(slot,rng)
                guide=deform(perturb_canonical(canonical,se,sx,timing,lateral),warp)
                if np.isfinite(guide).all() and LineString(guide).is_simple:break
                rejected+=1
            else:raise ValueError(f'Could not generate simple Guide: {wid}/{nav}/{slot}')
            for key in pose_keys:
                md=poses[key];h=md['heading_rad']
                crop,_=sample_world_guide(guide,np.asarray(md['ego_world_xy']),np.array([np.cos(h),np.sin(h)]))
                if not np.isfinite(crop).all() or np.linalg.norm(np.diff(crop,axis=0),axis=1).max()>1.001:
                    raise ValueError('Invalid Guide crop')
                arrays[index[key],slot]=crop
            parameters.append({'world_id':wid,'navigation':nav,'slot':slot,'timing_m':timing,'lateral_m':lateral})
        if count%60==0:print('Prepared',count,'routes',flush=True)
    arrays.flush();del arrays
    (output/'keys.json').write_text(json.dumps(keys)+'\n')
    (output/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    manifest={'version':1,'source':train_info,'validation_worlds_excluded':val_info['worlds'],
        'seed':seed,'variants_per_route':8,'keys':len(keys),'routes':len(grouped),
        'guide_route_reconstruction_max_m':closure,'rejected_self_intersection_draws':rejected,
        'files':{name:file_sha256(output/name) for name in ('guides.npy','keys.json')},
        'policy':'2 timing-only,2 lateral-only,4 joint signed continuous variants per World/nav. Fixed GT/BEV/nav. Straight inputs unchanged.',
        'parameters_sha256':file_sha256(output/'parameters.json')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('source','validation_worlds_excluded','files')},indent=2))
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('outputs/v4_world300_inside13'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/guide_train_cache_v1'))
    ap.add_argument('--seed',type=int,default=88421)
    a=ap.parse_args();build(a.source,a.output_dir,a.seed)
