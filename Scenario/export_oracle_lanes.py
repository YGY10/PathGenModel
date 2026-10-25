"""Export privileged exact entry/target-exit boundaries, preserving World splits.

Uses lane identities to SELECT physical boundaries, never GT path coordinates to
construct features. Coarse Guide and GT remain unchanged. This is an Oracle
diagnostic dataset, not deployable perception.
"""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from shapely.geometry import LineString, box

from Dataset.dataset import resample
from Dataset.audit_lane_visibility import ARMS, target_boundaries, transform, point_polyline_distance
from Scenario.build_dataset import V4_POSES, plot
from Scenario.build_dataset_v4 import verify
from train import inspect_split

FOV={'forward':100.,'rear':100.,'left':100.,'right':100.}


def entry_boundaries(world,route):
    lane=int(route['entry_lane_index']);count=int(world['metadata']['num_in'][0])
    if not 0<=lane<count:raise ValueError('Invalid incoming lane')
    def source(k):
        return f'{ARMS[0]}_curb_incoming' if k==count else f'{ARMS[0]}_divider_{k:+d}'
    return source(lane),source(lane+1)


def line_parts(geom):
    if geom.is_empty:return []
    if geom.geom_type=='LineString':return [geom]
    return [p for child in getattr(geom,'geoms',[]) for p in line_parts(child)]


def oracle_row(row,world,route):
    """All numeric geometry is copied from physical elements and rigidly transformed."""
    required=list(dict.fromkeys((*entry_boundaries(world,route),*target_boundaries(world,route))))
    physical={e['id']:e for e in world['physical']['elements']}
    md=row['metadata'];out=copy.deepcopy(row)
    out.update(lane_markings=[],road_boundaries=[],stop_lines=[],crosswalks=[])
    out.pop('evaluation',None)  # Old noisy-perception support proxy is no longer applicable.
    out['id']=f'{row["world_id"]}_{row["navigation"]}_{md["pose_name"]}_oracle'
    out['metadata'].update(quality='oracle',oracle_source_sample_id=row['id'],
        observation_mode='oracle_entry_target_lane_boundaries_v1',vcs_fov_m=dict(FOV),
        oracle_selected_source_ids=required)
    envelope=box(-FOV['rear'],-FOV['right'],FOV['forward'],FOV['left'])
    represented=set();max_encoding_error=0.
    for source in required:
        obj=physical[source]
        xy=transform(obj['points'],md['ego_world_xy'],md['heading_rad'])
        parts=line_parts(LineString(xy).intersection(envelope))
        for part_index,part in enumerate(parts):
            if part.length<.1:continue
            points=np.asarray(part.coords)
            encoded=resample(points,32)
            max_encoding_error=max(max_encoding_error,float(point_polyline_distance(points,encoded).max()))
            entry={'id':f'{source}_oracle_{part_index}','source_id':source,'points':points.tolist(),'confidence':1.}
            if obj['type']=='line':
                entry.update(color=obj.get('color'),line_style=obj.get('style'))
                out['lane_markings'].append(entry)
            elif obj['type']=='curb':out['road_boundaries'].append(entry)
            else:raise ValueError('Oracle lane edge must be line or curb')
            represented.add(source)
    if represented!=set(required):raise ValueError(f'{row["id"]}: required edge outside Oracle envelope')
    if len(out['lane_markings'])+len(out['road_boundaries'])>32:raise ValueError('Oracle capacity exceeded')
    for key in ('gt_path','reference_line','navigation','route_id','ego_pose_id'):
        if out[key]!=row[key]:raise AssertionError(f'Non-perception field changed: {key}')
    return out,max_encoding_error


def export(source,output,quality='clean',fit_worlds=4,fit_val_worlds=2):
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Use a new output directory')
    if min(fit_worlds,fit_val_worlds)<1:raise ValueError('Fit World counts must be positive')
    split_paths={s:source/'audit'/f'{s}.jsonl' for s in ('train','val','test')}
    source_info={s:inspect_split(p) for s,p in split_paths.items()}
    for a,b in (('train','val'),('train','test'),('val','test')):
        if set(source_info[a]['worlds'])&set(source_info[b]['worlds']):raise ValueError('Source split overlap')
    if fit_worlds>len(source_info['train']['worlds']) or fit_val_worlds>len(source_info['val']['worlds']):
        raise ValueError('Too many fit Worlds requested')
    small_train=set(source_info['train']['worlds'][:fit_worlds]);small_val=set(source_info['val']['worlds'][:fit_val_worlds])
    output.mkdir(parents=True);(output/'audit').mkdir();(output/'fit').mkdir()
    for directory in ('worlds','routes'):shutil.copytree(source/directory,output/directory)
    for name in ('world.json','accepted_world_seeds.json','rejected_worlds.json'):
        if (source/name).exists():shutil.copy2(source/name,output/name)
    if (source/'audit/split_manifest.json').exists():
        shutil.copy2(source/'audit/split_manifest.json',output/'audit/split_manifest.json')
    config=json.loads((source/'config.json').read_text())
    config.update(oracle_source=str(source.resolve()),oracle_mode='exact_entry_target_lane_boundaries_v1',
        oracle_input_quality=quality,oracle_selected_lane_identity=True,
        qualities={'oracle':dict(forward_m=FOV['forward'],rear_m=FOV['rear'],left_m=FOV['left'],right_m=FOV['right'],
            line_miss_rate=0.,curb_miss_rate=0.,box_miss_rate=0.,break_rate=0.,sigma_m=0.,angle_sigma_deg=0.,offset_sigma_m=0.)})
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    counts=Counter();deviations=[];seen=set();first_group=[];first_world=None
    with (output/'samples.jsonl').open('w') as all_file, (output/'fit/train.jsonl').open('w') as fit_train, (output/'fit/val.jsonl').open('w') as fit_val:
        for split,path in split_paths.items():
            current=None;routes=None;world=None;group=[];source_pose_keys=set();selected_keys=set()
            with path.open() as inp,(output/'audit'/f'{split}.jsonl').open('w') as dst:
                for line in inp:
                    row=json.loads(line);pose_key=(row['world_id'],row['route_id'],row['ego_pose_id'])
                    source_pose_keys.add(pose_key)
                    if row['metadata']['quality']!=quality:continue
                    if pose_key in selected_keys:raise ValueError('Duplicate selected quality')
                    selected_keys.add(pose_key)
                    if row['world_id']!=current:
                        if group:verify(group)
                        current=row['world_id'];group=[]
                        world=json.loads((source/'worlds'/f'{current}.json').read_text())
                        routes=json.loads((source/'routes'/f'{current}.json').read_text())
                        if first_world is None:first_world=world
                    out,encoding_error=oracle_row(row,world,routes[row['navigation']])
                    if out['id'] in seen:raise ValueError('Oracle duplicate ID')
                    seen.add(out['id']);group.append(out);deviations.append(encoding_error)
                    serialized=json.dumps(out)+'\n';dst.write(serialized);all_file.write(serialized)
                    if split=='train' and current in small_train:fit_train.write(serialized)
                    if split=='val' and current in small_val:fit_val.write(serialized)
                    if current==first_world['world_id']:first_group.append(out)
                    counts[split]+=1
                if group:verify(group)
            if selected_keys!=source_pose_keys:raise ValueError(f'{split}: quality missing for some route poses')
            print(f'Exported {split}: {counts[split]} unique Oracle samples',flush=True)
    output_info={s:inspect_split(output/'audit'/f'{s}.jsonl') for s in split_paths}
    for s in split_paths:
        if output_info[s]['worlds']!=source_info[s]['worlds']:raise AssertionError('World assignments changed')
    plot(first_world,first_group,output/'preview.png',pose_order=V4_POSES)
    report={'mode':'ORACLE: true lane IDs select physical boundaries; no GT path coordinates construct input',
        'source_splits':source_info,'output_splits':output_info,'source_quality_kept':quality,
        'source_samples':sum(v['samples'] for v in source_info.values()),'output_samples':sum(counts.values()),
        'fit_train':inspect_split(output/'fit/train.jsonl'),'fit_val':inspect_split(output/'fit/val.jsonl'),
        'encoded_boundary_deviation_m':{'max':max(deviations),'p90':float(np.quantile(deviations,.9)),
            'samples_over_5cm':sum(d>.05 for d in deviations)},
        'notes':['Original GT, coarse Guide and Navigation unchanged; no target centerline or GT residual in features.',
                 'No perception noise/dropout; other lane candidates omitted to encode target identity implicitly.',
                 'Quality duplicates removed; exact original World train/val/test memberships retained.',
                 '32-point encoder still approximates physical curves; boundary approximation is explicitly measured.',
                 'Fit train/validation Worlds are disjoint. Evaluate LAST checkpoint on fit TRAIN in eval mode to test fitting.']}
    (output/'export_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/v4_world300_inside13'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/oracle_lane_v1'))
    p.add_argument('--quality',default='clean')
    p.add_argument('--fit-worlds',type=int,default=4)
    p.add_argument('--fit-val-worlds',type=int,default=2)
    a=p.parse_args();export(a.source,a.output_dir,a.quality,a.fit_worlds,a.fit_val_worlds)
