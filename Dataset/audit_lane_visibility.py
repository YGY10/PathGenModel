"""V4.1 route-target exit lane visibility audit for frozen V4 samples.

Maps target outgoing lane to independently generated physical boundary source IDs.
No inputs are added to BEVDataset; original data is never modified.
"""
import argparse,csv,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

ARMS=('south','east','north','west')

def read_jsonl(path):
    with open(path,encoding='utf8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def transform(points,origin,heading):
    p=np.asarray(points,float)[:,:2]-np.asarray(origin,float)
    c,s=np.cos(heading),np.sin(heading)
    return np.stack((c*p[:,0]+s*p[:,1],-s*p[:,0]+c*p[:,1]),-1)

def sample_at(poly,stations):
    p=np.asarray(poly,float)
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(d)>1e-8];p,d=p[keep],d[keep]
    if len(p)<2 or np.max(stations)>d[-1]+1e-3:return None
    return np.column_stack([np.interp(stations,d,p[:,j]) for j in (0,1)])

def point_polyline_distance(q,poly):
    """True point-to-segment distance, rather than nearest sparse vertex."""
    p=np.asarray(poly,float)
    if len(p)<2:return np.full(len(q),np.inf)
    a=p[:-1];v=np.diff(p,axis=0);den=np.maximum(np.sum(v*v,axis=1),1e-12)
    result=np.full(len(q),np.inf)
    for start in range(0,len(q),32):
        pts=q[start:start+32]
        u=np.clip(np.sum((pts[:,None,:]-a[None,:,:])*v[None,:,:],axis=2)/den[None,:],0,1)
        nearest=a[None,:,:]+u[:,:,None]*v[None,:,:]
        result[start:start+32]=np.min(np.linalg.norm(pts[:,None,:]-nearest,axis=2),axis=1)
    return result

def target_boundaries(world,route):
    arm=ARMS[int(route['exit_arm'])]
    lane=int(route['exit_lane_index'])
    no=int(world['metadata']['num_out'][int(route['exit_arm'])])
    if not 0<=lane<no:raise ValueError(f'invalid target exit lane {lane}/{no}')
    def name(boundary_index):
        if boundary_index==no:return f'{arm}_curb_outgoing'
        return f'{arm}_divider_{-boundary_index:+d}'
    return name(lane),name(lane+1)

def analyze(rows,worlds_dir,routes_dir,out,station_start=3.,station_end=25.,distance_m=1.25,required_points=5):
    worlds_dir=Path(worlds_dir);routes_dir=Path(routes_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cache={};stats=[];issues=[]
    for row in rows:
        wid=row['world_id'];nav=row['navigation'];md=row['metadata']
        try:
            if wid not in cache:
                w=json.loads((worlds_dir/f'{wid}.json').read_text());r=json.loads((routes_dir/f'{wid}.json').read_text())
                cache[wid]=(w,r,{e['id']:e for e in w['physical']['elements']})
            w,routes,physical=cache[wid];route=routes[nav]
            a_id,b_id=target_boundaries(w,route)
            if a_id not in physical or b_id not in physical:raise ValueError('missing target lane boundary physical element')
            gt_route=route['gt_world'];exit_s=float(route['exit_s']);total=float(np.sum(np.linalg.norm(np.diff(np.asarray(gt_route),axis=0),axis=1)))
            ss=exit_s+np.arange(station_start,min(station_end,total-exit_s-1e-4)+1e-6,1.)
            if len(ss)<required_points:raise ValueError('insufficient exit station samples')
            truth=sample_at(gt_route,ss)
            origin=md['ego_world_xy'];heading=md['heading_rad'];pts=transform(truth,origin,heading)
            fov=md['vcs_fov_m']
            fovmask=((pts[:,0]>=-fov['rear'])&(pts[:,0]<=fov['forward'])&(pts[:,1]>=-fov['right'])&(pts[:,1]<=fov['left']))
            observations=defaultdict(list)
            for entry in row.get('lane_markings',[])+row.get('road_boundaries',[]):
                if entry.get('source_id'):observations[entry['source_id']].append(np.asarray(entry['points'],float))
            side_masks=[];side_id=[]
            for bid in (a_id,b_id):
                # Match actual observed fragments by the exact ground-truth boundary source ID.
                # Ground-truth geometry is used only in evaluation, not supplied to the model.
                ground=transform(physical[bid]['points'],origin,heading)
                near_truth=[]
                for p in pts:
                    # Nearest true boundary point to target center station
                    v=ground[1:]-ground[:-1];den=np.maximum(np.sum(v*v,axis=1),1e-12)
                    t=np.clip(np.sum((p-ground[:-1])*v,axis=1)/den,0,1)
                    proj=ground[:-1]+t[:,None]*v
                    near_truth.append(proj[np.argmin(np.sum((p-proj)**2,axis=1))])
                near_truth=np.asarray(near_truth)
                detected=np.zeros(len(pts),bool)
                for obs in observations.get(bid,[]):
                    detected|=point_polyline_distance(near_truth,obs)<=distance_m
                side_masks.append(detected)
                side_id.append(bid)
            both=fovmask & side_masks[0] & side_masks[1]
            either=fovmask & (side_masks[0] | side_masks[1])
            eligible=int(fovmask.sum());full=int(both.sum());partial=int((either&~both).sum())
            stats.append({'world_id':wid,'sample_id':row['id'],'navigation':nav,'pose':md['pose_name'],'quality':md['quality'],
                          'exit_lane_index':int(route['exit_lane_index']),'left_edge_id':side_id[0],'right_edge_id':side_id[1],
                          'exit_stations':len(ss),'ego_fov_station_count':eligible,'two_side_stations':full,'one_side_stations':partial,
                          'no_side_stations':eligible-full-partial,'visibility':('two_sided' if full>=required_points else 'partial' if either.any() else 'none'),
                          'no_observation_possible':int(eligible==0)})
        except (KeyError,ValueError,IndexError,FileNotFoundError) as exc:
            issues.append({'sample_id':row.get('id','?'),'reason':str(exc)})
    fields=list(stats[0]) if stats else []
    if stats:
        with open(out/'per_sample_exit_visibility.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(stats)
    grouped=defaultdict(list)
    for x in stats:grouped[(x['navigation'],x['pose'],x['quality'])].append(x)
    groups=[]
    for k,xs in sorted(grouped.items()):
        groups.append({'navigation':k[0],'pose':k[1],'quality':k[2],'samples':len(xs),
                       'two_sided':sum(x['visibility']=='two_sided' for x in xs),
                       'partial':sum(x['visibility']=='partial' for x in xs),
                       'none':sum(x['visibility']=='none' for x in xs),
                       'fov_unavailable':sum(x['no_observation_possible'] for x in xs),
                       'mean_two_side_stations':round(float(np.mean([x['two_side_stations'] for x in xs])),2)})
    if groups:
        with open(out/'by_nav_pose_quality.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    with open(out/'mapping_issues.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['sample_id','reason']);writer.writeheader();writer.writerows(issues)
    summary={'samples_requested':len(rows),'samples_checked':len(stats),'worlds_checked':len(cache),'mapping_issues':len(issues),
             'visibility_class_counts':dict(Counter(x['visibility'] for x in stats)),
             'fov_unavailable':sum(x['no_observation_possible'] for x in stats),
             'metric_definition':f'Exact target exit boundary source IDs; stations {station_start:g}..{station_end:g}m after exit; both observed sides within {distance_m:g}m at >= {required_points} stations and ego FOV contains target center; excludes box geometry; pointwise association metric, not sensor-level visibility.',
             'gt_polygon_legality':'NOT_IMPLEMENTED: needs canonical drivable polygon and interior-path validation'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--worlds-dir',required=True);ap.add_argument('--routes-dir',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--distance-m',type=float,default=1.25);ap.add_argument('--min-points',type=int,default=5)
    args=ap.parse_args()
    analyze(list(read_jsonl(args.input)),args.worlds_dir,args.routes_dir,args.output_dir,distance_m=args.distance_m,required_points=args.min_points)
if __name__=='__main__':main()
