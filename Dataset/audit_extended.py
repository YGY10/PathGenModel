"""V4 dataset post-audit: distributions, box FOV opportunity, route consistency.
Run after Dataset.audit. Does not claim strict lane-based visibility without Lane boundary IDs.
"""
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np


def read_jsonl(path):
    with open(path,encoding='utf8') as f:
        for ln in f:
            if ln.strip(): yield json.loads(ln)

def write_csv(path,rows,fields):
    with open(path,'w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def transform_to_vcs(points,origin,heading):
    p=np.asarray(points,dtype=float)[:,:2]-np.asarray(origin,dtype=float)
    c,s=math.cos(heading),math.sin(heading)
    return np.column_stack((c*p[:,0]+s*p[:,1],-s*p[:,0]+c*p[:,1]))

def polygon_clipped_area(poly, fov):
    # Sutherland-Hodgman clipping against VCS rectangle, used ONLY for box visibility opportunity.
    q=np.asarray(poly,dtype=float)
    if q.shape!=(4,2):return 0.0
    for axis, limit, keep_less in [(0,fov['forward'],True),(0,-fov['rear'],False),(1,fov['left'],True),(1,-fov['right'],False)]:
        if len(q)==0:return 0.0
        acc=[]
        def inside(v):return v[axis]<=limit+1e-9 if keep_less else v[axis]>=limit-1e-9
        for i in range(len(q)):
            a,b=q[i-1],q[i];ia,ib=inside(a),inside(b)
            if ia!=ib:
                t=(limit-a[axis])/(b[axis]-a[axis]);acc.append(a+t*(b-a))
            if ib:acc.append(b)
        q=np.asarray(acc).reshape(-1,2)
    if len(q)<3:return 0.0
    return float(abs(np.dot(q[:,0],np.roll(q[:,1],-1))-np.dot(q[:,1],np.roll(q[:,0],-1)))*.5)

def area(q):
    q=np.asarray(q,float)
    return float(abs(np.dot(q[:,0],np.roll(q[:,1],-1))-np.dot(q[:,1],np.roll(q[:,0],-1)))*.5)

def line_stats(path):
    p=np.asarray(path,float)
    if len(p)<2:return None
    return np.cumsum(np.r_[0,np.linalg.norm(np.diff(p,axis=0),axis=1)])

def audit(rows,worlds_dir,routes_dir,out,box_threshold=.10):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    byworld=defaultdict(list)
    for r in rows:byworld[r['world_id']].append(r)
    worlds={};routes={};problems=[];world_stats=[]
    for wid,rr in byworld.items():
        wp=Path(worlds_dir)/f'{wid}.json'
        rp=Path(routes_dir)/f'{wid}.json'
        if not wp.exists():problems.append({'world_id':wid,'sample_id':'','issue':'missing_world_json'});continue
        w=json.loads(wp.read_text());worlds[wid]=w
        if rp.exists(): routes[wid]=json.loads(rp.read_text())
        else: problems.append({'world_id':wid,'sample_id':'','issue':'missing_routes_json'})
        meta=w.get('metadata',{});warp=meta.get('arm_warp',{})
        world_stats.append({'world_id':wid,'num_in':str(meta.get('num_in')),'num_out':str(meta.get('num_out')),
                            'angle_deg':str(warp.get('angle_deg')),'lateral_m':str(warp.get('lateral_m')),
                            'has_lane_boundary_mapping':bool(w.get('semantic',{}).get('lanes'))})
    details=[]; groups=defaultdict(list);opportunities=Counter();hits=Counter()
    for r in rows:
        wid=r['world_id'];md=r.get('metadata',{});nav=r['navigation'];pose=md.get('pose_name','unknown');quality=md.get('quality','unknown')
        key=(nav,pose,quality)
        gt=np.asarray(r['gt_path'],float);guide=np.asarray(r['reference_line'],float)
        ade=float(np.linalg.norm(gt-guide,axis=1).mean());fde=float(np.linalg.norm(gt[-1]-guide[-1]))
        groups[key].append((ade,fde))
        details.append({'world_id':wid,'sample_id':r['id'],'navigation':nav,'pose':pose,'quality':quality,'guide_ade_m':round(ade,5),'guide_fde_m':round(fde,5),'line_count':len(r.get('lane_markings',[])),'curb_count':len(r.get('road_boundaries',[])),'stop_count':len(r.get('stop_lines',[])),'crosswalk_count':len(r.get('crosswalks',[]))})
        if wid not in worlds:continue
        fov=md.get('vcs_fov_m',{});fov={'forward':fov.get('forward',0),'rear':fov.get('rear',0),'left':fov.get('left',0),'right':fov.get('right',0)}
        origin=md.get('ego_world_xy');heading=md.get('heading_rad')
        if origin is None or heading is None or min(fov.values())<=0:
            problems.append({'world_id':wid,'sample_id':r['id'],'issue':'missing_pose_or_fov'});continue
        for category,field in [('stop_line','stop_lines'),('crosswalk','crosswalks')]:
            observed={str(x.get('id',x.get('source_id',''))) for x in r.get(field,[])}
            true=[e for e in worlds[wid]['physical']['elements'] if e['type']==category]
            for elem in true:
                poly=transform_to_vcs(elem['points'],origin,heading)
                a=area(poly)
                frac=polygon_clipped_area(poly,fov)/max(a,1e-9)
                if frac >= box_threshold:
                    opportunities[(category,quality)]+=1
                    if elem['id'] in observed: hits[(category,quality)]+=1
        # GT world-coordinate closure: samples must correspond to their cached route, up to quantization.
        route=routes.get(wid,{}).get(nav)
        if route:
            global_route=np.asarray(route['gt_world'],float)
            sroute=line_stats(global_route)
            s0=float(md.get('ego_route_s_m',0))
            if sroute is None or s0+80>sroute[-1]+1e-2:
                problems.append({'world_id':wid,'sample_id':r['id'],'issue':'global_route_too_short'})
            else:
                # Sample world route at arc length s0 + i and compare ego transformed GT.
                tg=np.arange(81,dtype=float)+s0
                expected=np.column_stack([np.interp(tg,sroute,global_route[:,j]) for j in range(2)])
                actual=transform_to_vcs(expected,origin,heading)
                diff=np.linalg.norm(actual-gt,axis=1)
                if diff.max()>.025: problems.append({'world_id':wid,'sample_id':r['id'],'issue':f'world_route_consistency:{diff.max():.3f}m'})
            # Fixed-guide datasets must reconstruct the SAME cached world guide
            # at all ego poses. Legacy anchored datasets have no such convention.
            if md.get('guide_coordinate_convention') == 'world_fixed_ego_transform_v1':
                guide_world = np.asarray(route.get('guide_world', []), float)
                gs = md.get('guide_start_s_m')
                arc = line_stats(guide_world)
                if gs is None or arc is None or gs < 0 or gs + 80 > arc[-1] + 1e-2:
                    problems.append({'world_id':wid,'sample_id':r['id'],'issue':'fixed_guide_station_invalid'})
                else:
                    stations = float(gs) + np.arange(81)
                    expected_guide = np.column_stack([np.interp(stations, arc, guide_world[:,j]) for j in range(2)])
                    expected_local = transform_to_vcs(expected_guide, origin, heading)
                    deviation = float(np.linalg.norm(expected_local-guide, axis=1).max())
                    if deviation > .0002:
                        problems.append({'world_id':wid,'sample_id':r['id'],
                                         'issue':f'fixed_guide_world_consistency:{deviation:.6f}m'})
    gr=[]
    for (nav,pose,quality),vals in sorted(groups.items()):
        a=np.asarray(vals)
        gr.append({'navigation':nav,'pose':pose,'quality':quality,'samples':len(vals),'guide_ade_mean_m':round(float(a[:,0].mean()),4),'guide_fde_mean_m':round(float(a[:,1].mean()),4),'guide_ade_p90_m':round(float(np.percentile(a[:,0],90)),4)})
    box=[]
    for k,n in sorted(opportunities.items()):
        h=hits[k];box.append({'type':k[0],'quality':k[1],'fov_eligible':n,'matched':h,'recall_given_fov':round(h/n,4)})
    write_csv(out/'distribution_by_nav_pose_quality.csv',gr,['navigation','pose','quality','samples','guide_ade_mean_m','guide_fde_mean_m','guide_ade_p90_m'])
    write_csv(out/'per_sample_extended.csv',details,list(details[0]))
    write_csv(out/'world_parameters.csv',world_stats,['world_id','num_in','num_out','angle_deg','lateral_m','has_lane_boundary_mapping'])
    write_csv(out/'box_fov_recall.csv',box,['type','quality','fov_eligible','matched','recall_given_fov'])
    write_csv(out/'extended_issues.csv',problems,['world_id','sample_id','issue'])
    summary={'sample_count':len(rows),'world_count':len(byworld),'issues':len(problems),'box_fov_recall':box,
      'strict_exit_visibility':'NOT_AVAILABLE: semantic.lanes lacks target lane boundary IDs',
      'strict_gt_drivable_polygon':'NOT_AVAILABLE: explicit drivable polygon or lane polygon missing',
      'fov_recall_definition':f'fraction of truth rectangle area inside VCS FOV >= {box_threshold}, matched by ID; not sensor detection recall'}
    (out/'extended_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--worlds-dir',required=True);p.add_argument('--routes-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--box-visible-fraction',type=float,default=.10)
    a=p.parse_args();audit(list(read_jsonl(a.input)),a.worlds_dir,a.routes_dir,a.output_dir,a.box_visible_fraction)
if __name__=='__main__':main()
