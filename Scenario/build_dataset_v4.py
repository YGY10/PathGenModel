"""V4 multilanes + stochastic intersection geometry + configurable perception/guide.
Execute: python -m Scenario.build_dataset_v4 --worlds 4 --seed 42
"""
import argparse, json
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
from World.intersection_generator import IntersectionConfig
from World.intersection_generator_v4 import build_world_v4,parse_spec,arm_xy,OUT,LEFT
from World.road_warp import ArmWarp,deform,warp_world
from Scenario.build_dataset import (NAVS,V4_POSES,pose_seed_index, arclength,at_s,bezier,unit,to_ego,local_pose,
                                    sample_path,rect_observation,observation_support,plot)

@dataclass(frozen=True)
class PerceptionQuality:
    forward_m:float
    rear_m:float
    left_m:float
    right_m:float
    line_miss_rate:float
    curb_miss_rate:float
    box_miss_rate:float
    break_rate:float
    sigma_m:float
    angle_sigma_deg:float
    offset_sigma_m:float

DEFAULT_QUALITIES={
 'clean':PerceptionQuality(85,18,22,22,.02,.02,.02,.06,.018,.35,.025),
 'medium':PerceptionQuality(65,14,15,15,.16,.10,.12,.22,.07,1.5,.12),
 'hard':PerceptionQuality(45,10,9,9,.36,.26,.33,.48,.18,3.5,.30),
}

def read_qualities(path=None):
    if path is None:return DEFAULT_QUALITIES
    raw=json.loads(Path(path).read_text())
    keys=set(PerceptionQuality.__dataclass_fields__)
    out={}
    for name,settings in raw.items():
        missing=keys-set(settings)
        if missing:raise ValueError(f'{name}: missing {sorted(missing)}')
        if set(settings)-keys:raise ValueError(f'{name}: unknown fields {sorted(set(settings)-keys)}')
        q=PerceptionQuality(**settings)
        if min(q.forward_m,q.rear_m,q.left_m,q.right_m)<=0:raise ValueError('Bad FOV')
        if any(not 0<=getattr(q,k)<=1 for k in ('line_miss_rate','curb_miss_rate','box_miss_rate','break_rate')):
            raise ValueError('Probability outside [0,1]')
        out[name]=q
    return out

def observe(world,ego,t,q,rng,max_lines=32):
    # Avoid altering the V3 global quality presets; each call gets its own quality.
    local={'forward_m':q.forward_m,'rear_m':q.rear_m,'lateral_m':max(q.left_m,q.right_m),
           'sigma_m':q.sigma_m,'angle_sigma_deg':q.angle_sigma_deg}
    def visible(x):
        return (-q.rear_m<=x[...,0])&(x[...,0]<=q.forward_m)&(-q.right_m<=x[...,1])&(x[...,1]<=q.left_m)
    lines=[];curbs=[];stops=[];cross=[]
    for obj in world['physical']['elements']:
        kind=obj['type'];p=to_ego(np.asarray(obj['points']),ego,t)
        if kind in ('line','curb'):
            if rng.random()<(q.line_miss_rate if kind=='line' else q.curb_miss_rate):continue
            # Densely sample curves; break once or twice at continuous ranges, never vertex Bernoulli dropout.
            pp=[]
            for a,b in zip(p[:-1],p[1:]):
                n=max(1,int(np.ceil(np.linalg.norm(b-a)/.4)))
                pp.extend(a[None,:]+(np.arange(n)/n)[:,None]*(b-a))
            if len(p):pp.append(p[-1])
            pp=np.asarray(pp,float)
            mask=visible(pp)
            if len(pp)>15 and rng.random()<q.break_rate:
                k=int(rng.integers(5,len(pp)-5)); width=max(3,int(rng.uniform(.04,.13)*len(pp)))
                mask[k:min(len(pp),k+width)]=False
            sections=[]; start=None
            for i,ok in enumerate(mask):
                if ok and start is None:start=i
                if start is not None and (not ok or i==len(mask)-1):
                    end=i if not ok else i+1
                    if end-start>=3 and np.linalg.norm(pp[end-1]-pp[start])>1.3:sections.append(pp[start:end])
                    start=None
            for part in sections:
                shift=rng.normal(0,q.offset_sigma_m,2)
                noisy=part+shift+rng.normal(0,q.sigma_m*.35,part.shape)
                item={'type':'lane_marking' if kind=='line' else 'road_boundary',
                      'points':np.round(noisy[::max(1,len(noisy)//64)],3).tolist(),
                      'confidence':float(np.clip(1-q.line_miss_rate+rng.normal(0,.04),.35,.99)),
                      'source_id':obj['id']}
                if kind=='line':item.update(color=obj.get('color','white'),line_style=obj.get('style','solid'))
                (lines if kind=='line' else curbs).append(item)
        else:
            if rng.random()<q.box_miss_rate:continue
            # Existing rectangle estimator returns exactly 4 corners or None.
            # Pre-check the asymmetric FOV; cropping uses a conservative lateral limit.
            if not visible(p).any() and not visible(p.mean(axis=0)[None,:]).any():continue
            b=rect_observation(p,local,rng)
            if b is None:continue
            # Keep asymmetric limits: if nearly no corner is on the allowed side, drop.
            if not visible(b).any():continue
            b+=rng.normal(0,q.offset_sigma_m,size=(1,2))
            o={'id':obj['id'],'points':np.round(b,3).tolist(),'confidence':.8}
            (stops if kind=='stop_line' else cross).append(o)
    ranked=sorted([(min(np.linalg.norm(np.asarray(o['points']),axis=1)),kind,o)
                   for kind,items in (('line',lines),('curb',curbs)) for o in items],key=lambda x:x[0])
    if max_lines is not None:ranked=ranked[:max_lines]
    return ([o for _,k,o in ranked if k=='line'],[o for _,k,o in ranked if k=='curb'],stops,cross)

def route_base(cfg,spec,nav,in_lane,out_lane,guide=False,guide_noise=None):
    w=cfg.lane_width_m;L=cfg.arm_length_m;r=cfg.corner_radius_m
    ni=np.array(spec.num_in);no=np.array(spec.num_out)
    j=np.array([max(ni[(k+1)%4],no[(k+1)%4],ni[(k+3)%4],no[(k+3)%4])*w+r for k in range(4)])
    k={'straight':2,'left':3,'right':1}[nav]
    start=arm_xy(0,j[0],(in_lane+.5)*w)
    end=arm_xy(k,j[k],-(out_lane+.5)*w)
    in_dir=-OUT[0];out_dir=OUT[k]
    h=.45*np.linalg.norm(end-start)
    if guide:
        gn=guide_noise or {}
        # Coarse navigation route, built directly from entry/exit lane identities, not GT coordinates.
        shift=float(gn.get('lateral_offset_m',0))
        normal=np.array([-out_dir[1],out_dir[0]])
        end=end+shift*normal
        h*=float(gn.get('handle_scale',.65))
        start=start+float(gn.get('entry_lateral_m',0))*LEFT[0]
    turn=bezier(start,start+in_dir*h,end-out_dir*h,end,n=420)
    before=arm_xy(0,np.linspace(L,j[0],900),(in_lane+.5)*w)
    # Gradually approach the biased entrance when generating a guide.
    if guide:
        wgt=np.sin(np.linspace(0,np.pi/2,len(before)))**4
        before=before+wgt[:,None]*(start-before[-1])
    after=end[None,:]+np.linspace(0,L-j[k],900)[:,None]*out_dir
    p=np.concatenate([before[:-1],turn[:-1],after])
    se=arclength(before)[-1]
    sx=se+arclength(turn)[-1]
    return p,se,sx,k

def match_lane_after(cfg,spec,route,nav,warp,exit_s,exit_arm,out_lane):
    # Compare fixed GT world route to *independently reconstructed* destination centerline.
    w=cfg.lane_width_m; r=cfg.corner_radius_m
    ni=np.array(spec.num_in);no=np.array(spec.num_out)
    k=exit_arm;j=max(ni[(k+1)%4],no[(k+1)%4],ni[(k+3)%4],no[(k+3)%4])*w+r
    st=np.arange(3,min(cfg.arm_length_m-j-3,95),1.)
    expected=deform(arm_xy(k,st+j,-(out_lane+.5)*w),warp)
    # Interpolate GT by its actual path arc length measured from its own exit station.
    samples=at_s(route,exit_s+arclength(expected))
    # Compare geometry by nearest projected station; deformation changes local arc scale.
    dev=np.min(np.linalg.norm(route[None,:,:]-expected[:,None,:],axis=-1),axis=1)
    if np.max(dev)>0.20:raise AssertionError(f'{nav} GT exit misaligned: {dev.max():.3f}m')
    return float(dev.max())

GUIDE_CONVENTION = 'world_fixed_ego_transform_v1'


def sample_world_guide(guide_world, ego, heading, n=81):
    """Crop a fixed world guide and rigidly transform it into the ego frame.

    Preserve V4's nearest-vertex crop and subsequent 1m arc sampling. The first
    point need not be the origin: reanchoring would erase the map's lateral bias.
    """
    g = np.asarray(guide_world, dtype=float)
    lengths = arclength(g)
    start = int(np.argmin(np.linalg.norm(g - ego, axis=1)))
    station = float(lengths[start])
    if station + n - 1 > lengths[-1]:
        raise ValueError('Guide route too short')
    return to_ego(at_s(g, station + np.arange(n)), ego, heading), station


def validate_drivable_route(base, cfg, spec, nav):
    """Check the full canonical GT centerline against all four road strips + rounded fillets."""
    w=cfg.lane_width_m; r=cfg.corner_radius_m
    ni=np.asarray(spec.num_in);no=np.asarray(spec.num_out)
    p=at_s(base,np.arange(0,arclength(base)[-1],.35))
    good=np.zeros(len(p),bool)
    for k in range(4):
        longitudinal=p@OUT[k]; lateral=p@LEFT[k]
        good|=(longitudinal>=-1e-6)&(longitudinal<=cfg.arm_length_m+1e-6)&(lateral>=-no[k]*w-1e-6)&(lateral<=ni[k]*w+1e-6)
    for sx in (-1,1):
        for sy in (-1,1):
            kv=2 if sy>0 else 0;kh=1 if sx>0 else 3
            xw=(no[kv] if sx>0 else ni[kv]) if kv==2 else (ni[kv] if sx>0 else no[kv])
            yw=(ni[kh] if sy>0 else no[kh]) if kh==1 else (no[kh] if sy>0 else ni[kh])
            ex=xw*w;ey=yw*w
            x=sx*p[:,0]; y=sy*p[:,1]
            good|=(x>=0)&(y>=0)&(x<=ex+r)&(y<=ey+r)&((x-ex-r)**2+(y-ey-r)**2>=r*r-1e-5)
    if not np.all(good):
        bad=p[~good][0]
        raise AssertionError(f'{nav} traverses non-drivable area near {bad.tolist()}')
    return True

def build_one(world_seed, cfg, qualities, guide_max_offset=3., guide_handle_range=(.45,1.2),max_attempts=24):
    rng=np.random.default_rng(world_seed)
    spec=parse_spec(world_seed)
    from World.road_warp import densify
    # One common deformation map for all geometry. Unbiased range includes near-regular cases.
    warp=ArmWarp(angle_deg=(0.,*rng.uniform(-16,16,3)),
                 lateral_m=(0.,*rng.uniform(-5,5,3)))
    world_id=f'intersection_{world_seed:010d}'
    world=warp_world(build_world_v4(cfg,spec,world_id),warp)
    world['metadata'].update({'seed':world_seed,'generator':'intersection_v4','num_in':list(spec.num_in),'num_out':list(spec.num_out)})
    routes={}
    for nav in NAVS:
        exit_arm={'straight':2,'left':3,'right':1}[nav]
        # Choose legal existing lanes. Allow all combinations for this synthetic generic junction.
        choices=[(ii,oo) for ii in range(spec.num_in[0]) for oo in range(spec.num_out[exit_arm])]
        rng.shuffle(choices)
        selected=None
        for in_lane,out_lane in choices:
            base,se,sx,k=route_base(cfg,spec,nav,in_lane,out_lane)
            try:validate_drivable_route(base,cfg,spec,nav)
            except AssertionError:continue
            selected=(in_lane,out_lane,base,se,sx,k)
            break
        if selected is None:raise ValueError(f'{nav}: no drivable lane connection; reject world')
        in_lane,out_lane,base,se,sx,k=selected
        gt=deform(base,warp)
        warped_s=arclength(gt);base_s=arclength(base)
        entry=float(np.interp(se,base_s,warped_s));exit_s=float(np.interp(sx,base_s,warped_s))
        match_lane_after(cfg,spec,gt,nav,warp,exit_s,k,out_lane)
        gn={'lateral_offset_m':float(rng.uniform(-guide_max_offset,guide_max_offset)),
            'entry_lateral_m':float(rng.uniform(-.4,.4)),
            'handle_scale':float(rng.uniform(*guide_handle_range))}
        guide_base,_,_,_=route_base(cfg,spec,nav,in_lane,out_lane,True,gn)
        guide=deform(guide_base,warp)
        routes[nav]={'gt_world':gt,'guide_world':guide,'entry_s':entry,'exit_s':exit_s,
                     'entry_lane_index':in_lane,'exit_lane_index':out_lane,'exit_arm':k,'guide_parameters':gn}
    samples=[]
    for nav in NAVS:
        rt=routes[nav]; route=rt['gt_world']
        for pose,s0 in pose_stations(rt['entry_s'],rt['exit_s']):
            ego,t=local_pose(route,s0)
            gt=to_ego(sample_path(route,s0),ego,t)
            # All poses view one fixed world guide; never translate it to the ego origin.
            ref,gs=sample_world_guide(rt['guide_world'],ego,t)
            for quality,q in qualities.items():
                local_rng=np.random.default_rng(np.random.SeedSequence([world_seed,{'straight':0,'left':1,'right':2}[nav],pose_seed_index(pose),list(qualities).index(quality)]))
                ln,cb,st,cw=observe(world,ego,t,q,local_rng)
                row={'id':f'{world_id}_{nav}_{pose}_{quality}','world_id':world_id,
                     'route_id':f'{world_id}_{nav}_in{rt["entry_lane_index"]}_out{rt["exit_lane_index"]}',
                     'ego_pose_id':f'{nav}_{pose}', 'coordinate_system':'ego_at_start_x_forward_y_left_m',
                     'navigation':nav,'gt_path_sample_spacing_m':1.0,
                     'gt_path':np.round(gt,4).tolist(),'reference_line':np.round(ref,4).tolist(),
                     'lane_markings':ln,'road_boundaries':cb,'crosswalks':cw,'stop_lines':st,
                     'evaluation':{'support_heuristic':observation_support(gt,ln,cb,cfg)},
                     'metadata':{'generator_version':'intersection_v4','quality':quality,'pose_name':pose,
                                 'num_in':list(spec.num_in),'num_out':list(spec.num_out),
                                 'ego_world_xy':np.round(ego,6).tolist(),
                                 'heading_rad':float(np.arctan2(t[1],t[0])),'ego_route_s_m':float(s0),
                                 'guide_coordinate_convention':GUIDE_CONVENTION,'guide_start_s_m':gs,
                                 'vcs_fov_m':{'forward':q.forward_m,'rear':q.rear_m,'left':q.left_m,'right':q.right_m}}}
                samples.append(row)
    return world,samples,routes

def pose_stations(entry_s,exit_s):
    """Arc-length positions on GT; inside_13 exists only strictly before exit."""
    stations=(entry_s-30,entry_s-20,entry_s-10,entry_s,entry_s+8,entry_s+13,exit_s+5)
    return [(pose,float(s)) for pose,s in zip(V4_POSES,stations)
            if pose!='inside_13' or s<exit_s-1e-6]


def verify(rows):
    seen={}
    for r in rows:
        gt=np.asarray(r['gt_path']);ref=np.asarray(r['reference_line'])
        if gt.shape!=(81,2) or ref.shape!=(81,2):raise AssertionError('Bad path shape')
        if np.max(np.abs(np.linalg.norm(np.diff(gt,axis=0),axis=1)-1))>.03:raise AssertionError('Bad spacing')
        if len(r['lane_markings'])+len(r['road_boundaries'])>32:raise AssertionError('Too many lines')
        if any(np.asarray(x['points']).shape!=(4,2) for x in r['crosswalks']+r['stop_lines']):raise AssertionError('Bad box')
        key=(r['route_id'],r['ego_pose_id'])
        if not np.isfinite(gt).all() or not np.isfinite(ref).all():raise AssertionError('Non-finite path')
        if key in seen:
            if not np.array_equal(gt,seen[key][0]):raise AssertionError('GT changed with perception')
            if not np.array_equal(ref,seen[key][1]):raise AssertionError('Guide changed with perception')
        seen[key]=(gt,ref)
    return {'samples':len(rows),'unique_pose_routes':len(seen)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261023);p.add_argument('--output-dir',default='outputs/intersection_v4')
    p.add_argument('--perception-config',default=None);p.add_argument('--guide-max-offset',type=float,default=3.)
    p.add_argument('--max-attempts-per-world',type=int,default=50)
    a=p.parse_args();q=read_qualities(a.perception_config)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'worlds').mkdir(exist_ok=True);(out/'routes').mkdir(exist_ok=True)
    if a.worlds < 1 or a.max_attempts_per_world < 1:
        p.error('--worlds and --max-attempts-per-world must be positive')
    all_rows=[]; reports=[]; rejected=[]; accepted_seeds=[]
    # Every requested world owns its own deterministic seed range.
    # Reject only the explicit geometric infeasibility raised by build_one.
    for i in range(a.worlds):
        for attempt in range(a.max_attempts_per_world):
            sd=a.seed+7919*i+attempt
            try:
                world,rows,routes=build_one(sd,IntersectionConfig(),q,a.guide_max_offset)
            except ValueError as exc:
                reason=str(exc)
                if not (reason.endswith('no drivable lane connection; reject world')):
                    raise
                rejected.append({'slot':i,'seed':sd,'reason':reason})
                print(f'[reject] seed={sd}: {reason}',flush=True)
                continue
            # Never hide validation errors: investigate invalid accepted geometry.
            report=verify(rows)
            reports.append(report);all_rows.extend(rows);accepted_seeds.append(sd)
            if i==0:
                (out/'world.json').write_text(json.dumps(world,indent=2))
                plot(world,rows,out/'preview.png',pose_order=V4_POSES)
            (out/'worlds'/f'{world["world_id"]}.json').write_text(json.dumps(world))
            (out/'routes'/f'{world["world_id"]}.json').write_text(json.dumps({k:{**{field:value for field,value in v.items() if field not in ('gt_world','guide_world')},
                  'gt_world':np.round(v['gt_world'],6).tolist(),'guide_world':np.round(v['guide_world'],6).tolist()} for k,v in routes.items()}))
            print(f'[accept] {i+1}/{a.worlds}: seed={sd}, rows={len(rows)}',flush=True)
            break
        else:
            (out/'rejected_worlds.json').write_text(json.dumps(rejected,indent=2))
            raise RuntimeError(f'Could not generate valid world for slot {i} after {a.max_attempts_per_world} attempts. Inspect rejected_worlds.json')
    (out/'rejected_worlds.json').write_text(json.dumps(rejected,indent=2))
    (out/'accepted_world_seeds.json').write_text(json.dumps(accepted_seeds,indent=2))
    with (out/'samples.jsonl').open('w') as f:
        for x in all_rows:f.write(json.dumps(x)+'\n')
    (out/'config.json').write_text(json.dumps({'seed':a.seed,'worlds':a.worlds,'qualities':{k:asdict(v) for k,v in q.items()},'guide_max_offset_m':a.guide_max_offset,'guide_coordinate_convention':GUIDE_CONVENTION,'poses':list(V4_POSES),'inside_13_policy':'entry_s + 13 m along GT; omit when at or beyond exit_s','accepted_world_seeds':accepted_seeds,'rejected_count':len(rejected)},indent=2))
    print('saved',out,'samples',len(all_rows),'reports',reports[:2])
if __name__=='__main__':main()
