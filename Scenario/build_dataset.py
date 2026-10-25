"""Intersection World -> navigation/GT/guide/perception samples (no training)."""
import argparse, json, math
from pathlib import Path
import numpy as np
from World.intersection_generator import IntersectionConfig,build_world,validate_curbs
from World.road_warp import ArmWarp, deform, warp_world, densify

NAVS=('straight','left','right')
POSES=('approach_30','approach_20','approach_10','entry','inside_8','exit_5')
# Chronological V4 display order is separate from stable perception RNG IDs.
V4_POSES=(*POSES[:-1],'inside_13',POSES[-1])
POSE_SEED_ORDER=(*POSES,'inside_13')

def pose_seed_index(pose):
    return POSE_SEED_ORDER.index(pose)

# VCS: x=forward, y=left. Four unequal extents, independent of world XY.
QUALITY={
 'clean':  dict(forward_m=85., rear_m=18., lateral_m=22., sigma_m=.018, miss=.02, angle_sigma_deg=.35),
 'medium': dict(forward_m=65., rear_m=14., lateral_m=15., sigma_m=.07,  miss=.16, angle_sigma_deg=1.5),
 'hard':   dict(forward_m=45., rear_m=10., lateral_m=9.,  sigma_m=.18,  miss=.36, angle_sigma_deg=3.5),
}


def unit(a):
    return a/max(np.linalg.norm(a),1e-12)
def bezier(a,b,c,d,n=200):
    t=np.linspace(0,1,n)[:,None]
    return (1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d

def arclength(points):
    return np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
def at_s(points,s):
    d=arclength(points)
    return np.column_stack([np.interp(np.asarray(s),d,points[:,i]) for i in (0,1)])

def route_geometry(cfg, nav, guide=False):
    w=cfg.lane_width_m; j=w+cfg.corner_radius_m; L=cfg.arm_length_m
    # Northbound incoming on east half of south road. Right-hand traffic.
    start=np.array([w/2,-j]); in_t=np.array([0.,1.])
    if nav=='straight': end=np.array([w/2,j]); out_t=np.array([0.,1.])
    elif nav=='left': end=np.array([-j,w/2]);out_t=np.array([-1.,0.])
    else: end=np.array([j,-w/2]);out_t=np.array([1.,0.])
    # Guide is independently built from routing anchors + coarse geometric approximation;
    # it never reads GT or a GT-derived array.
    if guide:
        offset={'straight':0.35,'left':0.7,'right':-0.55}[nav]
        end=end+np.array([-out_t[1],out_t[0]])*offset
        handle=.32*np.linalg.norm(end-start)
    else: handle=.45*np.linalg.norm(end-start)
    middle=bezier(start,start+in_t*handle,end-out_t*handle,end)
    before=np.stack([np.full(320,w/2),np.linspace(-L,-j,320)],axis=1)
    after=end[None,:]+np.linspace(0,L-j,450)[:,None]*out_t
    path=np.concatenate([before[:-1],middle[:-1],after])
    entry_s=L-j
    exit_s=entry_s+arclength(middle)[-1]
    return path,entry_s,exit_s

def local_pose(path,s):
    d=arclength(path); p=at_s(path,[s])[0]
    p0=at_s(path,[max(0,s-.15)])[0];p1=at_s(path,[min(d[-1],s+.15)])[0]
    t=unit(p1-p0);return p,t

def to_ego(x,p,t):
    delta=np.asarray(x)-p
    return np.stack([delta[...,0]*t[0]+delta[...,1]*t[1],-delta[...,0]*t[1]+delta[...,1]*t[0]],axis=-1)

def sample_path(route,s0,n=81):
    if arclength(route)[-1]<s0+n-1: raise ValueError('Route shorter than 80m from ego')
    return at_s(route,s0+np.arange(n,dtype=float))

def make_guide(cfg,nav,s0,ego,t):
    route,_,_=route_geometry(cfg,nav,guide=True)
    # s0 is measured along the incoming true lane. Inside/exit poses use map projection
    # on the independent coarse route, not GT points or GT curve.
    d=np.linalg.norm(route-ego,axis=1); idx=int(np.argmin(d))
    lengths=arclength(route); offset=float(lengths[idx])
    if lengths[-1]<offset+80: raise ValueError('Guide too short')
    ref=to_ego(at_s(route,offset+np.arange(81)),ego,t)
    # Anchor in ego coordinates and ensure first point is exact origin.
    return ref-ref[:1]

def in_fov(p, q):
    """Rectangular asymmetric vehicle-coordinate observation envelope."""
    p=np.asarray(p)
    return ((p[...,0]>=-q['rear_m']) & (p[...,0]<=q['forward_m']) &
            (np.abs(p[...,1])<=q['lateral_m']))


def visible_parts(points, q, rng):
    p=np.asarray(points,float)
    chunks=[];cur=[]
    for a,b in zip(p[:-1],p[1:]):
        n=max(2,int(np.linalg.norm(b-a)/.35)+1)
        for v in np.linspace(a,b,n,endpoint=False):
            if in_fov(v,q) and rng.random()>q['miss']*.20:
                cur.append(v)
            else:
                if len(cur)>=3 and np.linalg.norm(cur[0]-cur[-1])>1.3:chunks.append(np.asarray(cur))
                cur=[]
    if len(cur)>=3 and np.linalg.norm(cur[0]-cur[-1])>1.3:chunks.append(np.asarray(cur))
    out=[]
    for part in chunks:
        drift=rng.normal(0,q['sigma_m'],size=2)
        out.append(part+drift+rng.normal(0,q['sigma_m']*.4,size=part.shape))
    return out


def rect_observation(corners,q,rng):
    """Estimate one smaller oriented rectangle when only part is in VCS FOV.

    Never polygon-clip a box: returned geometry always has 4 vertices.
    Reconstruct from visible samples in the true rectangle's local axes.
    """
    p=np.asarray(corners,float)
    assert p.shape==(4,2)
    center=p.mean(axis=0)
    a=p[1]-p[0];b=p[3]-p[0]
    len_a=np.linalg.norm(a);len_b=np.linalg.norm(b)
    if min(len_a,len_b)<1e-6:return None
    ua=a/len_a;ub=b/len_b
    # Rectangles are defined with perpendicular axes.
    if abs(np.dot(ua,ub))>1e-3:raise ValueError('World box must be rectangular')
    u=np.linspace(0,1,17)
    grid=p[0][None,None,:]+u[:,None,None]*a+u[None,:,None]*b
    visible=in_fov(grid,q)
    frac=float(visible.mean())
    if frac<.18:return None
    if not visible.any():return None
    # Work in rectangle-local coordinates to preserve rotation and right angles.
    vv=grid[visible]-center
    aa=vv@ua;bb=vv@ub
    a0,a1=float(aa.min()),float(aa.max());b0,b1=float(bb.min()),float(bb.max())
    # Enough span in both dimensions to be a meaningful detection.
    if (a1-a0)<max(.12,len_a*.12) or (b1-b0)<max(.10,len_b*.12):return None
    a_center=(a0+a1)/2;b_center=(b0+b1)/2
    c=center+a_center*ua+b_center*ub
    # Correlated position/heading error, size error, NEVER independent vertex jitter.
    c+=rng.normal(0,q['sigma_m']*1.8,size=2)
    ang=np.deg2rad(rng.normal(0,q['angle_sigma_deg']))
    rot=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    ua=rot@ua; ub=rot@ub
    sa=(a1-a0)*np.clip(1+rng.normal(0,q['sigma_m']*.3),.75,1.25)
    sb=(b1-b0)*np.clip(1+rng.normal(0,q['sigma_m']*.3),.75,1.25)
    return np.stack([c-sa/2*ua-sb/2*ub,c+sa/2*ua-sb/2*ub,
                     c+sa/2*ua+sb/2*ub,c-sa/2*ua+sb/2*ub])


def perceive(world,ego,t,quality,rng):
    q=QUALITY[quality];lines=[];curbs=[];cross=[];stops=[]
    for obj in world['physical']['elements']:
        kind=obj['type'];p=to_ego(np.asarray(obj['points'],float),ego,t)
        if kind in ('line','curb'):
            if rng.random()<q['miss']:continue
            for part in visible_parts(p,q,rng):
                item={'type':'lane_marking' if kind=='line' else 'road_boundary',
                      'points':np.round(part[::max(1,len(part)//64)],3).tolist(),
                      'confidence':round(float(rng.uniform(.8,.99) if quality=='clean' else rng.uniform(.52,.94)),3),
                      'source_id':obj['id']}
                if kind=='line':item.update(color=obj.get('color','white'),line_style=obj.get('style','solid'))
                (lines if kind=='line' else curbs).append(item)
        elif kind in ('stop_line','crosswalk'):
            if rng.random()<q['miss']:continue
            distorted=rect_observation(p,q,rng)
            if distorted is None:continue
            item={'id':obj['id'],'points':np.round(distorted,3).tolist(),
                  'confidence':round(float(rng.uniform(.78,.98) if quality=='clean' else rng.uniform(.50,.90)),3)}
            (stops if kind=='stop_line' else cross).append(item)
    def dist(obj):return min(np.linalg.norm(np.asarray(obj['points']),axis=1))
    lines.sort(key=dist);curbs.sort(key=dist)
    return lines,curbs,stops,cross

def observation_support(gt, lines, curbs, cfg):
    """Evaluation-only evidence labels; never passed to model inputs.
    A 4m proximity heuristic is deliberately not claimed as strict two-sided visibility.
    """
    observed=lines+curbs
    p=np.concatenate([np.asarray(e['points']) for e in observed],axis=0) if observed else np.empty((0,2))
    mask=[]
    for g in gt:
        near=(len(p)>0 and float(np.min(np.linalg.norm(p-g,axis=1)))<4.0)
        mask.append('direct_geometry_nearby' if near else 'unobserved_or_extrapolated')
    return mask

def validate_exit_lane_alignment(cfg, warp, routes, sample_every_m=0.5):
    """Check GT on the selected outbound lane using world-transformed boundaries.

    The two flanks and GT are all transformed from consistent canonical
    stations. This detects a GT route selecting the wrong outgoing lane.
    """
    w=cfg.lane_width_m
    j=w+cfg.corner_radius_m
    L=cfg.arm_length_m
    for nav in NAVS:
        end={'straight':np.array([w/2,j]),'left':np.array([-j,w/2]),
             'right':np.array([j,-w/2])}[nav]
        tangent={'straight':np.array([0.,1.]),'left':np.array([-1.,0.]),
                 'right':np.array([1.,0.])}[nav]
        normal=np.array([-tangent[1],tangent[0]])
        z=np.arange(3.,min(L-j-2.,100.),sample_every_m)
        central=end[None,:]+z[:,None]*tangent
        center=deform(central,warp)
        left=deform(central+w/2*normal,warp)
        right=deform(central-w/2*normal,warp)
        across=left-right
        fraction=np.sum((center-right)*across,axis=1)/np.maximum(np.sum(across*across,axis=1),1e-8)
        if not np.all((fraction>0.03)&(fraction<0.97)):
            raise AssertionError(f'{nav}: transformed lane center escapes outbound lane')
        # The GT is evaluated at world-route stations, never defined by these flanks.
        gt=routes[nav]['gt_world']
        dist=np.min(np.linalg.norm(gt[None,:,:]-center[:,None,:],axis=2),axis=1)
        if np.max(dist)>0.65:
            raise AssertionError(f'{nav}: fixed GT differs from outbound lane center; max={np.max(dist):.3f}m')
    return True


def build_fixed_routes(cfg, warp):
    """Create one absolute world-space GT/guide route per navigation, shared by all poses."""
    routes={}
    for nav in NAVS:
        base,entry,exit_=route_geometry(cfg,nav,guide=False)
        warped=deform(base,warp)
        original_s=arclength(base); warped_s=arclength(warped)
        entry_world_s=float(np.interp(entry,original_s,warped_s))
        exit_world_s=float(np.interp(exit_,original_s,warped_s))
        guide_base,_,_=route_geometry(cfg,nav,guide=True)
        routes[nav]={'gt_world':warped,'guide_world':deform(guide_base,warp),
                     'entry_s':entry_world_s,'exit_s':exit_world_s}
    return routes

def guide_from_world(guide_route,ego,t):
    lengths=arclength(guide_route)
    idx=int(np.argmin(np.linalg.norm(guide_route-ego,axis=1)))
    if lengths[-1]<lengths[idx]+80:raise ValueError('Guide route too short')
    ref=to_ego(at_s(guide_route,lengths[idx]+np.arange(81)),ego,t)
    return ref-ref[:1]

def make_dataset(cfg=IntersectionConfig(),seed=42,warp=ArmWarp(),world_id='intersection_0001'):
    base=build_world(cfg,world_id=world_id)
    validate_curbs(base,cfg);validate_drivability(cfg)
    world=warp_world(base,warp)
    routes=build_fixed_routes(cfg,warp)
    validate_exit_lane_alignment(cfg,warp,routes)
    rng=np.random.default_rng(seed)
    samples=[]
    for nav in NAVS:
        rt=routes[nav]; route=rt['gt_world'];entry_s=rt['entry_s'];exit_s=rt['exit_s']
        stations=[entry_s-30,entry_s-20,entry_s-10,entry_s,entry_s+8,exit_s+5]
        for pose_name,s0 in zip(POSES,stations):
            ego,t=local_pose(route,s0)
            gt_world=sample_path(route,s0)
            gt=to_ego(gt_world,ego,t)
            guide=guide_from_world(rt['guide_world'],ego,t)
            for quality in QUALITY:
                ln,cb,st,cw=perceive(world,ego,t,quality,rng)
                if len(ln)+len(cb)>32:
                    tagged=sorted([(min(np.linalg.norm(np.asarray(e['points']),axis=1)),k,e)
                                   for k,ls in [('ln',ln),('cb',cb)] for e in ls],key=lambda x:x[0])[:32]
                    ln=[e for _,k,e in tagged if k=='ln'];cb=[e for _,k,e in tagged if k=='cb']
                row={'id':f'{world_id}_{nav}_{pose_name}_{quality}','world_id':world_id,
                     'route_id':f'{world_id}_{nav}_south_entry','ego_pose_id':f'{nav}_{pose_name}',
                     'coordinate_system':'ego_at_start_x_forward_y_left_m',
                     'navigation':nav,'gt_path_sample_spacing_m':1.0,
                     'gt_path':np.round(gt,4).tolist(),'reference_line':np.round(guide,4).tolist(),
                     'lane_markings':ln,'road_boundaries':cb,'crosswalks':cw,'stop_lines':st,
                     'evaluation':{'support_heuristic':observation_support(gt,ln,cb,cfg)},
                     'metadata':{'generator_version':'intersection_parametric_v3','quality':quality,
                                 'pose_name':pose_name,'vcs_fov_m':{k:qv for k,qv in QUALITY[quality].items() if k in ('forward_m','rear_m','lateral_m')},'world_seed':seed,
                                 'ego_world_xy':np.round(ego,6).tolist(),
                                 'heading_rad':float(np.arctan2(t[1],t[0])),
                                 'ego_route_s_m':float(s0),
                                 'true_connection_length_m':float(exit_s-entry_s)}}
                samples.append(row)
    return world,samples,routes

def validate_drivability(cfg):
    """Dense station checks against two crossing road strips and rounded corners.
    Polygon union includes road strips and junction corner fillets.
    """
    w=cfg.lane_width_m;r=cfg.corner_radius_m;j=w+r
    for nav in NAVS:
        path,_,_=route_geometry(cfg,nav)
        points=at_s(path,np.arange(0,arclength(path)[-1],.25))
        x=np.abs(points[:,0]);y=np.abs(points[:,1])
        in_strips=(x<=w+1e-5)|(y<=w+1e-5)
        # Concave rounded road polygon at the four interior corners.
        in_corner=(x<=j)&(y<=j)&(((x-j)**2+(y-j)**2)>=r**2-1e-5)
        if not np.all(in_strips|in_corner):
            bad=points[~(in_strips|in_corner)][0]
            raise AssertionError(f'{nav} leaves drivable surface near {bad}')
    return True

def validate(samples):
    groups={}
    for r in samples:
        gt=np.asarray(r['gt_path']);ref=np.asarray(r['reference_line']);step=np.linalg.norm(np.diff(gt,axis=0),axis=1)
        assert gt.shape==(81,2) and ref.shape==(81,2)
        assert np.linalg.norm(gt[0])<.002 and np.max(abs(step-1))<.03
        assert all(len(x['points'])==4 for x in r['crosswalks'])
        assert all(len(x['points'])==4 for x in r['stop_lines'])
        assert len(r['lane_markings'])+len(r['road_boundaries'])<=32
        key=(r['world_id'],r['route_id'],r['ego_pose_id'])
        if key in groups:assert np.array_equal(gt,groups[key])
        groups[key]=gt
    return {'samples':len(samples),'groups':len(groups),'navs':len(NAVS)}

def plot(world,samples,out,pose_order=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    poses=tuple(pose_order) if pose_order is not None else POSES
    fig,axs=plt.subplots(3,len(poses),figsize=(4*len(poses),12),layout='constrained',squeeze=False)
    for i,nav in enumerate(NAVS):
      for j,pose in enumerate(poses):
        ax=axs[i,j]
        candidates=[x for x in samples if x['navigation']==nav and x['metadata']['pose_name']==pose]
        if not candidates:
            ax.set_title(nav+' / '+pose+' (not generated)',fontsize=10);ax.axis('off');continue
        r=next((x for x in candidates if x['metadata']['quality']=='medium'),candidates[0])
        ego=np.asarray(r['metadata']['ego_world_xy']);a=r['metadata']['heading_rad'];t=np.array([np.cos(a),np.sin(a)])
        for e in world['physical']['elements']:
            pts=to_ego(np.asarray(e['points']),ego,t)
            if e['type']=='curb':ax.plot(pts[:,0],pts[:,1],color='#8c9ba8',lw=1,alpha=.65)
            if e['type']=='line':ax.plot(pts[:,0],pts[:,1],color='#dcb45d',lw=.7,alpha=.7)
            if e['type']=='crosswalk':ax.add_patch(Polygon(pts,fc='#789a9b',ec='#43747c',alpha=.18))
            if e['type']=='stop_line':ax.add_patch(Polygon(pts,fc='#d94e4e',ec='#c13a3a',alpha=.8,zorder=4))
        for e in r['lane_markings']+r['road_boundaries']:
            pts=np.asarray(e['points']);ax.plot(pts[:,0],pts[:,1],color='#8f63bf',lw=1.3)
        for cw in r['crosswalks']:
            ax.add_patch(Polygon(cw['points'],fill=False,edgecolor='#8f63bf',lw=1.6,zorder=5))
        for st in r['stop_lines']:
            ax.add_patch(Polygon(st['points'],fill=False,edgecolor='#8f63bf',lw=2,zorder=5))
        gt=np.asarray(r['gt_path']);ref=np.asarray(r['reference_line'])
        ax.plot(gt[:,0],gt[:,1],color='#1c9c62',lw=2.1,label='GT')
        ax.plot(ref[:,0],ref[:,1],color='#e29b30',lw=1.5,ls='--',label='Guide')
        ax.scatter(0,0,c='black',s=15,zorder=5)
        ax.set_title(nav+' / '+pose,fontsize=10);ax.set_xlim(-24,90);ax.set_ylim(-70,70);ax.set_aspect('equal');ax.grid(alpha=.12)
    axs[0,0].legend(fontsize=8);fig.savefig(out,dpi=120);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='outputs/intersection_v2');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--worlds',type=int,default=1)
    args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    all_samples=[]; summaries=[]
    for i in range(args.worlds):
        rng=np.random.default_rng(args.seed+7919*i)
        warp=ArmWarp(angle_deg=(0.,*rng.uniform(-14,14,3)),
                     lateral_m=(0.,*rng.uniform(-3,3,3)))
        world_id=f'intersection_{i:05d}'
        world,samples,routes=make_dataset(seed=args.seed+1009*i,warp=warp,world_id=world_id)
        report=validate(samples)
        if i==0:
            (out/'world.json').write_text(json.dumps(world,ensure_ascii=False,indent=2))
            plot(world,samples,out/'preview.png')
        (out/'worlds').mkdir(exist_ok=True)
        (out/'routes').mkdir(exist_ok=True)
        (out/'worlds'/f'{world_id}.json').write_text(json.dumps(world,ensure_ascii=False))
        (out/'routes'/f'{world_id}.json').write_text(json.dumps({nav:{'gt_world':np.round(v['gt_world'],6).tolist(),
            'guide_world':np.round(v['guide_world'],6).tolist(),'entry_s':v['entry_s'],'exit_s':v['exit_s']}
            for nav,v in routes.items()}))
        all_samples.extend(samples);summaries.append(report)
    with (out/'samples.jsonl').open('w') as f:
        for row in all_samples:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('Saved',out,'worlds:',args.worlds,'samples:',len(all_samples),'first validation:',summaries[0])
if __name__=='__main__':main()
