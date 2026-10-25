"""V4 multi-lane four-arm intersection truth. Geometry is deterministic given spec.
Coordinate conventions: global east/north; local arm s runs outward, l to its left.
Right-hand traffic: incoming lanes on positive local lateral, outgoing on negative.
"""
from dataclasses import dataclass
import numpy as np
from World.intersection_generator import IntersectionConfig

ARM_NAMES=('south','east','north','west')
OUT=np.array([[0.,-1.],[1.,0.],[0.,1.],[-1.,0.]])
LEFT=np.stack([-OUT[:,1],OUT[:,0]],axis=1)

@dataclass(frozen=True)
class WorldSpec:
    num_in: tuple=(1,1,1,1)
    num_out: tuple=(1,1,1,1)
    # Geometry deformation is stored separately as ArmWarp.

def parse_spec(seed, min_lanes=1, max_lanes=3):
    rng=np.random.default_rng(seed)
    return WorldSpec(tuple(int(x) for x in rng.integers(min_lanes,max_lanes+1,size=4)),
                     tuple(int(x) for x in rng.integers(min_lanes,max_lanes+1,size=4)))

def arm_xy(k,s,l):
    return np.asarray(s)[...,None]*OUT[k]+np.asarray(l)[...,None]*LEFT[k]

def build_world_v4(cfg=IntersectionConfig(),spec=WorldSpec(),world_id='intersection_0001'):
    w=float(cfg.lane_width_m);r=float(cfg.corner_radius_m);L=float(cfg.arm_length_m)
    ni=np.asarray(spec.num_in,int);no=np.asarray(spec.num_out,int)
    if ni.shape!=(4,) or no.shape!=(4,) or np.any(ni<1) or np.any(no<1) or np.any(ni>5) or np.any(no>5):
        raise ValueError('Each arm must have 1..5 inbound and outbound lanes')
    if L<110+max(ni.max(),no.max())*w+r:raise ValueError('Arms too short')
    # Each arm near-junction stop station depends on the width of the crossing roadway.
    # Opposite arms are not required to have identical widths.
    half={k:(ni[k]*w,no[k]*w) for k in range(4)}
    # For a given arm use max transverse half-width of other road + corner clearance.
    entry_s=np.array([max(ni[(k+1)%4],no[(k+1)%4],ni[(k+3)%4],no[(k+3)%4])*w+r for k in range(4)],float)
    elements=[]
    def add(kind, ident, pts, **kwargs):
        elements.append(dict(id=ident,type=kind,points=np.round(np.asarray(pts,float),5).tolist(),**kwargs))
    for k,name in enumerate(ARM_NAMES):
        j=entry_s[k]
        # Both outer road borders are curbs; use two-point centerline elements, densified before deformation.
        for side,l in [('incoming',ni[k]*w),('outgoing',-no[k]*w)]:
            # Each flank reaches its own adjacent corner tangency (widths may differ).
            neighbor=(k+1)%4 if side=='incoming' else (k+3)%4
            toward=float(np.dot(OUT[k],LEFT[neighbor]))
            neighbor_half=ni[neighbor] if toward>0 else no[neighbor]
            curb_end=neighbor_half*w+r
            add('curb',f'{name}_curb_{side}',arm_xy(k,[L,curb_end],l))
        # All lane division lines, including the central opposing-flow divider.
        for ix in range(-no[k]+1,ni[k]):
            add('line',f'{name}_divider_{ix:+d}',arm_xy(k,[L,j],ix*w),
                color='yellow' if ix==0 else 'white',style='solid' if ix==0 else 'dashed')
        # Incoming-only stop line; full lane group but NOT opposing carriageway.
        stop_s=j+cfg.stop_setback_m;th=cfg.stop_line_thickness_m/2
        add('stop_line',f'{name}_stop',
            [arm_xy(k,stop_s-th,0),arm_xy(k,stop_s-th,ni[k]*w),
             arm_xy(k,stop_s+th,ni[k]*w),arm_xy(k,stop_s+th,0)],
            controls=f'{name}_incoming')
        cross_s=j+cfg.crosswalk_setback_m
        a=cross_s;b=cross_s+cfg.crosswalk_depth_m
        add('crosswalk',f'{name}_crosswalk',
            [arm_xy(k,a,-no[k]*w),arm_xy(k,a,ni[k]*w),
             arm_xy(k,b,ni[k]*w),arm_xy(k,b,-no[k]*w)])
    # Four rounded corners. Each connects its adjacent arm's outer curb in one quadrant.
    # Calculate the two independently sized half-widths rather than assuming symmetry.
    for sx in (-1,1):
        for sy in (-1,1):
            # x side boundary on vertical branch: north if sy=+, south otherwise.
            kv=2 if sy==1 else 0
            width_x=(no[kv] if sx==1 else ni[kv]) if kv==2 else (ni[kv] if sx==1 else no[kv])
            # y side boundary on horizontal branch: east if sx=+, west otherwise.
            kh=1 if sx==1 else 3
            width_y=(ni[kh] if sy==1 else no[kh]) if kh==1 else (no[kh] if sy==1 else ni[kh])
            ex=width_x*w;ey=width_y*w
            t=np.linspace(0,np.pi/2,42)
            q=np.stack([ex+r-r*np.sin(t),ey+r-r*np.cos(t)],axis=1)*[sx,sy]
            add('curb',f'corner_{sx:+d}_{sy:+d}',q)
    roads=[{'id':f'road_{name}','axis':np.round(arm_xy(k,[L,entry_s[k]],0),5).tolist(),
            'num_in':int(ni[k]),'num_out':int(no[k]),'lane_width_m':w} for k,name in enumerate(ARM_NAMES)]
    # The drivable surface is implicit in curbs; exact polygon gets constructed in validator.
    return {'world_id':world_id,'schema_version':'4.0',
            'metadata':{'generator':'parametric_intersection_v4','num_in':ni.tolist(),
                        'num_out':no.tolist(),'lane_width_m':w,'corner_radius_m':r,
                        'arm_length_m':L,'entry_distances_m':entry_s.tolist()},
            'physical':{'elements':elements},
            'semantic':{'roads':roads,'lanes':[], 'intersections':[{'id':'junction_0','road_ids':[x['id'] for x in roads]}],
                        'lane_connections':[]}}
