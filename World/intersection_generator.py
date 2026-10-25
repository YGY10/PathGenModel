"""Parametric four-way intersection truth. Local XY: east/right, north/up.
No GT path or perception is used to construct physical geometry.
"""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class IntersectionConfig:
    arm_length_m: float = 180.0
    lane_width_m: float = 3.5
    corner_radius_m: float = 9.0
    crosswalk_depth_m: float = 3.0
    crosswalk_setback_m: float = 3.0
    stop_setback_m: float = 9.0
    curve_samples: int = 28
    stop_line_thickness_m: float = 0.4


def build_world(cfg=IntersectionConfig(), world_id='intersection_0001'):
    w=float(cfg.lane_width_m); r=float(cfg.corner_radius_m); e=w+r; L=float(cfg.arm_length_m)
    if not (L>e+100 and w>2.5 and r>2 and cfg.stop_setback_m>cfg.crosswalk_setback_m+cfg.crosswalk_depth_m):
        raise ValueError('Invalid intersection geometry configuration')
    elements=[]
    def add(kind, id, points, **kw):
        elements.append(dict(id=id,type=kind,points=np.round(points,5).tolist(),**kw))
    # Rotation of south arm geometry to east, north, west arms.
    def rot(pts,k):
        a=k*np.pi/2; R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        return np.asarray(pts)@R.T
    for k, name in enumerate(['south','east','north','west']):
        # South arm: spans y=[-L,-e], two-way road with yellow divider.
        add('line',f'{name}_divider',rot([[0,-L],[0,-e]],k), color='yellow',style='solid')
        for side,x in [('left',-w),('right',w)]:
            add('curb',f'{name}_curb_{side}',rot([[x,-L],[x,-e]],k))
        # Stop lines on incoming half only: south incoming x>0.
        ys=-(e+cfg.stop_setback_m)
        h=cfg.stop_line_thickness_m/2
        add('stop_line',f'{name}_stop',rot([[0,ys-h],[w,ys-h],[w,ys+h],[0,ys+h]],k), controls=f'{name}_incoming')
        y0=-(e+cfg.crosswalk_setback_m+cfg.crosswalk_depth_m)
        y1=-(e+cfg.crosswalk_setback_m)
        add('crosswalk',f'{name}_crosswalk',rot([[-w,y0],[w,y0],[w,y1],[-w,y1]],k))
    # Four concave corner curb arcs, tangent to both adjacent arm boundaries.
    for sx in (-1,1):
        for sy in (-1,1):
            t=np.linspace(0,np.pi/2,cfg.curve_samples)
            # First-quadrant arc from (w+r,w) to (w,w+r), concave outward corner.
            # Center (w+r,w+r), radius r: lower->left quarter.
            positive=np.stack([w+r-r*np.sin(t),w+r-r*np.cos(t)],axis=1)
            pts=positive*np.array([sx,sy])
            add('curb',f'corner_{sx:+d}_{sy:+d}',pts)
    roads=[]
    for k,name in enumerate(['south','east','north','west']):
        roads.append({'id':f'road_{name}','axis':rot([[0,-L],[0,-e]],k).tolist(), 'width_m':2*w,'lane_ids':[]})
    drivable=[ [[-w,-L],[w,-L],[w,L],[-w,L]], [[-L,-w],[L,-w],[L,w],[-L,w]] ]
    return {'drivable_areas':drivable,'schema_version':'1.1','world_id':world_id,
            'metadata':{'generator':'parametric_intersection_v2','lane_width_m':w,'corner_radius_m':r,
                        'arm_length_m':L,'junction_station_m':e},
            'physical':{'elements':elements},
            'semantic':{'roads':roads,'lanes':[],'intersections':[{'id':'junction_0','center':[0,0],
                'half_size_m':e,'road_ids':[x['id'] for x in roads]}], 'lane_connections':[]}}


def validate_curbs(world, cfg):
    """Curb arc endpoints meet straight curbs, with matching G1 tangent."""
    w=cfg.lane_width_m; e=w+cfg.corner_radius_m
    es={x['id']:np.asarray(x['points'],float) for x in world['physical']['elements']}
    for sx in (-1,1):
        for sy in (-1,1):
            p=es[f'corner_{sx:+d}_{sy:+d}']
            assert np.allclose(p[0],[sx*e,sy*w],atol=1e-3)
            assert np.allclose(p[-1],[sx*w,sy*e],atol=1e-3)
            a=p[1]-p[0]; b=p[-1]-p[-2]
            assert abs(a[1])<abs(a[0])*0.1 and abs(b[0])<abs(b[1])*0.1
    return True
