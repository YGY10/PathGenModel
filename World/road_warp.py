"""Smooth arm-specific geometric displacement; applied consistently to physical and route geometry."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ArmWarp:
    # South, East, North, West: far-field heading and lateral-offset errors.
    angle_deg: tuple = (0.0, 8.0, -9.0, -6.0)
    lateral_m: tuple = (0.0, 3.0, -4.0, 2.5)
    start_m: float = 13.0
    full_m: float = 65.0

def deform(points, warp=ArmWarp()):
    """One common continuous deformation F: R^2 -> R^2 (degrees and meters).
    Near junction F is identity; each distant arm gets an independent heading and offset.
    Smooth blending means no physical geometry is stitched across inconsistent frames.
    """
    p=np.asarray(points,dtype=float)
    flat=p.reshape(-1,2)
    x,y=flat[:,0],flat[:,1]
    radius=np.hypot(x,y)
    # Directional smooth gates; no abrupt choice between adjacent arms.
    directional=np.maximum(np.stack([-y,x,y,-x],axis=1),0.0)**4
    directional/=np.maximum(directional.sum(axis=1,keepdims=True),1e-12)
    v=np.clip((radius-warp.start_m)/(warp.full_m-warp.start_m),0,1)
    v=v*v*(3-2*v)
    displacement=np.zeros_like(flat)
    for k in range(4):
        angle=np.deg2rad(warp.angle_deg[k])
        c,s=np.cos(angle),np.sin(angle)
        # Angle change is around intersection origin. Offset is transverse to arm.
        rotation=np.column_stack([(c-1)*x-s*y,s*x+(c-1)*y])
        outward=np.array([[0,-1],[1,0],[0,1],[-1,0]],float)[k]
        normal=np.array([-outward[1],outward[0]])
        displacement+= (v*directional[:,k])[:,None]*(rotation+warp.lateral_m[k]*normal)
    return (flat+displacement).reshape(p.shape)

def transformed_rectangle(p,warp):
    """Maintain strict four-point rectangles for world stop-line/crosswalk.
    Locally transform midpoint plus tangent vectors, then orthogonalize.
    """
    p=np.asarray(p,float); c=p.mean(axis=0)
    raw=deform(p,warp)
    u=raw[1]-raw[0]; v=raw[3]-raw[0]
    lu=np.linalg.norm(u); u/=max(lu,1e-12)
    v=v-u*np.dot(v,u); lv=np.linalg.norm(v); v/=max(lv,1e-12)
    center=deform(c[None,:],warp)[0]
    return np.stack([center-lu/2*u-lv/2*v,center+lu/2*u-lv/2*v,
                     center+lu/2*u+lv/2*v,center-lu/2*u+lv/2*v])

def densify(points, max_step_m=0.35):
    """Densify original geometry BEFORE nonlinear map deformation.

    Otherwise a two-point straight line becomes a chord, while the dense GT
    curves under the same deformation, making them appear to leave the lane.
    """
    p=np.asarray(points,dtype=float)
    if len(p)<2:return p.copy()
    chunks=[p[:1]]
    for a,b in zip(p[:-1],p[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/max_step_m)))
        chunks.append(a[None,:]+np.linspace(1/n,1,n)[:,None]*(b-a)[None,:])
    return np.concatenate(chunks,axis=0)

def warp_world(world,warp):
    import copy
    w=copy.deepcopy(world)
    for obj in w['physical']['elements']:
        pts=np.asarray(obj['points'],float)
        q=transformed_rectangle(pts,warp) if obj['type'] in ('stop_line','crosswalk') else deform(densify(pts),warp)
        obj['points']=np.round(q,5).tolist()
    for road in w['semantic']['roads']:
        road['axis']=deform(densify(np.asarray(road['axis'])),warp).tolist()
    w['metadata']['arm_warp']={'angle_deg':list(warp.angle_deg),'lateral_m':list(warp.lateral_m),
                              'start_m':warp.start_m,'full_m':warp.full_m}
    w['metadata']['generator']='parametric_intersection_v3'
    return w
