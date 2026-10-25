"""V4.2 drivable-surface auditor, computed from the same lane widths and curb fillets as Real Curb.
Requires numpy and shapely>=2.0. Works on existing V4 world/routes JSON, no regeneration.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,LineString,Point
from shapely.ops import unary_union
from shapely.validation import explain_validity
from World.road_warp import ArmWarp,deform,densify
from World.intersection_generator_v4 import OUT,LEFT


def drivable_polygon(world,step=.25):
    md=world['metadata'];w=float(md['lane_width_m']);r=float(md['corner_radius_m']);L=float(md['arm_length_m'])
    ni=np.asarray(md['num_in'],int);no=np.asarray(md['num_out'],int)
    # Road strips and rounded corner clearance are the *same canonical geometry* used
    # by World/intersection_generator_v4.py to produce the outer Real Curb.
    pieces=[]
    for k in range(4):
        p=[s*OUT[k]+l*LEFT[k] for s,l in [(0,-no[k]*w),(L,-no[k]*w),(L,ni[k]*w),(0,ni[k]*w)]]
        pieces.append(Polygon(p))
    for sx in (-1,1):
        for sy in (-1,1):
            kv=2 if sy>0 else 0; kh=1 if sx>0 else 3
            xw=(no[kv] if sx>0 else ni[kv]) if kv==2 else (ni[kv] if sx>0 else no[kv])
            yw=(ni[kh] if sy>0 else no[kh]) if kh==1 else (no[kh] if sy>0 else ni[kh])
            ex=xw*w;ey=yw*w
            n=max(42,int(np.ceil(np.pi*r/2/step)))
            t=np.linspace(0,np.pi/2,n)
            arc=np.stack([ex+r-r*np.sin(t),ey+r-r*np.cos(t)],axis=1)
            q=np.vstack([[0,0],[ex+r,0],arc,[0,ey+r]])*np.array([sx,sy])
            pieces.append(Polygon(q))
    canonical=unary_union(pieces)
    if not canonical.is_valid or canonical.geom_type!='Polygon':
        raise ValueError(f'canonical drivable invalid: {explain_validity(canonical)}')
    wm=md.get('arm_warp')
    if not wm:raise ValueError('world.metadata.arm_warp missing: cannot reconstruct warped curb area')
    warp=ArmWarp(tuple(wm['angle_deg']),tuple(wm['lateral_m']),float(wm['start_m']),float(wm['full_m']))
    # Transform densely sampled canonical OUTER RING, not sparse corner or arm endpoints.
    ring=densify(np.asarray(canonical.exterior.coords),step)
    warped=Polygon(deform(ring,warp))
    if not warped.is_valid or warped.is_empty:
        raise ValueError(f'warped drivable invalid: {explain_validity(warped)}')
    if warped.geom_type!='Polygon':raise ValueError('warped drivable is not single polygon')
    return warped


def audit_one(world,routes, tolerance=.08, curb_tolerance=.12):
    poly=drivable_polygon(world)
    boundary=poly.boundary
    curb_stats=[];issues=[]
    for e in world['physical']['elements']:
        if e['type']!='curb':continue
        coords=np.asarray(e['points'],float)
        if len(coords)<2:continue
        # World coordinates are rounded to 1e-5; compare actual Real Curb to derived surface.
        dist=np.array([boundary.distance(Point(float(p[0]),float(p[1]))) for p in coords])
        mx=float(dist.max());curb_stats.append(mx)
        if mx>curb_tolerance:issues.append({'kind':'curb_polygon_mismatch','object':e['id'],'error_m':round(mx,5)})
    route_stats=[]
    for nav,rt in routes.items():
        pts=np.asarray(rt['gt_world'],float)
        gt=LineString(pts)
        # Entire GT polyline: includes transitions BETWEEN samples.
        illegal=gt.difference(poly.buffer(tolerance))
        illegal_length=float(illegal.length)
        max_outside=float(np.max([poly.distance(Point(*p)) for p in pts]))
        # Boundary clearance is diagnostic only: a centerline close to a curb is still legal.
        min_clearance=float(np.min([boundary.distance(Point(*p)) for p in pts[::max(1,len(pts)//400)]]))
        route_stats.append({'nav':nav,'illegal_length_m':round(illegal_length,5),'max_outside_distance_m':round(max_outside,5),
                            'min_curb_clearance_m':round(min_clearance,5),'valid':illegal_length<.0001})
        if illegal_length>=.0001:issues.append({'kind':'gt_outside_drivable','object':nav,'error_m':round(illegal_length,5)})
    return {'world_id':world['world_id'],'polygon_area_m2':round(poly.area,2),
            'polygon_valid':poly.is_valid,'curb_max_distance_m':round(max(curb_stats,default=0),5),
            'routes':route_stats,'issues':issues},poly


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worlds-dir',required=True);p.add_argument('--routes-dir',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--tolerance-m',type=float,default=.08)
    p.add_argument('--curb-tolerance-m',type=float,default=.12)
    a=p.parse_args();wd=Path(a.worlds_dir);rd=Path(a.routes_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    results=[];issues=[];rows=[]
    files=sorted(wd.glob('*.json'))
    if not files:raise SystemExit(f'No worlds in {wd}')
    for fp in files:
        try:
            world=json.loads(fp.read_text());routes=json.loads((rd/fp.name).read_text())
            rec,poly=audit_one(world,routes,a.tolerance_m,a.curb_tolerance_m)
            results.append(rec)
            for issue in rec['issues']:issues.append({'world_id':rec['world_id'],**issue})
            for r in rec['routes']:rows.append({'world_id':rec['world_id'],**r})
        except Exception as exc:
            issues.append({'world_id':fp.stem,'kind':'audit_exception','object':fp.name,'error_m':'','detail':str(exc)})
    for file,content in [('per_world.json',results),('issues.json',issues)]:
        (out/file).write_text(json.dumps(content,indent=2,ensure_ascii=False))
    if rows:
        with (out/'per_route.csv').open('w',newline='') as f:
            dw=csv.DictWriter(f,fieldnames=list(rows[0]));dw.writeheader();dw.writerows(rows)
    summary={'worlds_requested':len(files),'worlds_audited':len(results),'routes_audited':len(rows),
             'routes_valid':sum(int(x['valid']) for x in rows),'issues':len(issues),
             'polygon_method':'union of canonical road strips and 4 rounded curb-fillets, then same dense nonlinear deformation as World',
             'gt_check':'full LineString covered by buffered drivable polygon (tolerance)',
             'limits':['Drivable area is reconstructed from generator geometry and checked against Real Curb; not solely inferred from disconnected curb lines',
                       'This is physical road-surface legality, not route-specific lane connectivity or vehicle-footprint clearance',
                       'Tolerance is a numeric guard and must not hide substantial curb violations']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if issues:raise SystemExit(2)

if __name__=='__main__':main()
