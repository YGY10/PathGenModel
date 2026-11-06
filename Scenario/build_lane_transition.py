"""Urban lane additions/removals with next-maneuver navigation and reachable targets.
Default CLI mode is urban; --scene-mode legacy reproduces the old straight-only geometry.

python -m Scenario.build_lane_transition --worlds 5 --output-dir outputs/lane_transition_preview
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from Dataset.dataset import encode_sample
from Scenario.build_dataset import arclength, at_s, local_pose, sample_path, to_ego
from Scenario.build_dataset_v4 import (GUIDE_CONVENTION, PERCEPTION_CONVENTION, observe,
                                       read_qualities, sample_world_guide)


PREVIEW_PRESETS = ((1, 3, 'right'), (3, 1, 'left'), (2, 4, 'left'),
                   (4, 2, 'right'), (3, 4, 'right'))
# Cover every allowed count change and both sides when generating larger batches.
PRESETS = PREVIEW_PRESETS + tuple(
    (a, b, side) for a in range(1, 5) for b in range(1, 5)
    for side in ('left', 'right')
    if 0 < abs(a-b) <= 2 and (a, b, side) not in PREVIEW_PRESETS)
URBAN_PRESETS = ((2, 3, 'right'), (3, 2, 'left'), (2, 4, 'right'),
                 (4, 3, 'right'), (3, 4, 'left'))
POSES = ('before_10', 'during_mid', 'after_5')


@dataclass(frozen=True)
class TransitionConfig:
    lanes_in: int
    lanes_out: int
    change_side: str
    width_m: float
    length_m: float
    lateral_shift_m: float
    guide_offset_m: float
    guide_phase_m: float
    guide_shape: str = 'piecewise_linear'
    edge_line_inset_m: float = .25
    scene_mode: str = 'legacy'

    def __post_init__(self):
        if not (1 <= self.lanes_in <= 4 and 1 <= self.lanes_out <= 4
                and 0 < abs(self.lanes_in-self.lanes_out) <= 2):
            raise ValueError('Lane counts must be 1..4 and change by 1 or 2')
        if self.scene_mode not in ('legacy', 'urban'):
            raise ValueError('scene_mode must be legacy or urban')
        low, high = (35, 50) if self.scene_mode == 'urban' else (15, 25)
        if not low <= self.length_m <= high:
            raise ValueError(f'Transition length must be {low}..{high} m')
        if self.change_side not in ('left', 'right'):
            raise ValueError('change_side must be left or right')
        if self.guide_shape not in ('piecewise_linear', 'smooth'):
            raise ValueError('guide_shape must be piecewise_linear or smooth')
        if not 0 < self.edge_line_inset_m < self.width_m/2:
            raise ValueError('edge_line_inset_m must lie between zero and half a lane width')


def smooth(t):
    t = np.clip(t, 0, 1)
    return t**3 * (10 - 15*t + 6*t*t)


def world_xy(x, lateral, cfg):
    """Lane indices run from the fixed edge towards the changing edge."""
    side = 1 if cfg.change_side == 'left' else -1
    baseline = cfg.lateral_shift_m * smooth(np.asarray(x) / cfg.length_m)
    return np.column_stack([x, baseline + side * np.asarray(lateral)])


def connections(n_in, n_out):
    """Preserve inner lanes; outermost lane fans out or outer lanes merge."""
    common = min(n_in, n_out)
    return ([(j, j) for j in range(common - 1)] +
            ([(common - 1, j) for j in range(common - 1, n_out)] if n_out > n_in
             else [(j, common - 1) for j in range(common - 1, n_in)]))


def linear_guide_anchors(cfg, entry, exit_):
    """Four independent map anchors: approach, entry, exit, departure."""
    x = np.array([-90., cfg.guide_phase_m, cfg.length_m + cfg.guide_phase_m, 241.])
    lateral = np.array([entry+.5, entry+.5, exit_+.5, exit_+.5])*cfg.width_m
    lateral += cfg.guide_offset_m*np.array([.65, .65, 1., 1.])
    return world_xy(x, lateral, cfg)


def connection_curve(cfg, entry, exit_, guide=False):
    # Build GT and guide independently from lane endpoints and road parameters.
    if guide and cfg.guide_shape == 'piecewise_linear':
        anchors = linear_guide_anchors(cfg, entry, exit_)
        # Keep exact knots. Dense straight segments support the shared nearest-
        # vertex crop without snapping an inside pose to a distant endpoint.
        x = np.unique(np.r_[np.arange(-90., 241., .2), anchors[:, 0]])
        return np.column_stack([x, np.interp(x, anchors[:, 0], anchors[:, 1])])
    x = np.arange(-90., 241., .2 if not guide else 1.)
    phase = cfg.guide_phase_m if guide else 0.
    blend = smooth((x - phase) / cfg.length_m)
    lateral = cfg.width_m * (entry + .5 + (exit_ - entry)*blend)
    if guide:
        lateral = lateral + cfg.guide_offset_m * (.65 + .35*smooth(x / cfg.length_m))
    return world_xy(x, lateral, cfg)


def make_world(seed, preset, guide_shape='piecewise_linear'):
    rng = np.random.default_rng(seed)
    cfg = TransitionConfig(*preset, float(rng.uniform(3.3, 3.8)),
                           float(rng.uniform(15, 25)), float(rng.uniform(-2, 2)),
                           float(rng.choice([-1, 1])*rng.uniform(.3, .65)),
                           float(rng.uniform(-2, 2)), guide_shape)
    wid = f'lane_transition_{seed:010d}'
    x = np.arange(-90., 241., .2)
    blend = smooth(x / cfg.length_m)
    count = cfg.lanes_in + (cfg.lanes_out - cfg.lanes_in)*blend
    elements = []
    for name, lateral in [('fixed_edge', np.zeros_like(x)), ('changing_edge', count*cfg.width_m)]:
        elements.append({'id':name, 'type':'curb', 'points':world_xy(x, lateral, cfg).tolist()})
        inset = lateral + (cfg.edge_line_inset_m if name=='fixed_edge' else -cfg.edge_line_inset_m)
        elements.append({'id':name+'_marking', 'type':'line', 'color':'white', 'style':'solid',
                         'role':'outer_edge_marking', 'points':world_xy(x, inset, cfg).tolist()})
    # Marked approach and departure, with NO internal markings in (0,length).
    # Store each section separately; never connect a polyline across the gap.
    for section, count, xx in (
            ('before', cfg.lanes_in, np.r_[x[x < 0], 0.]),
            ('after', cfg.lanes_out, np.r_[cfg.length_m, x[x > cfg.length_m]])):
        for j in range(1, count):
            yy = np.full_like(xx, j*cfg.width_m)
            elements.append({'id':f'{section}_separator_{j}', 'type':'line', 'color':'white',
                             'style':'dashed', 'points':world_xy(xx, yy, cfg).tolist()})
    routes = {}
    for entry, exit_ in connections(cfg.lanes_in, cfg.lanes_out):
        rid = f'in{entry}_out{exit_}'
        gt = connection_curve(cfg, entry, exit_)
        guide = connection_curve(cfg, entry, exit_, guide=True)
        # Check continuous drivable envelope independently of discretized BEV.
        xx = gt[:, 0]
        baseline = cfg.lateral_shift_m*smooth(xx/cfg.length_m)
        side = 1 if cfg.change_side == 'left' else -1
        lateral = side*(gt[:, 1]-baseline)
        widths = cfg.width_m*(cfg.lanes_in+(cfg.lanes_out-cfg.lanes_in)*smooth(xx/cfg.length_m))
        if np.any(lateral < 0) or np.any(lateral > widths):
            raise AssertionError('GT leaves the road')
        routes[rid] = {'entry_lane_index':entry, 'exit_lane_index':exit_,
                       'gt_world':gt.tolist(), 'guide_world':guide.tolist()}
    world = {'world_id':wid, 'metadata':{'seed':seed, 'generator':'lane_transition_v4', **asdict(cfg)},
             'physical':{'elements':elements}}
    return cfg, world, routes


def urban_marking_gaps(elements, cfg, seed, pattern):
    """Physical missing markings, represented as separate polylines (never bridged).

    Use a separate RNG so GT, Guide, curb and target choices do not change when
    comparing patterns. Per-separator endpoints vary within a common core gap.
    """
    if pattern not in ('unmarked','partial','continuous'):
        raise ValueError('Unknown urban marking pattern')
    rng=np.random.default_rng(np.random.SeedSequence([seed,9183]))
    start=cfg.length_m*float(rng.uniform(.10,.20))
    end=cfg.length_m*float(rng.uniform(.76,.88))
    internal=[e for e in elements if e.get('role','').endswith('_separator')]
    keep=internal[0]['id'] if pattern=='partial' and len(internal)>1 else None
    output=[];gaps=[]
    for e in elements:
        if e not in internal or pattern=='continuous' or e['id']==keep:
            output.append(e);continue
        a=start+float(rng.uniform(-.035,.035))*cfg.length_m
        b=end+float(rng.uniform(-.035,.035))*cfg.length_m
        p=np.asarray(e['points'])
        gaps.append({'source_id':e['id'],'start_x_m':a,'end_x_m':b})
        # Keep exact cut endpoints if the original polyline crosses them.
        for name,lo,hi in [('before',float(p[0,0]),min(a,float(p[-1,0]))),
                           ('after',max(b,float(p[0,0])),float(p[-1,0]))]:
            if hi-lo<.1:continue
            xx=np.r_[lo,p[(p[:,0]>lo)&(p[:,0]<hi),0],hi]
            yy=np.interp(xx,p[:,0],p[:,1])
            output.append({**e,'id':e['id']+'_'+name,'source_separator_id':e['id'],
                           'points':np.column_stack([xx,yy]).tolist()})
    core=[max(g['start_x_m'] for g in gaps),min(g['end_x_m'] for g in gaps)] if pattern=='unmarked' and gaps else None
    return output,{'marking_pattern':pattern,'marking_gap_intervals':gaps,
                   'unmarked_core_world_x_m':core,'continuous_separator_id':keep}


def shade_marking_gap(ax, world, ego, heading):
    """Shade the guaranteed fully unmarked physical strip in ego coordinates."""
    core=world['metadata'].get('unmarked_core_world_x_m')
    if not core:return
    from matplotlib.patches import Polygon
    a,b=core
    polygon=to_ego(np.array([[a,-40],[b,-40],[b,40],[a,40]]),ego,heading)
    ax.add_patch(Polygon(polygon,facecolor='#ffc85c',alpha=.13,edgecolor='none',zorder=0))


def make_urban_world(seed, preset, guide_shape='piecewise_linear', marking_pattern='unmarked'):
    """Navigation-conditioned lane preparation, with at most one lane adjustment.

    Navigation denotes the NEXT maneuver; this road segment remains forward.
    Slots are measured from the fixed road edge. Retained separators continue;
    added separators start inside the widening and removed ones terminate.
    """
    rng = np.random.default_rng(seed)
    cfg = TransitionConfig(*preset, float(rng.uniform(3.3, 3.8)),
                           float(rng.uniform(35, 50)), float(rng.uniform(-1, 1)),
                           float(rng.choice([-1, 1])*rng.uniform(.3, .9)),
                           float(rng.uniform(-7, 7)), guide_shape, scene_mode='urban')
    wid = f'urban_transition_{seed:010d}'
    x = np.arange(-90., 241., .2)
    blend = smooth(x/cfg.length_m)
    count = cfg.lanes_in+(cfg.lanes_out-cfg.lanes_in)*blend
    elements = []
    for name, lateral in [('fixed_edge', np.zeros_like(x)), ('changing_edge', count*cfg.width_m)]:
        elements.append({'id':name, 'type':'curb', 'points':world_xy(x,lateral,cfg).tolist()})
        inset = lateral+(cfg.edge_line_inset_m if name=='fixed_edge' else -cfg.edge_line_inset_m)
        elements.append({'id':name+'_marking','type':'line','color':'white','style':'solid',
                         'role':'outer_edge_marking','points':world_xy(x,inset,cfg).tolist()})
    common = min(cfg.lanes_in,cfg.lanes_out)
    for j in range(1,max(cfg.lanes_in,cfg.lanes_out)):
        if j < common:
            xx=x; role='continuing_separator'
        elif cfg.lanes_out > cfg.lanes_in:
            # Start only when widening has provided room for the new boundary.
            threshold=j+.18
            valid=np.flatnonzero(count>=threshold)
            start=float(x[valid[0]])
            xx=x[x>=start]; role='added_separator'
        else:
            threshold=j+.18
            valid=np.flatnonzero(count>=threshold)
            end=float(x[valid[-1]])
            xx=x[x<=end]; role='ending_separator'
        elements.append({'id':f'separator_{j}','type':'line','color':'white','style':'dashed',
                         'role':role,'points':world_xy(xx,np.full_like(xx,j*cfg.width_m),cfg).tolist()})
    elements, marking_metadata = urban_marking_gaps(elements,cfg,seed,marking_pattern)
    routes={}
    side=1 if cfg.change_side=='left' else -1
    for entry in range(cfg.lanes_in):
        # Conservative preview scope: do not synthesize multi-lane traversals.
        reachable=[j for j in range(cfg.lanes_out) if abs(j-entry)<=1]
        if not reachable:continue
        targets={'left':max(reachable,key=lambda j:side*j),
                 'right':min(reachable,key=lambda j:side*j),
                 'straight':min(reachable,key=lambda j:(abs(cfg.lateral_shift_m+side*(j-entry)*cfg.width_m),abs(j-entry),j))}
        for nav,exit_ in targets.items():
            gt=connection_curve(cfg,entry,exit_)
            guide=connection_curve(cfg,entry,exit_,guide=True)
            # Both road feasibility and a bounded geometric demand, not a claim
            # of speed-dependent vehicle dynamic feasibility.
            lateral=side*(gt[:,1]-cfg.lateral_shift_m*smooth(gt[:,0]/cfg.length_m))
            width=cfg.width_m*(cfg.lanes_in+(cfg.lanes_out-cfg.lanes_in)*smooth(gt[:,0]/cfg.length_m))
            if np.any(lateral<0) or np.any(lateral>width):
                raise AssertionError('Urban GT leaves road envelope')
            if np.max(np.abs(np.diff(gt[:,1])/np.diff(gt[:,0])))>.3:
                raise AssertionError('Urban lane adjustment too steep')
            rid=f'in{entry}_out{exit_}_{nav}'
            routes[rid]={'entry_lane_index':entry,'exit_lane_index':exit_,
                         'navigation':nav,'reachable_exit_lanes':reachable,
                         'lane_adjustment_count':abs(exit_-entry),
                         'gt_world':gt.tolist(),'guide_world':guide.tolist()}
    world={'world_id':wid,'metadata':{'seed':seed,'generator':'urban_lane_transition_v1',
            'navigation_semantics':'next_maneuver_lane_preparation',**asdict(cfg),**marking_metadata},
           'physical':{'elements':elements}}
    return cfg,world,routes


def urban_preview(cfg, world, routes, rows, destination):
    """Rows=left/straight/right, columns=three poses; clean perception."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Same entry across nav rows makes the target selection directly comparable.
    entry=min(cfg.lanes_in//2, cfg.lanes_out-1)
    selected={}
    for nav in ('left','straight','right'):
        rid=next(k for k,r in routes.items() if r['entry_lane_index']==entry and r['navigation']==nav)
        for pose in POSES:
            selected[nav,pose]=next(r for r in rows if r['route_id']==world['world_id']+'_'+rid
                                    and r['metadata']['pose_name']==pose and r['metadata']['quality']==('clean' if any(r['metadata']['quality']=='clean' for r in rows) else rows[0]['metadata']['quality']))
    fig,axes=plt.subplots(3,3,figsize=(18,10),layout='constrained')
    for ni,nav in enumerate(('left','straight','right')):
        for pi,pose in enumerate(POSES):
            ax=axes[ni,pi];row=selected[nav,pose];md=row['metadata']
            ego=np.asarray(md['ego_world_xy']);h=np.array([np.cos(md['heading_rad']),np.sin(md['heading_rad'])])
            shade_marking_gap(ax,world,ego,h)
            for e in world['physical']['elements']:
                p=to_ego(np.asarray(e['points']),ego,h)
                ax.plot(p[:,0],p[:,1],color='#677785' if e['type']=='curb' else '#c5ac70',lw=1,alpha=.8)
            for e in row['lane_markings']+row['road_boundaries']:
                p=np.asarray(e['points']);ax.plot(p[:,0],p[:,1],color='#9961c0',lw=1,alpha=.65)
            for key,col,style,label in [('gt_path','#159447','-','GT'),('reference_line','#e99a00','--','Guide')]:
                p=np.asarray(row[key]);ax.plot(p[:,0],p[:,1],color=col,ls=style,lw=2,label=label)
                ax.scatter(p[-1,0],p[-1,1],color=col,s=15)
            ax.scatter(0,0,c='black',marker='^',s=25)
            ax.set_title(f'{nav} | {pose} | slot {md["entry_lane_index"]} -> {md["exit_lane_index"]}',fontsize=10)
            ax.set(xlim=(-18,90),ylim=(-18,18),xlabel='Forward x (m)',ylabel='Left y (m)')
            ax.set_aspect('equal');ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle(f'{world["world_id"]}: {cfg.lanes_in}->{cfg.lanes_out}, {cfg.change_side} side; '
                 f'length={cfg.length_m:.1f}m | markings={world["metadata"]["marking_pattern"]} | nav=NEXT maneuver\n'
                 'Pale yellow: fully unmarked core; gray: curb; tan: markings; purple: perception; slots counted from fixed edge')
    fig.savefig(destination,dpi=140);plt.close(fig)


def urban_overview(output_dir, reports, rows):
    """Compare nav targets at the identical entry pose in each preview World."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(reports),1,figsize=(16,3*len(reports)),squeeze=False,layout='constrained')
    for ax,report in zip(axes.flat,reports):
        wid=report['world_id'];entry=min(report['lanes_in']//2,report['lanes_out']-1)
        selected=[r for r in rows if r['world_id']==wid and r['metadata']['entry_lane_index']==entry
                  and r['metadata']['pose_name']=='before_10']
        quality='clean' if any(r['metadata']['quality']=='clean' for r in selected) else selected[0]['metadata']['quality']
        selected=[r for r in selected if r['metadata']['quality']==quality]
        world=json.loads((output_dir/'worlds'/wid/'world.json').read_text())
        md=selected[0]['metadata'];ego=np.asarray(md['ego_world_xy']);h=np.array([np.cos(md['heading_rad']),np.sin(md['heading_rad'])])
        shade_marking_gap(ax,world,ego,h)
        for e in world['physical']['elements']:
            p=to_ego(np.asarray(e['points']),ego,h)
            ax.plot(p[:,0],p[:,1],color='#65717b' if e['type']=='curb' else '#bfc3c6',lw=1)
        targets=[]
        for nav,col in [('left','#2875d4'),('straight','#159447'),('right','#c43586')]:
            row=next(r for r in selected if r['navigation']==nav)
            for key,style,label in [('gt_path','-',nav+' GT'),('reference_line','--',nav+' Guide')]:
                p=np.asarray(row[key]);ax.plot(p[:,0],p[:,1],color=col,ls=style,lw=2 if key=='gt_path' else 1.2,label=label)
            targets.append(f'{nav}: {entry}->{row["metadata"]["exit_lane_index"]}')
        ax.scatter(0,0,c='black',marker='^',s=30)
        ax.set_title(f'{wid}: {report["lanes_in"]}->{report["lanes_out"]} lanes, {report["change_side"]} side, '
                     f'{report["length_m"]:.1f}m, markings={report.get("marking_pattern","legacy")} | '+', '.join(targets),fontsize=10)
        ax.set(xlim=(-5,90),ylim=(-14,14),xlabel='Forward x (m)',ylabel='Left y (m)')
        ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(ncol=3,fontsize=7,loc='upper right')
    fig.savefig(output_dir/'overview.png',dpi=140);plt.close(fig)
    html=['<!doctype html><meta charset="utf-8"><title>城区增减道导航预览</title>',
          '<style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:24px}img{width:100%}</style>',
          '<h1>5 个 World：城区增减道 + 导航选道</h1>',
          '<p>浅黄色区域为物理上完全没有内部标线的公共空白段；partial 为部分分隔线缺失，continuous 为连续对照。导航表示后续意图，当前最多调整一个车道。总览实线为 GT，虚线为 Guide；蓝/绿/红分别表示左/直/右导航。详细图三行为导航、三列为自车位置，紫色为实际感知。</p>',
          '<a href="overview.png"><img src="overview.png"></a>']
    for report in reports:
        wid=report['world_id'];html.append(f'<h2>{wid}</h2><a href="previews/{wid}.png"><img src="previews/{wid}.png"></a>')
    (output_dir/'gallery.html').write_text('\n'.join(html))


def pose_stations(gt, length):
    arc = arclength(gt)
    entry, exit_ = np.interp([0, length], gt[:, 0], arc)
    return (entry-10, (entry+exit_)/2, exit_+5)


def build_samples(cfg, world, routes, qualities):
    rows = []
    seed = world['metadata']['seed']
    for ri, (rid, route) in enumerate(routes.items()):
        gt_world, guide_world = np.asarray(route['gt_world']), np.asarray(route['guide_world'])
        for pi, (pose, station) in enumerate(zip(POSES, pose_stations(gt_world, cfg.length_m))):
            ego, heading = local_pose(gt_world, station)
            gt = to_ego(sample_path(gt_world, station), ego, heading)
            guide, guide_s = sample_world_guide(guide_world, ego, heading)
            for qi, (quality, q) in enumerate(qualities.items()):
                perception_route = route['entry_lane_index'] if cfg.scene_mode=='urban' else ri
                rng = np.random.default_rng(np.random.SeedSequence([seed, perception_route, pi, qi]))
                lines, curbs, stops, crosswalks = observe(world, ego, heading, q, rng)
                row = {'id':f'{world["world_id"]}_{rid}_{pose}_{quality}', 'world_id':world['world_id'],
                       'route_id':f'{world["world_id"]}_{rid}', 'ego_pose_id':f'{rid}_{pose}',
                       'coordinate_system':'ego_at_start_x_forward_y_left_m', 'navigation':route.get('navigation','straight'),
                       'gt_path_sample_spacing_m':1., 'gt_path':np.round(gt, 4).tolist(),
                       'reference_line':np.round(guide, 4).tolist(), 'lane_markings':lines,
                       'road_boundaries':curbs, 'stop_lines':stops, 'crosswalks':crosswalks,
                       'metadata':{'generator_version':world['metadata']['generator'],
                                   'navigation_semantics':world['metadata'].get('navigation_semantics','straight'),
                                   'marking_pattern':world['metadata'].get('marking_pattern','legacy'),
                                   'marking_gap_intervals':world['metadata'].get('marking_gap_intervals',[]),
                                   'unmarked_core_world_x_m':world['metadata'].get('unmarked_core_world_x_m'),
                                   'reachable_exit_lanes':route.get('reachable_exit_lanes',[]),
                                   'lane_adjustment_count':route.get('lane_adjustment_count',abs(route['exit_lane_index']-route['entry_lane_index'])),
                                   'perception_convention':PERCEPTION_CONVENTION,'scene_type':'lane_transition',
                                   'quality':quality, 'pose_name':pose, **asdict(cfg),
                                   'entry_lane_index':route['entry_lane_index'],
                                   'exit_lane_index':route['exit_lane_index'],
                                   'ego_world_xy':ego.tolist(), 'heading_rad':float(np.arctan2(heading[1],heading[0])),
                                   'ego_route_s_m':float(station), 'guide_start_s_m':guide_s,
                                   'guide_coordinate_convention':GUIDE_CONVENTION,
                                   'vcs_fov_m':{'forward':q.forward_m,'rear':q.rear_m,'left':q.left_m,'right':q.right_m}}}
                encode_sample(row)
                # Fully missed observations are valid, especially for hard/one-lane
                # scenes. Keep them: the model explicitly supports empty BEV.
                assert np.max(np.abs(gt[0])) < 1e-8
                rows.append(row)
    return rows


def preview(cfg, world, routes, rows, destination):
    """Same ego axes and truth/observation colours as intersection previews.

    Rows are quality levels; columns are poses. All panels use identical limits.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rid = max(routes, key=lambda k: abs(routes[k]['entry_lane_index']-routes[k]['exit_lane_index']))
    qualities = list(dict.fromkeys(row['metadata']['quality'] for row in rows))
    selected = {(r['metadata']['quality'], r['metadata']['pose_name']): r
                for r in rows if r['route_id'].endswith('_'+rid)}
    paths = np.concatenate([np.asarray(r[key]) for r in selected.values()
                            for key in ('gt_path', 'reference_line')])
    lateral_limit = max(30., float(np.abs(paths[:, 1]).max())+5.)
    fig, axes = plt.subplots(len(qualities), len(POSES), figsize=(15, 3.8*len(qualities)),
                             squeeze=False, layout='constrained')
    for qi, quality in enumerate(qualities):
        for pi, pose in enumerate(POSES):
            ax = axes[qi, pi]
            row = selected[quality, pose]
            md = row['metadata']
            ego = np.asarray(md['ego_world_xy'])
            heading = np.array([np.cos(md['heading_rad']), np.sin(md['heading_rad'])])
            used = set()
            for obj in world['physical']['elements']:
                p = to_ego(np.asarray(obj['points']), ego, heading)
                name = 'World boundary' if obj['type']=='curb' else 'World lane marking'
                ax.plot(p[:, 0], p[:, 1], color='#8c9ba8' if obj['type']=='curb' else '#dcb45d',
                        lw=1 if obj['type']=='curb' else .7, alpha=.65,
                        label=name if name not in used else None)
                used.add(name)
            # Raw observation fragments, as in the intersection preview (not world truth).
            for obj in row['lane_markings'] + row['road_boundaries']:
                p = np.asarray(obj['points'])
                ax.plot(p[:, 0], p[:, 1], color='#8f63bf', lw=1.3,
                        label='Perception' if 'Perception' not in used else None)
                used.add('Perception')
            for key, color, style, label in [('gt_path','#1c9c62','-','GT'),
                                             ('reference_line','#e29b30','--','Guide')]:
                p = np.asarray(row[key])
                ax.plot(p[:, 0], p[:, 1], color=color, ls=style, lw=2 if key=='gt_path' else 1.5,
                        label=label)
            q = md['vcs_fov_m']
            ax.add_patch(Rectangle((-q['rear'], -q['right']), q['forward']+q['rear'],
                                   q['left']+q['right'], fill=False, ec='#7ab6d1', ls=':', lw=1, label='FOV'))
            ax.scatter(0,0,c='black',s=15,zorder=5)
            n_lines, n_curbs = len(row['lane_markings']), len(row['road_boundaries'])
            ax.set_title(f'{quality} / {pose} | lines={n_lines}, boundaries={n_curbs}', fontsize=10)
            ax.set(xlim=(-24,90), ylim=(-lateral_limit,lateral_limit),
                   xlabel='Forward (ego x), m', ylabel='Left (ego y), m')
            ax.set_aspect('equal')
            ax.grid(alpha=.12)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=6, fontsize=8)
    fig.suptitle(f'{world["world_id"]}: {cfg.lanes_in} -> {cfg.lanes_out}, {cfg.change_side} side | '
                 f'{rid} | unmarked transition={cfg.length_m:.1f} m | guide={cfg.guide_shape}')
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds',type=int,default=5)
    parser.add_argument('--seed',type=int,default=41000000)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/lane_transition_preview'))
    parser.add_argument('--perception-config',type=Path)
    parser.add_argument('--scene-mode',choices=('urban','legacy'),default='urban')
    parser.add_argument('--marking-pattern',choices=('mixed','unmarked','partial','continuous'),default='unmarked',
                        help='Urban physical markings; mixed cycles unmarked/partial/continuous/unmarked/partial')
    parser.add_argument('--guide-shape',choices=('piecewise_linear','smooth'),default='piecewise_linear')
    parser.add_argument('--preview-worlds',type=int,default=5,
                        help='Only render the first N worlds; 0 disables previews')
    args=parser.parse_args()
    if args.worlds<1:parser.error('--worlds must be positive')
    if args.preview_worlds<0:parser.error('--preview-worlds must be nonnegative')
    qualities=read_qualities(args.perception_config)
    if not qualities:parser.error('At least one perception quality is required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name in ('worlds','previews','audit'):(args.output_dir/name).mkdir()
    # Stream full samples to disk; retain only preview Worlds in memory.
    # Split indices use the same RNG/order as the original in-memory exporter.
    from contextlib import ExitStack
    order=np.random.default_rng(args.seed).permutation(args.worlds).tolist()
    n_val=n_test=max(1,args.worlds//10) if args.worlds>=3 else 0
    split_ids={'train':order[:args.worlds-n_val-n_test],
               'val':order[args.worlds-n_val-n_test:args.worlds-n_test],
               'test':order[args.worlds-n_test:] if n_test else []}
    split_for_index={i:name for name,indices in split_ids.items() for i in indices}
    split_report={name:{'world_ids':[],'samples':0} for name in split_ids}
    preview_rows=[];reports=[];sample_count=0
    navigation_counts={nav:0 for nav in ('left','straight','right')}
    with ExitStack() as stack:
        samples_file=stack.enter_context((args.output_dir/'samples.jsonl').open('w'))
        split_files={name:stack.enter_context((args.output_dir/'audit'/f'{name}.jsonl').open('w'))
                     for name in split_ids}
        for i in range(args.worlds):
            if args.scene_mode=='urban':
                pattern=('unmarked','partial','continuous','unmarked','partial')[i%5] if args.marking_pattern=='mixed' else args.marking_pattern
                cfg,world,routes=make_urban_world(args.seed+i,URBAN_PRESETS[i%len(URBAN_PRESETS)],args.guide_shape,pattern)
            else:
                cfg,world,routes=make_world(args.seed+i,PRESETS[i%len(PRESETS)],args.guide_shape)
            rows=build_samples(cfg,world,routes,qualities)
            folder=args.output_dir/'worlds'/world['world_id'];folder.mkdir()
            (folder/'world.json').write_text(json.dumps(world))
            (folder/'routes.json').write_text(json.dumps(routes))
            if i < args.preview_worlds:
                renderer=urban_preview if args.scene_mode=='urban' else preview
                renderer(cfg,world,routes,rows,args.output_dir/'previews'/f'{world["world_id"]}.png')
            if i < args.preview_worlds:preview_rows.extend(rows)
            split=split_for_index[i]
            split_report[split]['world_ids'].append(world['world_id'])
            split_report[split]['samples']+=len(rows)
            sample_count+=len(rows)
            for row in rows:
                serialized=json.dumps(row)+'\n'
                samples_file.write(serialized);split_files[split].write(serialized)
                navigation_counts[row['navigation']]+=1
            reports.append({'world_id':world['world_id'],**asdict(cfg),
                            'marking_pattern':world['metadata'].get('marking_pattern','legacy'),
                            'unmarked_core_world_x_m':world['metadata'].get('unmarked_core_world_x_m'),
                            'routes':len(routes),'samples':len(rows),
                            'empty_bev_samples':sum(not (r['lane_markings'] or r['road_boundaries']) for r in rows)})
            print(f'[{i+1}/{args.worlds}] {cfg.lanes_in}->{cfg.lanes_out} {cfg.change_side}: {len(rows)} samples',flush=True)
    for values in split_report.values():values['world_ids'].sort()
    report={'generator':'urban_lane_transition_v1' if args.scene_mode=='urban' else 'lane_transition_v4','perception_convention':PERCEPTION_CONVENTION,
            'guide_shape':args.guide_shape,'edge_line_inset_m':.25,
            'seed':args.seed,'samples':sample_count,'poses':list(POSES),
            'internal_markings_in_transition':args.scene_mode=='urban','outer_edge_markings':True,
            'constraints':{'lane_counts':[1,4],'absolute_count_change':[1,2],'length_m':[35,50] if args.scene_mode=='urban' else [15,25]},
            'qualities':{k:asdict(v) for k,v in qualities.items()},'worlds':reports,'splits':split_report,
            'preview_worlds':min(args.worlds,args.preview_worlds),
            'empty_bev_samples':sum(r['empty_bev_samples'] for r in reports),
            'navigation_counts':navigation_counts,
            'scene_mode':args.scene_mode,'marking_pattern_recipe':args.marking_pattern,
            'marking_pattern_counts':{k:sum(r['marking_pattern']==k for r in reports) for k in ('unmarked','partial','continuous','legacy')},
            'scope':'One-sided urban lane addition/removal; next-maneuver nav, max one lane adjustment; unreachable entries omitted.' if args.scene_mode=='urban' else 'Legacy straight nav.'}
    (args.output_dir/'generation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.scene_mode=='urban' and args.preview_worlds:
        urban_overview(args.output_dir,reports[:args.preview_worlds],preview_rows)
    print(f'Saved {sample_count} samples and {min(args.worlds,args.preview_worlds)} previews: {args.output_dir}')


if __name__=='__main__':main()
