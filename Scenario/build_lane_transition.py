"""Preview one-sided lane additions/removals in the existing V4 sample format.

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

    def __post_init__(self):
        if not (1 <= self.lanes_in <= 4 and 1 <= self.lanes_out <= 4
                and 0 < abs(self.lanes_in-self.lanes_out) <= 2):
            raise ValueError('Lane counts must be 1..4 and change by 1 or 2')
        if not 15 <= self.length_m <= 25:
            raise ValueError('Transition length must be 15..25 m')
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
                rng = np.random.default_rng(np.random.SeedSequence([seed, ri, pi, qi]))
                lines, curbs, stops, crosswalks = observe(world, ego, heading, q, rng)
                row = {'id':f'{world["world_id"]}_{rid}_{pose}_{quality}', 'world_id':world['world_id'],
                       'route_id':f'{world["world_id"]}_{rid}', 'ego_pose_id':f'{rid}_{pose}',
                       'coordinate_system':'ego_at_start_x_forward_y_left_m', 'navigation':'straight',
                       'gt_path_sample_spacing_m':1., 'gt_path':np.round(gt, 4).tolist(),
                       'reference_line':np.round(guide, 4).tolist(), 'lane_markings':lines,
                       'road_boundaries':curbs, 'stop_lines':stops, 'crosswalks':crosswalks,
                       'metadata':{'generator_version':'lane_transition_v4',
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
    all_rows=[];reports=[]
    for i in range(args.worlds):
        cfg,world,routes=make_world(args.seed+i,PRESETS[i%len(PRESETS)],args.guide_shape)
        rows=build_samples(cfg,world,routes,qualities)
        folder=args.output_dir/'worlds'/world['world_id'];folder.mkdir()
        (folder/'world.json').write_text(json.dumps(world))
        (folder/'routes.json').write_text(json.dumps(routes))
        if i < args.preview_worlds:
            preview(cfg,world,routes,rows,args.output_dir/'previews'/f'{world["world_id"]}.png')
        all_rows.extend(rows)
        reports.append({'world_id':world['world_id'],**asdict(cfg),'routes':len(routes),'samples':len(rows),
                        'empty_bev_samples':sum(not (r['lane_markings'] or r['road_boundaries']) for r in rows)})
        print(f'[{i+1}/{args.worlds}] {cfg.lanes_in}->{cfg.lanes_out} {cfg.change_side}: {len(rows)} samples',flush=True)
    def write_rows(path,rows):
        with path.open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    write_rows(args.output_dir/'samples.jsonl',all_rows)
    # World-disjoint split. Small batches intentionally may have an empty split.
    order=np.random.default_rng(args.seed).permutation(args.worlds).tolist()
    n_val=n_test=max(1,args.worlds//10) if args.worlds>=3 else 0
    split_ids={'train':order[:args.worlds-n_val-n_test],
               'val':order[args.worlds-n_val-n_test:args.worlds-n_test],
               'test':order[args.worlds-n_test:] if n_test else []}
    split_report={}
    for name,indices in split_ids.items():
        ids={reports[i]['world_id'] for i in indices}
        rows=[r for r in all_rows if r['world_id'] in ids]
        write_rows(args.output_dir/'audit'/f'{name}.jsonl',rows)
        split_report[name]={'world_ids':sorted(ids),'samples':len(rows)}
    report={'generator':'lane_transition_v4','perception_convention':PERCEPTION_CONVENTION,
            'guide_shape':args.guide_shape,'edge_line_inset_m':.25,
            'seed':args.seed,'samples':len(all_rows),'poses':list(POSES),
            'internal_markings_in_transition':False,'outer_edge_markings':True,
            'constraints':{'lane_counts':[1,4],'absolute_count_change':[1,2],'length_m':[15,25]},
            'qualities':{k:asdict(v) for k,v in qualities.items()},'worlds':reports,'splits':split_report,
            'preview_worlds':min(args.worlds,args.preview_worlds),
            'empty_bev_samples':sum(r['empty_bev_samples'] for r in reports),
            'scope':'Single-direction, one-sided lane addition/removal; straight nav.'}
    (args.output_dir/'generation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved {len(all_rows)} samples and {min(args.worlds,args.preview_worlds)} previews: {args.output_dir}')


if __name__=='__main__':main()
