"""Seven review-only road hypotheses, fixed references, and uncropped observations.

Hand drawings are observations, never ground truth. This does not alter the
training generators or run inference/training. Approve the hypotheses visually.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from shapely.geometry import LineString, Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer import encode_input, prepare_sample
from Scenario.build_dataset import arclength, at_s, bezier


FIELDS = ('lane_markings', 'road_boundaries', 'stop_lines', 'crosswalks')
SPECS = (
    ('999977', '右转专用道', 'dedicated_right', 'right', (6., 38., -23., -46., 14., 14.)),
    ('350552', '路口左转', 'junction_left', 'left', (6., 40., 23., 35., 13., 13.)),
    ('632840', '路口直行', 'junction_straight', 'straight', None),
    ('989532', '增道右侧目标', 'addition_right', 'right', None),
    ('246038', '增道中间目标', 'addition_straight', 'straight', None),
    ('416201', '增道左侧目标', 'addition_left', 'left', None),
    ('620082', '路口右转', 'junction_right', 'right', (6., 35., -14., -32.5, 12., 12.)),
)


def smooth(t):
    t = np.clip(t, 0, 1)
    return t**3*(10-15*t+6*t*t)


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def center_curve(kind, parameters):
    if parameters is not None:
        entry, ex, ey, degrees, h1, h2 = parameters
        direction = np.array([np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))])
        a, b = np.array([entry, 0.]), np.array([ex, ey])
        turn = bezier(a, a+[h1,0], b-h2*direction, b, n=601)
        approach = np.column_stack([np.linspace(-25, entry, 351), np.zeros(351)])
        departure = b + np.linspace(0,150,1001)[:,None]*direction
        return np.concatenate([approach[:-1],turn[:-1],departure])
    x = np.linspace(-25,160,1851)
    if kind == 'junction_straight':
        y = -1.6*smooth((x-5)/25)
    else:
        target = {'addition_right':-7., 'addition_straight':-3.5, 'addition_left':0.}[kind]
        y = target*smooth((x-10)/42)
    return np.column_stack([x,y])


def geometry(kind, parameters):
    """Designed independent of Guide/observed coordinates; manual hypotheses."""
    center = center_curve(kind, parameters)
    tangent = np.gradient(center,axis=0)
    tangent /= np.linalg.norm(tangent,axis=1,keepdims=True)
    normal = np.column_stack([-tangent[:,1],tangent[:,0]])
    origin_index = int(np.argmin(np.linalg.norm(center,axis=1)))
    # Approach includes x=0 only approximately; interpolate at the exact origin
    # along its straight segment rather than anchoring an arbitrary sampled point.
    lengths = arclength(center)
    s0 = float(np.interp(0., center[:origin_index+3,0], lengths[:origin_index+3]))
    target = at_s(center, s0+np.arange(81))
    assert np.linalg.norm(target[0])<1e-8
    elements = []
    def add(id, kind_, p, **kwargs):
        elements.append({'id':id,'type':kind_,'points':np.asarray(p).tolist(),**kwargs})
    if kind.startswith('addition'):
        x = center[:,0]
        q = smooth((x-10)/42)
        for name,y in [('left_edge',np.full_like(x,5.25)),('right_edge',-1.75-7*q)]:
            add(name,'line',np.column_stack([x,y]),color='white',style='solid')
            add(name+'_curb','curb',np.column_stack([x,y+(.3 if name=='left_edge' else -.3)]))
        for name, y, start, end in [('old_separator',1.75,-25,160),
                                    ('new_separator_1',-1.75,10,160),
                                    ('new_separator_2',-5.25,35,160)]:
            for section,mask in [('before',(x>=start)&(x<=10)),('after',(x>=max(start,46))&(x<=end))]:
                if mask.sum()>2:
                    add(name+'_'+section,'line',np.column_stack([x[mask],np.full(mask.sum(),y)]),color='white',style='dashed')
        xx=np.linspace(-25,160,1851);qq=smooth((xx-10)/42)
        road=Polygon(np.r_[np.c_[xx,np.full_like(xx,5.55)],np.c_[xx,-2.05-7*qq][::-1]])
        lane_count={'entry':2,'exit':4}
    else:
        offsets = [-1.75,1.75,5.25,8.75] if kind in ('dedicated_right','junction_right') else [-8.75,-5.25,-1.75,1.75]
        for j,offset in enumerate(offsets):
            xy=center+offset*normal
            style='solid' if j in (0,len(offsets)-1) else 'dashed'
            # Dedicated turning road is marked; junction connectors are unmarked.
            if kind.startswith('junction_'):
                exit_station = 18 if kind == 'junction_straight' else 44
                for section,mask in [('before',lengths<=s0+6),('after',lengths>=s0+exit_station)]:
                    add(f'edge_{j}_{section}','line',xy[mask],color='white',style=style)
            else:
                add(f'edge_{j}','line',xy,color='white',style=style)
        low, high = min(offsets)-.3,max(offsets)+.3
        left,right=center+high*normal,center+low*normal
        add('left_curb','curb',left);add('right_curb','curb',right)
        road=Polygon(np.r_[left,right[::-1]])
        lane_count={'entry':3,'exit':3}
    if not road.is_valid:
        raise ValueError(f'{kind}: invalid proposed road envelope')
    if not road.buffer(1e-6).covers(LineString(target)):
        raise ValueError(f'{kind}: proposed GT outside road')
    return {'elements':elements,'road_polygon':list(road.exterior.coords),
            'center_world':center.tolist(),'candidate_lane_counts':lane_count}, target


def make_plan(observed, world):
    """Map visible spans to proposed lines; expose this heuristic for review.

    No FOV clipping. The selected spans approximate each hand-drawn visibility
    window; the physical line is sampled before applying smooth errors.
    """
    plan=[]
    for field,kind in [('lane_markings','line'),('road_boundaries','curb')]:
        candidates=[e for e in world['elements'] if e['type']==kind]
        for obj in observed[field]:
            p=np.asarray(obj['points'])
            samples=p[np.linspace(0,len(p)-1,min(25,len(p))).astype(int)]
            def score(e):
                curve=LineString(e['points'])
                return float(np.mean([curve.distance(Point(v)) for v in samples]))
            source=min(candidates,key=score)
            curve=LineString(source['points'])
            stations=[curve.project(Point(v)) for v in samples]
            lo,hi=min(stations),max(stations)
            if hi-lo<1.5:
                lo=max(0.,curve.project(Point(p.mean(0)))-1.)
                hi=min(curve.length,lo+2.)
            plan.append({'field':field,'observed_id':obj['id'],'source_id':source['id'],
                         'start_s_m':lo,'end_s_m':hi,'mapping_mean_distance_m':score(source)})
    return plan


def simulate(observed, world, plan, seed):
    rng=np.random.default_rng(seed)
    row=copy.deepcopy(observed)
    row['lane_markings']=[];row['road_boundaries']=[]
    sources={e['id']:e for e in world['elements']}
    events=[]
    for item in plan:
        source=sources[item['source_id']]
        p=np.asarray(source['points']);total=arclength(p)[-1]
        # Endpoint variation is relative to a visibility window, never an FOV box.
        lo=float(np.clip(item['start_s_m']+rng.normal(0,1.2),0,total))
        hi=float(np.clip(item['end_s_m']+rng.normal(0,1.2),lo+.2,total))
        if hi-lo<1.3:continue
        stations=np.linspace(lo,hi,max(8,int((hi-lo)/.25)+1))
        xy=at_s(p,stations);t=np.gradient(xy,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-9)
        n=np.c_[-t[:,1],t[:,0]];u=np.linspace(0,1,len(xy))
        fitted=bool(rng.random()<.65)
        shift=float(rng.normal(0,.12))
        tilt=float(rng.uniform(-.45,.45)) if fitted else 0.
        bend=float(rng.uniform(-.65,.65)) if fitted else 0.
        xy=xy+n*(shift+tilt*(2*u-1)+bend*np.sin(np.pi*u)**2)[:,None]
        keep=np.ones(len(xy),dtype=bool);gap=None
        if hi-lo>10 and rng.random()<.3:
            start=float(rng.uniform(.2,.65));width=float(rng.uniform(.06,.14))
            keep[(u>start)&(u<start+width)]=False;gap=[start,start+width]
        edges=np.flatnonzero(np.diff(np.r_[False,keep,False])).reshape(-1,2)
        for j,(a,b) in enumerate(edges):
            if b-a<3 or np.linalg.norm(xy[b-1]-xy[a])<1.3:continue
            obj={'id':item['observed_id']+f'_sim_{j}','source_id':source['id'],
                 'points':xy[a:b].tolist(),'confidence':.95}
            if item['field']=='lane_markings':
                original=next(o for o in observed['lane_markings'] if o['id']==item['observed_id'])
                obj.update(color=original.get('color','white'),line_style=original.get('line_style','solid'))
            row[item['field']].append(obj)
        events.append({**item,'sampled_start_s_m':lo,'sampled_end_s_m':hi,
                       'structured_error':fitted,'shift_m':shift,'tilt_m':tilt,'bend_m':bend,'gap_fraction':gap})
    # Boxes remain the supplied perceived context: no inferred physical box truth.
    return row,events


def validate(world, gt, rows):
    if gt.shape!=(81,2) or not np.isfinite(gt).all() or np.linalg.norm(gt[0])>1e-8:
        raise ValueError('Invalid candidate reference')
    step=np.linalg.norm(np.diff(gt,axis=0),axis=1)
    if step.max()>1.001 or step.min()<.98:raise ValueError('Invalid reference sampling')
    road=Polygon(world['road_polygon'])
    if not road.covers(LineString(gt)):raise ValueError('Reference outside proposed road')
    import torch
    reports=[]
    for row in rows:
        e=encode_input(prepare_sample(row),32,32)
        with_reference=encode_input(prepare_sample({**row,'gt_path':gt.tolist()}),32,32)
        for key in e:
            if not torch.isfinite(e[key]).all():
                raise ValueError(f'Nonfinite encoded input: {key}')
            torch.testing.assert_close(e[key],with_reference[key],rtol=0,atol=0)
        supplied=sum(len(row[k]) for k in FIELDS)
        retained=int(e['polyline_mask'].sum())
        if supplied!=retained:raise ValueError('Unexpected token truncation in review')
        reports.append({'objects':supplied,'retained':retained})
    return {'reference_within_road':True,'reference_max_step_m':float(step.max()),
            'reference_min_step_m':float(step.min()),'additional_fov_crop':False,
            'reference_does_not_enter_model_features':True,'inputs':reports}


def draw_case(world, gt, rows, titles, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(21,6),layout='constrained')
    paths=[np.asarray(rows[0]['reference_line']),gt]
    extent=np.concatenate(paths+[np.asarray(o['points']) for k in FIELDS for o in rows[0][k]])
    xlim=(-extent[:,1].max()-5,-extent[:,1].min()+5)
    ylim=(min(-5.,extent[:,0].min()-3),extent[:,0].max()+5)
    for i,ax in enumerate(axes):
        if i==0:
            for e in world['elements']:
                p=np.asarray(e['points'])
                ax.plot(-p[:,1],p[:,0],c='#333333' if e['type']=='curb' else '#aeb6bf',
                        ls='--' if e.get('style')=='dashed' else '-',lw=1)
            title='Candidate physical road (hypothesis)'
        else:
            row=rows[i-1]
            for key in FIELDS:
                for o in row[key]:
                    p=np.asarray(o['points'])
                    if key in ('stop_lines','crosswalks'):p=np.r_[p,p[:1]]
                    colors={'lane_markings':'#92999f','road_boundaries':'#333333',
                            'stop_lines':'#db5353','crosswalks':'#8f63bf'}
                    ax.plot(-p[:,1],p[:,0],c=colors[key],
                            ls='--' if o.get('line_style')=='dashed' else '-',lw=1.25)
            title=titles[i-1]
        g=np.asarray(rows[0]['reference_line'])
        ax.plot(-g[:,1],g[:,0],c='#e99a00',ls='--',lw=1.8,label='Guide (original)')
        ax.plot(-gt[:,1],gt[:,0],c='#159447',lw=2,label='Candidate GT (unapproved)')
        ax.scatter(0,0,c='black',marker='^',s=20)
        ax.set(xlim=xlim,ylim=ylim,title=title,xlabel='Right (-ego y), m',ylabel='Forward (ego x), m')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    fig.savefig(destination,dpi=140);plt.close(fig)


def build(source, output, seed=71107):
    source,output=Path(source),Path(output)
    locations=[]
    for suffix,*_ in SPECS:
        matches=[p.parent for p in source.glob('*/input.json') if p.parent.name.endswith(suffix)]
        if len(matches)!=1:raise ValueError(f'Expected exactly one saved case ending {suffix}')
        locations.append(matches[0])
    output.mkdir(parents=True,exist_ok=False)
    catalog=[];cards=[]
    for index,((suffix,title,kind,nav,params),folder) in enumerate(zip(SPECS,locations)):
        original=json.loads((folder/'input.json').read_text())
        if original['navigation']!=nav:raise ValueError(f'{suffix}: unexpected navigation')
        observed=prepare_sample(original)
        world,gt=geometry(kind,params)
        plan=make_plan(observed,world)
        row_a,events_a=simulate(observed,world,plan,seed+index*10)
        row_b,events_b=simulate(observed,world,plan,seed+index*10+1)
        reports=validate(world,gt,[observed,row_a,row_b])
        case=output/f'{suffix}_{kind}';case.mkdir()
        shutil.copyfile(folder/'input.json',case/'input_original.json')
        shutil.copyfile(folder/'scene.json',case/'scene_original.json')
        write(case/'input_sim_a.json',row_a);write(case/'input_sim_b.json',row_b)
        write(case/'world_hypothesis.json',world)
        write(case/'reference_candidate.json',{'gt_path':gt.tolist(),'status':'candidate_requires_user_review'})
        annotation={'title':title,'suffix':suffix,'kind':kind,'navigation':nav,
                    'source_folder':str(folder.resolve()),'source_sha256':hashlib.sha256((folder/'input.json').read_bytes()).hexdigest(),
                    'additional_fov_crop':False,'gt_status':'manual design hypothesis, not recovered ground truth',
                    'candidate_lane_counts':world['candidate_lane_counts'],
                    'target_connection':'fixed before observation perturbation; candidate center is designed independently of Guide',
                    'observation_plan':plan,'simulation_events':{'a':events_a,'b':events_b},'validation':reports,
                    'limitations':['Connected road hypothesis, not a full four-arm intersection reconstruction.',
                                   'Original observations are exact; candidate truth may disagree and must be reviewed.',
                                   'Object mapping is a nearest-line heuristic, not confirmed physical source identity.',
                                   'Observed boxes preserved; no physical stop/crosswalk truth is reconstructed.',
                                   'Probabilities and amplitudes are review assumptions, not calibrated statistics.']}
        write(case/'annotation.json',annotation)
        draw_case(world,gt,[observed,row_a,row_b],['Original perceived fragments','Simulated perception A','Simulated perception B'],case/'comparison.png')
        catalog.append({'folder':case.name,'title':title,'validation':reports})
        escaped=html.escape(case.name)
        cards.append(f'<section><h2>{html.escape(title)} ({suffix})</h2><a href="{escaped}/comparison.png"><img src="{escaped}/comparison.png"></a><p><a href="{escaped}/annotation.json">参数与假设</a> · <a href="{escaped}/input_original.json">原输入</a> · <a href="{escaped}/reference_candidate.json">候选GT</a></p></section>')
        print(f'{suffix}: {title}; candidate GT + original observation + 2 simulations',flush=True)
    write(output/'manifest.json',{'seed':seed,'status':'review_only_not_training_data','cases':catalog,
          'additional_fov_crop':False,'probabilities':{'structured_error_per_fragment':.65,'gap_per_eligible_fragment':.3},
          'notes':['No model loaded, no training started.','Each category has one candidate geometry/pose only.']})
    (output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>七类原型候选GT审核</title><style>body{font:16px sans-serif;max-width:1600px;margin:30px auto;background:#f5f5f5}section{background:white;padding:20px;margin:20px 0}img{width:100%}</style><h1>候选道路 / GT 与感知原型审核</h1><p>绿线为待确认的候选GT，不是模型预测；橙线保留原Guide；灰色车道线、黑色路沿、红色停止线、紫色斑马线。不做额外FOV裁剪。所有候选道路、映射和误差概率均为审核假设。</p>'+''.join(cards))
    return catalog


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/scene_editor')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/prototype_review/handdrawn_v1')
    parser.add_argument('--seed',type=int,default=71107)
    args=parser.parse_args();build(args.source,args.output_dir,args.seed)
