"""Matched world-station evaluation across entry / inside_8 / inside_13.

Evaluates the same GT exit stations at every pose, not each pose's moving horizon.
Geometry support uses only retained, 32-point-encoded observations; truth source
IDs are used for diagnostic association only, never model inputs.
"""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from Dataset.dataset import _objects, resample
from Dataset.audit_lane_visibility import sample_at, target_boundaries, point_polyline_distance
from Dataset.context_augmentation import inspect_context_cache
from diagnose_paths import to_world, tangent, components
from evaluate import EvaluationDataset
from train import inspect_split

POSES=('entry','inside_8','inside_13')


def at_world_stations(path, metadata, stations):
    """Prediction indices mean GT future arc stations; never nearest-match paths."""
    future=np.asarray(stations)-metadata['ego_route_s_m']
    if future.min() < -1e-7 or future.max()>len(path)-1+1e-7:
        raise ValueError('Requested stations outside prediction horizon')
    local=np.column_stack([np.interp(future,np.arange(len(path)),path[:,j]) for j in (0,1)])
    return to_world(local,metadata)


def expected_boundaries(world,route,centers):
    physical={e['id']:e for e in world['physical']['elements']};result={}
    for source in target_boundaries(world,route):
        line=np.asarray(physical[source]['points']);a=line[:-1];v=np.diff(line,axis=0)
        u=np.clip(((centers[:,None]-a)*v).sum(-1)/np.maximum((v*v).sum(-1),1e-12),0,1)
        projected=a+u[...,None]*v
        j=((centers[:,None]-projected)**2).sum(-1).argmin(-1)
        result[source]=projected[np.arange(len(centers)),j]
    return result


def observed_support(expected,objects,metadata,tolerance):
    sides=[]
    for source,points in expected.items():
        mask=np.zeros(len(points),dtype=bool)
        for kind,obj in objects:
            if kind not in ('lane_marking','road_boundary') or obj.get('source_id')!=source:continue
            # Exactly the model's retained polyline representation, rigidly transformed.
            observed=to_world(resample(obj['points'],32),metadata)
            mask|=point_polyline_distance(points,observed)<=tolerance
        sides.append(mask)
    return np.stack(sides)


def common_metrics(pred,gt,guide,direction,sides):
    delta=pred-gt;local=components(delta,direction)
    both=sides.all(0);one=sides.any(0)&~both;none=~sides.any(0)
    return {'common_error_m':float(np.linalg.norm(delta,axis=-1).mean()),
            'lateral_error_m':float(np.abs(local[:,1]).mean()),
            'longitudinal_error_m':float(np.abs(local[:,0]).mean()),
            'guide_common_error_m':float(np.linalg.norm(guide-gt,axis=-1).mean()),
            'both_supported_points':float(both.sum()),'one_supported_points':float(one.sum()),
            'unsupported_points':float(none.sum()),'both_supported_fraction':float(both.mean())}


def aggregate(rows):
    keys=('common_error_m','lateral_error_m','longitudinal_error_m','guide_common_error_m',
          'full_ade_m','both_supported_points','one_supported_points','unsupported_points','both_supported_fraction')
    return {'triplets':len(rows),'worlds':len({r['world_id'] for r in rows}),
            **{k:float(np.mean([r[k] for r in rows])) for k in keys}}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(args):
    report=json.loads((args.evaluation_dir/'summary.json').read_text())
    manifest=json.loads((args.supplement_cache/'manifest.json').read_text())
    info=inspect_split(manifest['source_path'])
    if info!=report['data_manifest']:raise ValueError('Data manifest mismatch')
    inspect_context_cache(args.supplement_cache,info,32,32)
    base=EvaluationDataset(manifest['source_path'],32,32)
    raw=[base.raw_row(i) for i in range(len(base))]
    with np.load(args.evaluation_dir/'predictions.npz',allow_pickle=False) as a:
        ids=a['ids'].tolist();arrays={k:a[k] for k in a.files if k!='ids'}
    if ids!=[r['id'] for r in raw]:raise ValueError('Prediction identity mismatch')
    pools={}
    for e in manifest['files']:
        pool=[json.loads(line) for line in (args.supplement_cache/e['path']).open()]
        if [r['id'] for r in pool]!=ids:raise ValueError('Supplement IDs differ')
        pools[e['seed']]=pool
    groups={}
    for i,r in enumerate(raw):
        pose=r['metadata']['pose_name']
        if pose in POSES:groups.setdefault((r['world_id'],r['navigation'],r['metadata']['quality']),{})[pose]=i
    worlds={};contexts={};records=[];curves={};skipped=[];closure=0.
    offsets=np.arange(args.start_m,args.end_m+1e-7,1.)
    for key,index in groups.items():
        wid,nav,quality=key
        if set(index)!=set(POSES):
            skipped.append({'key':list(key),'reason':'missing pose'});continue
        if wid not in worlds:
            w=json.loads((args.worlds_dir/f'{wid}.json').read_text())
            rt=json.loads((args.routes_dir/f'{wid}.json').read_text());worlds[wid]=(w,rt)
        world,routes=worlds[wid];route=routes[nav]
        eligible=np.ones(len(offsets),dtype=bool)
        for i in index.values():
            future=route['exit_s']+offsets-raw[i]['metadata']['ego_route_s_m']
            eligible&=(future>=0)&(future<=80)
        d=offsets[eligible]
        if len(d)<args.min_points:
            skipped.append({'key':list(key),'reason':'insufficient common exit horizon'});continue
        stations=route['exit_s']+d;truth=sample_at(route['gt_world'],stations);direction=tangent(truth)
        expected=expected_boundaries(world,route,truth)
        contexts[key]={'index':index,'offsets':d,'truth':truth,'world':world,'route':route,'expected':expected}
        for pose in POSES:
            i=index[pose];row=raw[i];md=row['metadata'];observations=_objects(row,32)
            reconstructed=at_world_stations(arrays['gt'][i],md,stations)
            closure=max(closure,float(np.linalg.norm(reconstructed-truth,axis=-1).max()))
            guide=at_world_stations(arrays['guide'][i],md,stations)
            for fov in ('original','expanded'):
                arms=[('original',None)] if fov=='original' else [(f'expanded_seed_{s}',pool[i]) for s,pool in pools.items()]
                values=[];error_curves=[];supported_curves=[]
                for arm,poolentry in arms:
                    added=[] if poolentry is None else [(o['kind'],o) for o in poolentry['added_objects']]
                    if len(observations)+len(added)>32:raise ValueError('Capacity overflow')
                    sides=observed_support(expected,observations+added,md,args.support_distance_m)
                    predicted=at_world_stations(arrays[arm][i],md,stations)
                    metrics=common_metrics(predicted,truth,guide,direction,sides)
                    metrics['full_ade_m']=float(np.linalg.norm(arrays[arm][i,1:]-arrays['gt'][i,1:],axis=-1).mean())
                    values.append(metrics);error_curves.append(np.linalg.norm(predicted-truth,axis=-1));supported_curves.append(sides.all(0))
                records.append({'world_id':wid,'navigation':nav,'quality':quality,'pose':pose,'fov':fov,'index':i,
                    'checked_points':len(d),'exit_start_m':float(d[0]),'exit_end_m':float(d[-1]),
                    **{k:float(np.mean([v[k] for v in values])) for k in values[0]}})
                curves[key,pose,fov]=(np.mean(error_curves,axis=0),np.mean(supported_curves,axis=0))
    if closure>.02:raise ValueError(f'GT station closure failure: {closure}')
    lookup={(r['world_id'],r['navigation'],r['quality'],r['pose'],r['fov']):r for r in records}
    pairs=[]
    for key in contexts:
        for fov in ('original','expanded'):
            start=lookup[(*key,'entry',fov)]
            for pose in POSES[1:]:
                end=lookup[(*key,pose,fov)]
                pairs.append({'world_id':key[0],'navigation':key[1],'quality':key[2],'fov':fov,'to_pose':pose,
                    'entry_error_m':start['common_error_m'],'later_error_m':end['common_error_m'],
                    'error_improvement_m':start['common_error_m']-end['common_error_m'],
                    'lateral_improvement_m':start['lateral_error_m']-end['lateral_error_m'],
                    'support_gain_points':end['both_supported_points']-start['both_supported_points'],
                    'guide_improvement_m':start['guide_common_error_m']-end['guide_common_error_m'],
                    'full_ade_improvement_m':start['full_ade_m']-end['full_ade_m']})
    def paired_summary(rs):
        if not rs:return {'triplets':0}
        return {'triplets':len(rs),'worlds':len({r['world_id'] for r in rs}),
            **{k:float(np.mean([r[k] for r in rs])) for k in
               ('entry_error_m','later_error_m','error_improvement_m','lateral_improvement_m','support_gain_points','guide_improvement_m','full_ade_improvement_m')},
            'fraction_improved_over_10cm':float(np.mean([r['error_improvement_m']>.1 for r in rs]))}
    summary={'checkpoint':report['checkpoint'],'data_manifest':info,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'matched_triplets':len(contexts),'skipped':skipped,'gt_closure_max_m':closure,
        'groups':{fov:{pose:aggregate([r for r in records if r['fov']==fov and r['pose']==pose]) for pose in POSES} for fov in ('original','expanded')},
        'paired':{fov:{pose:paired_summary([r for r in pairs if r['fov']==fov and r['to_pose']==pose]) for pose in POSES[1:]} for fov in ('original','expanded')},
        'by_navigation':{nav:{fov:{pose:aggregate([r for r in records if r['navigation']==nav and r['fov']==fov and r['pose']==pose]) for pose in POSES} for fov in ('original','expanded')} for nav in sorted({r['navigation'] for r in records})},
        'support_gain_groups':{fov:{group:paired_summary([r for r in pairs if r['fov']==fov and r['to_pose']=='inside_13' and
            ('increased' if r['support_gain_points']>=5 else 'decreased' if r['support_gain_points']<=-5 else 'similar')==group])
            for group in ('increased','similar','decreased')} for fov in ('original','expanded')},
        'notes':['Same World/navigation/quality, same GT world stations at all three poses.',
            'Positions linearly interpolated at GT future station indices. No nearest-path rematching of predictions.',
            'Boundary support is a 1.25m-threshold proxy using retained observations and diagnostic true source identities.',
            'Pose changes alter heading, remaining turn, Guide cropping and independently sampled perception noise. Associations are not causal.',
            'Common-window error is an auxiliary diagnostic, not a replacement for full 80m ADE.']}
    summary['no_direct_support_triplets']={}
    for nav in ('left','all'):
        rs=[]
        for r in pairs:
            if r['fov']!='original' or r['to_pose']!='inside_13' or (nav!='all' and r['navigation']!=nav):continue
            key=(r['world_id'],r['navigation'],r['quality'])
            if all(lookup[(*key,pose,'original')]['both_supported_points']+lookup[(*key,pose,'original')]['one_supported_points']==0 for pose in POSES):rs.append(r)
        summary['no_direct_support_triplets'][nav]=paired_summary(rs)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'per_pose.csv',records);write_csv(out/'paired.csv',pairs)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9));x=np.arange(3)
    for ax,metric,title in zip(axes.flat,('full_ade_m','common_error_m','lateral_error_m','both_supported_fraction'),
                              ('Full moving 80m ADE','Same world exit segment error','Same segment lateral error','Both-boundary support fraction')):
        for off,fov,color in [(-.18,'original','#7197bd'),(.18,'expanded','#75a883')]:
            bars=ax.bar(x+off,[summary['groups'][fov][p][metric] for p in POSES],width=.36,color=color,label=fov)
            ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
        ax.set(xticks=x,xticklabels=POSES,title=title,ylabel='Fraction' if 'fraction' in metric else 'Error (m)');ax.grid(axis='y',alpha=.2);ax.legend()
        ax.set_ylim(0,1 if 'fraction' in metric else max(summary['groups'][fov][p][metric] for fov in ('original','expanded') for p in POSES)*1.25)
    fig.suptitle(f'{len(contexts)} matched triplets; common exit window {args.start_m:g}-{args.end_m:g} m')
    fig.tight_layout();fig.savefig(out/'matched_comparison.png',dpi=140);plt.close(fig)
    chosen=[]
    left=[r for r in pairs if r['fov']=='original' and r['to_pose']=='inside_13' and r['navigation']=='left']
    for reason,row in [('Largest left-turn improvement',max(left,key=lambda r:r['error_improvement_m'])),
                       ('Left-turn persistent large error',max(left,key=lambda r:r['later_error_m'])),
                       ('Left-turn deterioration',min(left,key=lambda r:r['error_improvement_m']))]:
        key=(row['world_id'],row['navigation'],row['quality']);ctx=contexts[key];fig,axes=plt.subplots(1,3,figsize=(16,6))
        physical={e['id']:e for e in ctx['world']['physical']['elements']};truth=ctx['truth']
        stations=ctx['route']['exit_s']+ctx['offsets'];allpoints=[truth];paths={}
        for pose in POSES:
            i=ctx['index'][pose];md=raw[i]['metadata'];paths[pose]=at_world_stations(arrays['original'][i],md,stations);allpoints.append(paths[pose])
        extent=np.concatenate(allpoints)
        for ax,pose in zip(axes,POSES):
            i=ctx['index'][pose];md=raw[i]['metadata']
            for source in ctx['expected']:
                xy=np.asarray(physical[source]['points']);ax.plot(xy[:,0],xy[:,1],color='#b8c2ca',lw=2,label='True target boundary' if source==next(iter(ctx['expected'])) else None)
            for kind,obj in _objects(raw[i],32):
                xy=to_world(resample(obj['points'],32,closed=kind in ('stop_line','crosswalk')),md)
                if kind in ('stop_line','crosswalk'):xy=np.vstack([xy,xy[:1]])
                ax.plot(xy[:,0],xy[:,1],color='#555555',lw=.8,alpha=.6)
            guide=at_world_stations(arrays['guide'][i],md,stations)
            ax.plot(guide[:,0],guide[:,1],color='#e99b00',ls='--',label='Guide at matched indices')
            ax.plot(truth[:,0],truth[:,1],color='#159447',lw=2,label='Same GT exit segment')
            ax.plot(paths[pose][:,0],paths[pose][:,1],color='#d83737',lw=2,label='Prediction / original')
            v=lookup[(*key,pose,'original')];ax.set(title=f'{pose}: error {v["common_error_m"]:.2f}m; support {v["both_supported_points"]:.0f}/{v["checked_points"]}',
                xlim=(extent[:,0].min()-5,extent[:,0].max()+5),ylim=(extent[:,1].min()-5,extent[:,1].max()+5),xlabel='World x (m)',ylabel='World y (m)')
            ax.set_aspect('equal');ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.suptitle(f'{reason}: {key[0]} / {key[1]} / {key[2]}\nAll panels show identical world bounds and identical GT stations; gray = retained observations')
        fig.tight_layout();filename=f'case_{len(chosen)+1}.png';fig.savefig(out/filename,dpi=140);plt.close(fig)
        chosen.append({'reason':reason,**row,'plot':filename})
    (out/'plot_manifest.json').write_text(json.dumps(chosen,indent=2)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Matched pose progression</title>'
        '<style>body{font:16px system-ui;margin:30px}img{max-width:100%}</style><h1>Matched pose progression</h1>'
        '<p>Same world exit stations across poses; examples show original inputs only. Fixed checkpoint, no inference.</p>'
        '<img src="matched_comparison.png">'+''.join(f'<h2>{html.escape(r["reason"])}</h2><img src="{r["plot"]}">' for r in chosen),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k in ('matched_triplets','skipped','gt_closure_max_m','groups','paired','by_navigation','support_gain_groups')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-dir',type=Path,default=Path('outputs/context_comparison_inside13/mixed/pose_context_evaluation'))
    p.add_argument('--supplement-cache',type=Path,default=Path('outputs/context_comparison_inside13/fixed_val_supplements'))
    p.add_argument('--worlds-dir',type=Path,default=Path('outputs/v4_world300_inside13/worlds'))
    p.add_argument('--routes-dir',type=Path,default=Path('outputs/v4_world300_inside13/routes'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/context_comparison_inside13/mixed/pose_progression'))
    p.add_argument('--start-m',type=float,default=3.)
    p.add_argument('--end-m',type=float,default=30.)
    p.add_argument('--support-distance-m',type=float,default=1.25)
    p.add_argument('--min-points',type=int,default=5)
    args=p.parse_args()
    if not (0<=args.start_m<args.end_m and args.min_points>=2 and args.support_distance_m>0):p.error('Invalid window/threshold')
    run(args)
