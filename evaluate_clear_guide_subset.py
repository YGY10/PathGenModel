"""Matched control/mixed x standard/expanded comparison on audited clear guides.

Reuses saved predictions; does not alter guides, infer new paths or retrain.
Expanded metrics average errors over fixed seeds, never averaged trajectories.
"""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from Dataset.context_augmentation import file_sha256
from evaluate import EvaluationDataset
from evaluate_context_comparison import FixedInputs
from train import inspect_split


def error_metrics(error):
    return {'ade_m':float(error[1:].mean()),'fde_m':float(error[-1]),
            'ade_1_20m':float(error[1:21].mean()),'ade_21_40m':float(error[21:41].mean()),
            'ade_41_80m':float(error[41:81].mean())}


def aggregate(rows):
    fields=('ade_m','fde_m','ade_1_20m','ade_21_40m','ade_41_80m',
            'guide_ade_m','wrong_direction','outside_road','outside_target_lane')
    result={'samples':len(rows),'worlds':len({r['world_id'] for r in rows}),
            **{k:float(np.mean([r[k] for r in rows])) for k in fields}}
    result.update(ade_p50_m=float(np.median([r['ade_m'] for r in rows])),
                  ade_p90_m=float(np.quantile([r['ade_m'] for r in rows],.9)),
                  fraction_better_than_guide=float(np.mean([r['ade_m']<r['guide_ade_m'] for r in rows])))
    return result


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot_example(base,pool,pred,index,seed,path):
    import matplotlib.pyplot as plt
    raw=base.raw_row(index);gt=np.asarray(raw['gt_path']);guide=np.asarray(raw['reference_line'])
    encodings=[base[index],FixedInputs(base,pool)[index]]
    all_paths=[gt,guide]+[pred[f'{model}_{arm}'][index] for model in ('control','mixed')
                        for arm in ('original',f'expanded_seed_{seed}')]
    points=np.concatenate(all_paths)
    fig,axes=plt.subplots(2,2,figsize=(14,11))
    for row,model in enumerate(('control','mixed')):
        for col,arm in enumerate(('original',f'expanded_seed_{seed}')):
            ax=axes[row,col];enc=encodings[col];used=set()
            n_original=int(encodings[0]['polyline_mask'].sum())
            for j in range(32):
                mask=enc['point_mask'][j]
                if not mask.any():continue
                xy=enc['bev_points'][j,mask,:2].numpy()
                kind=int(enc['polyline_attr'][j,:4].argmax())
                color,label=(('#888888','Lane'),('#222222','Curb'),('#2984b9','Stop'),('#48b8c4','Crosswalk'))[kind]
                added=j>=n_original;label+=' added' if added else ' original'
                if enc['polyline_attr'][j,9]>.5:xy=np.concatenate([xy,xy[:1]])
                ax.plot(-xy[:,1],xy[:,0],color=color,ls='--' if added else '-',lw=1,alpha=.65,
                        label=label if label not in used else None);used.add(label)
            prediction=pred[f'{model}_{arm}'][index]
            for xy,color,ls,label in [(guide,'#e99b00','--','Guide'),(gt,'#159447','-','GT'),
                                      (prediction,'#d83737','-','Prediction')]:
                ax.plot(-xy[:,1],xy[:,0],color=color,ls=ls,label=label,lw=2)
            ax.scatter(0,0,c='black',marker='^',s=30)
            error=error_metrics(np.linalg.norm(prediction-gt,axis=-1))
            ax.set(title=f'{model} / {"standard" if col==0 else "expanded"}: ADE {error["ade_m"]:.3f} m, FDE {error["fde_m"]:.3f} m',
                   xlim=(-points[:,1].max()-4,-points[:,1].min()+4),ylim=(points[:,0].min()-4,points[:,0].max()+4),
                   xlabel='Right (-ego y), m',ylabel='Forward (ego x), m')
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=6,loc='lower center',fontsize=8)
    fig.suptitle(f'{raw["id"]}\nExpanded maps use fixed observation seed {seed}; summary averages three seeds.',fontsize=12)
    fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(path,dpi=130);plt.close(fig)


def run(args):
    audit=json.loads((args.audit_dir/'summary.json').read_text())
    comparison=json.loads((args.evaluation_dir/'summary.json').read_text())
    source=json.loads((args.source_dir/'summary.json').read_text())
    if inspect_split(source['input'])!=audit['data_manifest'] or audit['data_manifest']!=comparison['data_manifest']:
        raise ValueError('Dataset fingerprints differ')
    base=EvaluationDataset(source['input'],32,32)
    audit_rows=list(csv.DictReader((args.audit_dir/'per_sample.csv').open()))
    if len(audit_rows)!=len(base) or any(int(r['index'])!=i or r['id']!=base.raw_row(i)['id'] for i,r in enumerate(audit_rows)):
        raise ValueError('Audit sample identities differ')
    indices=[int(r['index']) for r in audit_rows if r['group']=='target_clear']
    if not indices:raise ValueError('Empty clear-guide subset')
    with np.load(args.evaluation_dir/'predictions.npz',allow_pickle=False) as archive:
        ids=archive['ids'].tolist();pred={k:archive[k] for k in archive.files if k!='ids'}
    if ids!=[r['id'] for r in audit_rows]:raise ValueError('Prediction identities differ')
    seeds=sorted(k.removeprefix('mixed_expanded_seed_') for k in pred if k.startswith('mixed_expanded_seed_'))
    if str(source['config']['seed']) not in seeds:raise ValueError('Plot source seed absent')
    saved={(r['model'],r['input'],int(r['index'])):r for r in csv.DictReader((args.evaluation_dir/'per_sample.csv').open())}
    rows,curves,seed_rows=[],{},[]
    for model in ('control','mixed'):
        for fov in ('standard','expanded'):
            arms=['original'] if fov=='standard' else [f'expanded_seed_{s}' for s in seeds]
            curves[model,fov]=[]
            for i in indices:
                raw=base.raw_row(i);gt=np.asarray(raw['gt_path']);guide=np.asarray(raw['reference_line'])
                errors=[]
                for arm in arms:
                    error=np.linalg.norm(pred[f'{model}_{arm}'][i]-gt,axis=-1)
                    checked=error_metrics(error)
                    if not np.isclose(checked['ade_m'],float(saved[model,arm,i]['ade_m']),atol=1e-6,rtol=0):
                        raise ValueError('Saved ADE and prediction arrays disagree')
                    if not np.isclose(checked['fde_m'],float(saved[model,arm,i]['fde_m']),atol=1e-6,rtol=0):
                        raise ValueError('Saved FDE and prediction arrays disagree')
                    errors.append(error)
                    seed_rows.append({'model':model,'input':arm,'index':i,'id':raw['id'],**checked})
                mean_error=np.mean(errors,axis=0);curves[model,fov].append(mean_error)
                rows.append({'model':model,'fov':fov,'index':i,'id':raw['id'],'world_id':raw['world_id'],
                    'navigation':raw['navigation'],'pose':raw['metadata']['pose_name'],'quality':raw['metadata']['quality'],
                    **error_metrics(mean_error),'guide_ade_m':error_metrics(np.linalg.norm(guide-gt,axis=-1))['ade_m'],
                    **{k:float(np.mean([saved[model,arm,i][k]=='True' for arm in arms]))
                       for k in ('wrong_direction','outside_road','outside_target_lane')}})
    summaries=[]
    for model in ('control','mixed'):
        for fov in ('standard','expanded'):
            selected=[r for r in rows if r['model']==model and r['fov']==fov]
            summaries.append({'model':model,'fov':fov,**aggregate(selected)})
    focused=[{'model':s['model'],'fov':s['fov'],**aggregate([r for r in rows if r['model']==s['model'] and
              r['fov']==s['fov'] and r['pose'] in ('entry','approach_10')])} for s in summaries]
    grouped={key:{value:[{'model':s['model'],'fov':s['fov'],**aggregate([r for r in rows if r[key]==value and
                r['model']==s['model'] and r['fov']==s['fov']])} for s in summaries]
                for value in sorted({r[key] for r in rows})} for key in ('pose','navigation','quality','world_id')}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'summary.csv',summaries);write_csv(out/'per_sample.csv',rows);write_csv(out/'per_seed_sample.csv',seed_rows)
    seed_summary=[]
    for model in ('control','mixed'):
        for arm in ['original']+[f'expanded_seed_{seed}' for seed in seeds]:
            rs=[r for r in seed_rows if r['model']==model and r['input']==arm]
            seed_summary.append({'model':model,'input':arm,'samples':len(rs),
                                 **{k:float(np.mean([r[k] for r in rs])) for k in ('ade_m','fde_m')}})
    write_csv(out/'per_seed_summary.csv',seed_summary)
    write_csv(out/'selected_samples.csv',[audit_rows[i] for i in indices])
    summary={'selection':'actual input guide group == target_clear; no guide changes',
        'audit_config':audit['config'],'audit_csv_sha256':file_sha256(args.audit_dir/'per_sample.csv'),
        'data_manifest':audit['data_manifest'],'checkpoints':{k:v['checkpoint'] for k,v in comparison['models'].items()},
        'supplement_seeds':seeds,'overall':summaries,'entry_and_approach10':focused,'groups':grouped,
        'per_seed_summary':seed_summary,
        'notes':['Same sample IDs for all four conditions. Expanded errors averaged across seeds, not trajectories.',
                 'Subset selected using Guide and true lane geometry only, without model-error selection.',
                 'Validation diagnostic subset, not a newly held-out benchmark; poses/qualities/seeds repeat the same worlds.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#7197bd','#b4c6d9','#d47b54','#75a883'];labels=['Control / standard','Control / expanded','Mixed / standard','Mixed / expanded']
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for ax,metric,title in [(axes[0,0],'ade_m','ADE'),(axes[0,1],'fde_m','FDE')]:
        bars=ax.bar(np.arange(4),[s[metric] for s in summaries],color=colors)
        ax.bar_label(bars,fmt='%.3f',padding=4);ax.set(xticks=np.arange(4),xticklabels=labels,ylabel='Error (m)',title=title)
        ax.tick_params(axis='x',labelsize=8);ax.set_ylim(0,max(s[metric] for s in summaries)*1.2);ax.grid(axis='y',alpha=.2)
    for s,label,color in zip(summaries,labels,colors):
        axes[1,0].plot(np.arange(1,81),np.mean(curves[s['model'],s['fov']],axis=0)[1:],label=label,color=color)
    axes[1,0].set(xlabel='GT path station index (1 m spacing)',ylabel='Mean position error (m)',title='Error by horizon')
    axes[1,0].legend(fontsize=8);axes[1,0].grid(alpha=.2)
    poses=list(grouped['pose']);x=np.arange(len(poses))
    for j,(label,color) in enumerate(zip(labels,colors)):
        axes[1,1].bar(x+(j-1.5)*.19,[grouped['pose'][p][j]['ade_m'] for p in poses],width=.19,color=color,label=label)
    axes[1,1].set(xticks=x,xticklabels=poses,ylabel='ADE (m)',title='By ego pose');axes[1,1].tick_params(axis='x',labelsize=8)
    axes[1,1].grid(axis='y',alpha=.2);axes[1,1].legend(fontsize=8)
    fig.suptitle(f'Clear-target-guide subset: {len(indices)} identical samples; expanded seeds {", ".join(seeds)}')
    fig.tight_layout();fig.savefig(out/'overview.png',dpi=140);plt.close(fig)
    lookup={(r['model'],r['fov'],r['index']):r for r in rows}
    ordered=sorted(indices,key=lambda i:lookup['mixed','expanded',i]['ade_m'])
    candidates=[('Worst mixed / expanded',ordered[-1]),('Median mixed / expanded',ordered[len(ordered)//2]),
        ('Largest mixed benefit under expanded FOV',max(indices,key=lambda i:lookup['control','expanded',i]['ade_m']-lookup['mixed','expanded',i]['ade_m'])),
        ('Largest mixed regression under expanded FOV',max(indices,key=lambda i:lookup['mixed','expanded',i]['ade_m']-lookup['control','expanded',i]['ade_m']))]
    pool=[json.loads(line) for line in (args.source_dir/'added_observations.jsonl').open()]
    if len(pool)!=len(base) or any(r['id']!=ids[i] for i,r in enumerate(pool)):raise ValueError('Plot supplement IDs differ')
    cards=[];rendered=set()
    for reason,i in candidates:
        filename=f'{i:06d}.png'
        if i not in rendered:plot_example(base,pool,pred,i,str(source['config']['seed']),out/filename);rendered.add(i)
        cards.append(f'<h2>{html.escape(reason)}</h2><p>{html.escape(ids[i])}</p><img src="{filename}">')
    (out/'plot_manifest.json').write_text(json.dumps([{'reason':reason,'index':i,'id':ids[i],'plot':f'{i:06d}.png'} for reason,i in candidates],indent=2)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Clear-guide comparison</title>'
        '<style>body{font:16px system-ui;margin:30px;background:#f7f7f9}img{max-width:100%}h2{margin-top:40px}</style>'
        '<h1>Clear-target-guide subset</h1><p>Same sample IDs in all four conditions. Summary: three fixed expanded seeds; '
        'example maps: fixed seed 724. Example ranking uses three-seed mean ADE.</p><img src="overview.png">'+''.join(cards),encoding='utf-8')
    print(json.dumps({'overall':summaries,'entry_and_approach10':focused,'output':str(out)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-dir',type=Path,default=Path('outputs/context_comparison/evaluation'))
    p.add_argument('--audit-dir',type=Path,default=Path('outputs/context_comparison/guide_lane_audit'))
    p.add_argument('--source-dir',type=Path,default=Path('outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/context_comparison/clear_guide_evaluation'))
    run(p.parse_args())
