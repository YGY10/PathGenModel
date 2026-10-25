"""Evaluate one checkpoint by pose on fixed original and cached expanded inputs.

Does not require the matched control run to finish. Reuses frozen supplementary
cache for later matched-model evaluation. Expanded metrics average errors, not paths.
"""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import torch

from analyze_context_failures import plot_case
from Dataset.context_augmentation import inspect_context_cache, file_sha256
from evaluate import EvaluationDataset, load_model
from evaluate_context_comparison import FixedInputs, inference
from evaluate_clear_guide_subset import error_metrics
from Scenario.build_dataset import V4_POSES
from train import inspect_split


def aggregate(rows):
    keys=('ade_m','fde_m','ade_1_20m','ade_21_40m','ade_41_80m','guide_ade_m','guide_fde_m')
    return {'samples':len(rows),'worlds':len({r['world_id'] for r in rows}),
        **{k:float(np.mean([r[k] for r in rows])) for k in keys},
        'ade_p90_m':float(np.quantile([r['ade_m'] for r in rows],.9)),
        'fraction_better_than_guide':float(np.mean([r['ade_m']<r['guide_ade_m'] for r in rows])),
        'regressed_over_10cm_fraction':float(np.mean([r['ade_m']>r['guide_ade_m']+.1 for r in rows]))}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(args):
    standard=json.loads((args.standard_evaluation/'summary.json').read_text())
    input_path=Path(standard['input']);info=inspect_split(input_path)
    device=torch.device(args.device)
    model,saved=load_model(args.checkpoint,device)
    if info!=standard['data_manifest'] or info!=saved['splits']['val']:
        raise ValueError('Validation identity mismatch')
    if set(info['worlds'])&set(saved['splits']['train']['worlds']):
        raise ValueError('Train/validation World overlap')
    if Path(standard['checkpoint']).resolve()!=args.checkpoint.resolve() or standard['epoch']!=saved['epoch']:
        raise ValueError('Standard evaluation checkpoint differs')
    if (saved['config']['max_lines'],saved['config']['max_points'])!=(32,32):
        raise ValueError('Expected 32 x 32 input capacity')
    fingerprints=inspect_context_cache(args.supplement_cache,info,32,32)
    manifest=json.loads((args.supplement_cache/'manifest.json').read_text())
    base=EvaluationDataset(input_path,32,32)
    with np.load(args.standard_evaluation/'predictions.npz',allow_pickle=False) as archive:
        ids=archive['ids'].tolist();gt=archive['gt'];guide=archive['guide'];original=archive['pred']
    raw=[base.raw_row(i) for i in range(len(base))]
    if ids!=[r['id'] for r in raw]:raise ValueError('Prediction ID mismatch')
    predictions={'original':original};pools={}
    for entry in manifest['files']:
        seed=entry['seed'];pool=[json.loads(line) for line in (args.supplement_cache/entry['path']).open()]
        if [r['id'] for r in pool]!=ids:raise ValueError('Supplement identity mismatch')
        pools[seed]=pool
        predictions[f'expanded_seed_{seed}']=inference(model,FixedInputs(base,pool),device,args.batch_size)
        print(f'Inferred fixed expanded seed {seed}: {len(base)} samples',flush=True)
    errors={key:np.linalg.norm(value-gt,axis=-1) for key,value in predictions.items()}
    guide_error=np.linalg.norm(guide-gt,axis=-1)
    expanded_error=np.mean([value for key,value in errors.items() if key!='original'],axis=0)
    rows=[]
    for fov,error in [('original',errors['original']),('expanded',expanded_error)]:
        for i,r in enumerate(raw):
            g=error_metrics(guide_error[i])
            rows.append({'index':i,'id':r['id'],'world_id':r['world_id'],'pose':r['metadata']['pose_name'],
                         'navigation':r['navigation'],'quality':r['metadata']['quality'],'fov':fov,
                         **error_metrics(error[i]),'guide_ade_m':g['ade_m'],'guide_fde_m':g['fde_m']})
    overall={fov:aggregate([r for r in rows if r['fov']==fov]) for fov in ('original','expanded')}
    poses=[p for p in V4_POSES if any(r['pose']==p for r in rows)]
    poses+=sorted({r['pose'] for r in rows}-set(poses))
    grouped={p:{fov:aggregate([r for r in rows if r['pose']==p and r['fov']==fov])
                for fov in ('original','expanded')} for p in poses}
    seed_groups=[]
    for key,error in errors.items():
        for pose in ['all',*poses]:
            index=[i for i,r in enumerate(raw) if pose=='all' or r['metadata']['pose_name']==pose]
            seed_groups.append({'input':key,'pose':pose,'samples':len(index),
                'ade_m':float(error[index,1:].mean()),'fde_m':float(error[index,-1].mean())})
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'per_sample.csv',rows)
    write_csv(out/'by_pose.csv',[{'pose':p,'fov':fov,**grouped[p][fov]} for p in poses for fov in ('original','expanded')])
    write_csv(out/'per_seed_by_pose.csv',seed_groups)
    np.savez_compressed(out/'predictions.npz',ids=np.array(ids),gt=gt,guide=guide,**predictions)
    history=[json.loads(line) for line in (args.checkpoint.parent/'history.jsonl').open()]
    best_log=min(history,key=lambda r:r['val']['ade_m'])
    if not np.isclose(overall['original']['ade_m'],standard['overall']['ade_m'],atol=1e-6):
        raise ValueError('Standard metrics do not reproduce')
    summary={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':file_sha256(args.checkpoint),
        'epoch':saved['epoch'],'data_manifest':info,'supplement_cache':str(args.supplement_cache.resolve()),
        'supplement_fingerprints':fingerprints,'fov_ranges':manifest['ranges'],'seeds':list(pools),
        'overall':overall,'by_pose':grouped,
        'pose_navigation':{p:{nav:{fov:aggregate([r for r in rows if r['pose']==p and r['navigation']==nav and r['fov']==fov])
            for fov in ('original','expanded')} for nav in sorted({r['navigation'] for r in rows if r['pose']==p})} for p in poses},
        'training':{'best_epoch':best_log['epoch'],'last_epoch':history[-1]['epoch'],
                    'best_train_ade_m':best_log['train']['ade_m'],'best_val_ade_m':best_log['val']['ade_m'],
                    'last_train_ade_m':history[-1]['train']['ade_m'],'last_val_ade_m':history[-1]['val']['ade_m']},
        'notes':['Full validation set; no Guide-ambiguity filtering.',
            'Same original observations, labels and 32 slots in every arm; supplements only fill padding.',
            'Expanded errors averaged per sample across fixed perception seeds; no trajectory averaging.',
            'Same checkpoint and Worlds across poses; repeated noise seeds are not independent Worlds.',
            'Best checkpoint selected using original validation ADE; control model not part of this report.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,5));x=np.arange(len(poses))
    for ax,key,title in zip(axes,('ade_m','fde_m'),('ADE by pose','FDE by pose')):
        for offset,fov,color in [(-.19,'original','#7197bd'),(.19,'expanded','#75a883')]:
            bars=ax.bar(x+offset,[grouped[p][fov][key] for p in poses],width=.38,color=color,label=fov)
            ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
        ax.set(xticks=x,xticklabels=poses,ylabel='Error (m)',title=title)
        ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle(f'Mixed best epoch {saved["epoch"]}: original vs expanded seeds {list(pools)}')
    fig.tight_layout();fig.savefig(out/'pose_comparison.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for ax,pose in zip(axes.flat,poses):
        index=[i for i,r in enumerate(raw) if r['metadata']['pose_name']==pose]
        for error,color,label in [(errors['original'],'#7197bd','Original'),(expanded_error,'#75a883','Expanded'),(guide_error,'#e99b00','Guide')]:
            ax.plot(np.arange(1,81),error[index,1:].mean(0),color=color,label=label)
        ax.set(title=pose,xlabel='GT station index (1 m)',ylabel='Mean error (m)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(poses):]:ax.axis('off')
    fig.tight_layout();fig.savefig(out/'error_by_horizon.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot([r['epoch'] for r in history],[r['train']['ade_m'] for r in history],label='Train (mixed inputs)')
    ax.plot([r['epoch'] for r in history],[r['val']['ade_m'] for r in history],label='Validation (original inputs)')
    ax.axvline(saved['epoch'],ls='--',c='gray',label='Selected best');ax.set(xlabel='Epoch',ylabel='ADE (m)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'training_curve.png',dpi=140);plt.close(fig)
    arrays={'mixed_original':original,**{f'mixed_{k}':v for k,v in predictions.items() if k!='original'}}
    selected=[];cards=[];shown_seed=next(iter(pools))
    for pose in poses:
        rs=[r for r in rows if r['pose']==pose and r['fov']=='expanded']
        worst=max(rs,key=lambda r:r['ade_m']);i=worst['index'];filename=f'{i:06d}.png'
        plot_case(base,pools[shown_seed],arrays,i,list(pools),shown_seed,out/filename)
        selected.append({'pose':pose,'index':i,'id':worst['id'],'mean_expanded_ade_m':worst['ade_m'],'plot':filename})
        cards.append(f'<h2>{html.escape(pose)}: worst mean expanded ADE {worst["ade_m"]:.3f} m</h2><img src="{filename}">')
    (out/'plot_manifest.json').write_text(json.dumps(selected,indent=2)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mixed by pose</title>'
        '<style>body{font:16px system-ui;margin:30px}img{max-width:100%}</style><h1>Mixed model by pose</h1>'
        '<p>Full validation set. Expanded metrics average fixed seeds; maps show first seed, residual curves show all seeds.</p>'
        '<img src="pose_comparison.png"><img src="error_by_horizon.png"><img src="training_curve.png">'+''.join(cards),encoding='utf-8')
    print(json.dumps({'overall':overall,'by_pose':grouped,'output':str(out)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--standard-evaluation',type=Path,required=True)
    p.add_argument('--supplement-cache',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    if args.batch_size<1:p.error('batch-size must be positive')
    run(args)
