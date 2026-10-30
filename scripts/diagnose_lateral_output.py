"""Diagnose exact same-index normal-only output floor; never train/change model.

Uses current model Guide finite differences, then normalizes float64 basis.
GT privileged projection is a representation diagnostic, not a trained predictor.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Model.pathgen_model_v1 import PathGenModel
from Dataset.audit_lane_visibility import point_polyline_distance


def decompose(guide,gt):
    g32=np.asarray(guide,dtype=np.float32);y32=np.asarray(gt,dtype=np.float32)
    if g32.ndim!=3 or g32.shape!=y32.shape or g32.shape[-1]!=2:
        raise ValueError('Guide and GT must have equal [B,T,2] shapes')
    if not np.isfinite(g32).all() or not np.isfinite(y32).all():raise ValueError('Non-finite path')
    with torch.inference_mode():
        t=PathGenModel.guide_features(torch.from_numpy(g32.copy()))[:,:,2:4].numpy().astype(np.float64)
    length=np.linalg.norm(t,axis=-1,keepdims=True)
    if np.any(length<1e-6):raise ValueError('Degenerate Guide tangent; no defined normal-only basis')
    t/=length;n=np.stack([-t[...,1],t[...,0]],axis=-1)
    g=g32.astype(np.float64);y=y32.astype(np.float64);r=y-g
    ds=np.sum(r*t,axis=-1);dl=np.sum(r*n,axis=-1)
    ideal=g+dl[...,None]*n
    ideal[:,0]=0.
    reconstruction=g+ds[...,None]*t+dl[...,None]*n
    error=np.linalg.norm(ideal[:,1:]-y[:,1:],axis=-1)
    if not np.allclose(error,np.abs(ds[:,1:]),atol=1e-10,rtol=1e-10):raise AssertionError('Projection-floor identity failed')
    if np.max(np.abs(reconstruction-y))>1e-10:raise AssertionError('Two-component reconstruction failed')
    return {'guide':g,'gt':y,'tangent':t,'normal':n,'ds':ds,'dl':dl,'ideal_l_only':ideal}


def resampling_diagnostic(ideal,gt):
    """Auxiliary projection-then-resample, only when 80m is actually available.

    This is not a lower bound for arbitrary normal-offset curves: dl here was
    selected for equal-index ADE, not optimized for the reparameterized curve.
    """
    lengths=[];before=[];after=[]
    for p,y in zip(ideal,gt):
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        lengths.append(float(arc[-1]))
        if arc[-1]<80.:continue
        q=np.column_stack([np.interp(np.arange(81),arc,p[:,k]) for k in (0,1)])
        before.append(float(np.linalg.norm(p[1:]-y[1:],axis=1).mean()))
        after.append(float(np.linalg.norm(q[1:]-y[1:],axis=1).mean()))
    short=[80-v for v in lengths if v<80.]
    return {'available_cases':len(ideal),'cases_with_at_least_80m_projected_polyline':len(after),
        'fraction':len(after)/len(ideal),
        'original_projection_ade_in_covered_subset_m':float(np.mean(before)) if before else None,
        'resampled_ade_in_covered_subset_m':float(np.mean(after)) if after else None,
        'resampled_ade_p50_p90_m':np.quantile(after,[.5,.9]).tolist() if after else None,
        'projected_polyline_length_min_p50_max_m':[float(np.min(lengths)),float(np.median(lengths)),float(np.max(lengths))],
        'insufficient_length_mean_deficit_m':float(np.mean(short)) if short else 0.,
        'note':'GT-perfect-dl projection resampled at0..80m ONLY where length>=80m. No extension/clamping. Not a theoretical optimum or full-population/model result.'}


def distribution(values):
    a=np.asarray(values).reshape(-1)
    return {'mean_signed_m':float(a.mean()),'std_m':float(a.std()),
        'signed_p05_p50_p95_m':np.quantile(a,[.05,.5,.95]).tolist(),
        'mean_abs_m':float(np.abs(a).mean()),'abs_p50_p90_p95_p99_m':np.quantile(abs(a),[.5,.9,.95,.99]).tolist(),
        'max_abs_m':float(abs(a).max()),'fraction_abs_over_5cm':float(np.mean(abs(a)>.05))}


def aggregate(indices,rows,result,shape_metrics=None):
    idx=np.asarray(indices,dtype=int)
    if not len(idx):return {'samples':0}
    ds=result['ds'][idx,1:];dl=result['dl'][idx,1:];floor=abs(ds).mean(-1)
    gt=result['gt'][idx];guide=result['guide'][idx]
    den=float((ds*ds+dl*dl).sum())
    answer={'samples':len(idx),'worlds':len({rows[i]['world_id'] for i in idx}),
        'routes':len({(rows[i]['world_id'],rows[i]['navigation']) for i in idx}),
        'ds':distribution(ds),'dl':distribution(dl),
        'normal_only_min_ade_m':float(floor.mean()),
        'normal_only_min_fde_m':float(abs(result['ds'][idx,-1]).mean()),
        'sample_floor_p50_p90_p95_m':np.quantile(floor,[.5,.9,.95]).tolist(),
        'fraction_samples_floor_le_5cm':float(np.mean(floor<=.05)),
        'fraction_samples_floor_le_10cm':float(np.mean(floor<=.1)),
        'longitudinal_squared_residual_share':float((ds*ds).sum()/den) if den else 0.,
        'guide_ade_m':float(np.linalg.norm(gt[:,1:]-guide[:,1:],axis=-1).mean()),
        'floor_by_distance_m':{'1_20':float(abs(result['ds'][idx,1:21]).mean()),
                              '21_40':float(abs(result['ds'][idx,21:41]).mean()),
                              '41_80':float(abs(result['ds'][idx,41:81]).mean())}}
    if shape_metrics is not None:
        answer.update({key:float(np.mean([shape_metrics[i][key] for i in idx])) for key in shape_metrics[0]})
    return answer


def shape_distances(result):
    """Auxiliary nearest-polyline metric, NOT a replacement for equal-index ADE."""
    rows=[]
    for p,y in zip(result['ideal_l_only'],result['gt']):
        forward=float(point_polyline_distance(p[1:],y).mean())
        backward=float(point_polyline_distance(y[1:],p).mean())
        rows.append({'ideal_to_gt_polyline_mean_m':forward,'gt_to_ideal_polyline_mean_m':backward,
                     'symmetric_polyline_mean_m':.5*(forward+backward)})
    return rows


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def read_split(path,split):
    meta=[];guides=[];gt=[]
    for line in path.open():
        row=json.loads(line)
        meta.append({'id':row['id'],'source_id':row['metadata'].get('oracle_source_sample_id',row['id']),
            'world_id':row['world_id'],'navigation':row['navigation'],'pose':row['metadata']['pose_name'],
            'split':split,'ego_s_m':row['metadata']['ego_route_s_m']})
        guides.append(row['reference_line']);gt.append(row['gt_path'])
    return meta,np.asarray(guides,dtype=np.float32),np.asarray(gt,dtype=np.float32)


def analyze_split(meta,guide,gt,out):
    r=decompose(guide,gt);shape=shape_distances(r)
    summary={'all':aggregate(range(len(meta)),meta,r,shape),
        'by_navigation':{name:aggregate([i for i,x in enumerate(meta) if x['navigation']==name],meta,r,shape) for name in ('left','right','straight')},
        'by_pose':{name:aggregate([i for i,x in enumerate(meta) if x['pose']==name],meta,r,shape) for name in sorted({x['pose'] for x in meta})},
        'by_navigation_pose':{nav+'/'+pose:aggregate([i for i,x in enumerate(meta) if x['navigation']==nav and x['pose']==pose],meta,r,shape) for nav in ('left','right','straight') for pose in sorted({x['pose'] for x in meta})}}
    records=[]
    for i,x in enumerate(meta):
        records.append({**x,'normal_only_min_ade_m':float(abs(r['ds'][i,1:]).mean()),
            'mean_abs_dl_m':float(abs(r['dl'][i,1:]).mean()),'ds_signed_mean_m':float(r['ds'][i,1:].mean()),
            'normal_only_min_fde_m':float(abs(r['ds'][i,-1])),**shape[i]})
    write_csv(out/f'{meta[0]["split"]}_per_sample.csv',records)
    np.savez_compressed(out/f'{meta[0]["split"]}_projections.npz',ids=np.array([x['id'] for x in meta]),
        guide=r['guide'].astype(np.float32),gt=r['gt'].astype(np.float32),
        ds=r['ds'].astype(np.float32),dl=r['dl'].astype(np.float32),ideal_l_only=r['ideal_l_only'].astype(np.float32))
    return summary,r,records


def model_errors(path,rows,result):
    if not path.exists():return {'available':False}
    data=np.load(path,allow_pickle=False);lookup={x['id']:i for i,x in enumerate(rows)}
    indices=[lookup[str(v)] for v in data['ids']]
    p=np.asarray(data['pred'],dtype=np.float64);g=result['guide'][indices];y=result['gt'][indices]
    assert np.max(abs(data['gt']-y))<1e-5
    assert np.max(abs(data['guide']-g))<1e-5
    t=result['tangent'][indices];n=result['normal'][indices];e=p-y
    dl_pred=np.sum((p-g)*n,axis=-1);clamped=g+dl_pred[...,None]*n;clamped[:,0]=0
    return {'available':True,'samples':len(indices),'ade_m':float(np.linalg.norm(e[:,1:],axis=-1).mean()),
        'guide_frame_longitudinal_error_mae_m':float(abs(np.sum(e*t,axis=-1)[:,1:]).mean()),
        'guide_frame_lateral_error_mae_m':float(abs(np.sum(e*n,axis=-1)[:,1:]).mean()),
        'drop_predicted_ds_ade_m':float(np.linalg.norm(clamped[:,1:]-y[:,1:],axis=-1).mean()),
        'normal_only_min_ade_m':float(abs(result['ds'][indices,1:]).mean())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('outputs/oracle_lane_v1'))
    ap.add_argument('--real-source',type=Path,default=Path('outputs/v4_world300_inside13'))
    ap.add_argument('--guide-cache',type=Path,default=Path('outputs/guide_train_cache_v1'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/lateral_output_diagnostic')/datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=ap.parse_args();torch.set_num_threads(4);out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    summary={'definition':'Exact pointwise best GT projection onto each Guide normal. Same81points, fixed origin, ADE excludes origin. No model fitting or path reparameterization.',
        'basis':'Current PathGenModel.guide_features centered finite differences (endpoints one-sided), float32 input; float64 unit renormalization.',
        'splits':{},'checks':{},'model_errors':{}}
    datasets={}
    for split in ('train','val'):
        meta,g,y=read_split(args.source/'audit'/f'{split}.jsonl',split)
        sub,r,records=analyze_split(meta,g,y,out);summary['splits'][split]=sub;datasets[split]=(meta,r,records)
        original={(x['world_id'],x['navigation'],x['pose']):i for i,x in enumerate(meta)};matches=0
        for line in (args.real_source/'audit'/f'{split}.jsonl').open():
            row=json.loads(line)
            i=original[row['world_id'],row['navigation'],row['metadata']['pose_name']]
            assert np.array_equal(np.asarray(row['reference_line'],dtype=np.float32),g[i])
            assert np.array_equal(np.asarray(row['gt_path'],dtype=np.float32),y[i]);matches+=1
        assert matches>=len(meta);summary['checks'][split+'_real_and_oracle_same_gt_guide']=matches
        summary['model_errors']['oracle_'+split]=model_errors(Path('outputs/training_oracle_v1/full')/('eval_'+split)/'predictions.npz',meta,r)
        print(split,sub['all']['normal_only_min_ade_m'],flush=True)
    # Training Guide pool is a separate left/right-only conditional population.
    if (args.guide_cache/'guides.npy').exists():
        keys=json.loads((args.guide_cache/'keys.json').read_text());cache=np.load(args.guide_cache/'guides.npy',mmap_mode='r')
        meta,r,_=datasets['train'];lookup={(x['world_id'],x['navigation'],x['pose']):i for i,x in enumerate(meta)}
        ds_all=[];dl_all=[];poolmeta=[]
        for start in range(0,len(keys),128):
            selected=keys[start:start+128];slots=cache.shape[1]
            gt=np.stack([r['gt'][lookup[tuple(k)]] for k in selected])
            g=np.asarray(cache[start:start+128]).reshape(-1,81,2)
            gt=np.repeat(gt,slots,axis=0);d=decompose(g,gt)
            ds_all.append(d['ds']);dl_all.append(d['dl'])
            poolmeta.extend({'world_id':key[0],'navigation':key[1],'pose':key[2]} for key in selected for _ in range(slots))
        ds=np.concatenate(ds_all);dl=np.concatenate(dl_all)
        summary['augmented_training_pool']={'cases':len(ds),'ds':distribution(ds[:,1:]),'dl':distribution(dl[:,1:]),
            'normal_only_min_ade_m':float(abs(ds[:,1:]).mean()),
            'by_navigation':{nav:float(abs(ds[np.array([x['navigation']==nav for x in poolmeta]),1:]).mean()) for nav in ('left','right')}}
        floors=abs(r['ds'][:,1:]).mean(1)
        pool_index={tuple(k):i for i,k in enumerate(keys)};augfloors=abs(ds[:,1:]).mean(1).reshape(len(keys),cache.shape[1]).mean(1)
        expected=[]
        for i,x in enumerate(meta):
            key=(x['world_id'],x['navigation'],x['pose'])
            expected.append(.5*floors[i]+.5*augfloors[pool_index[key]] if key in pool_index else floors[i])
        summary['expected_current_training_mixture_floor_ade_m']=float(np.mean(expected))
    summary['checks']['train_val_world_overlap']=sorted({x['world_id'] for x in datasets['train'][0]}&{x['world_id'] for x in datasets['val'][0]})
    summary['resampling_diagnostic']=resampling_diagnostic(datasets['val'][1]['ideal_l_only'],datasets['val'][1]['gt'])
    (out/'resampling_coverage.json').write_text(json.dumps(summary['resampling_diagnostic'],indent=2)+'\n')
    plot(out,datasets)
    summary['limits']=['This floor constrains same-index Guide+delta_l*normal only; does not constrain XY,delta_s+delta_l,or normal-offset curves followed by arc-length resampling.',
        'GT-perfect per-point offsets are privileged, independently chosen, unconstrained scalars; continuity/kinematic/loss constraints cannot improve this floor.',
        'Nearest-polyline distances are shape-only auxiliary metrics; cannot replace full80m equal-station ADE.',
        'No model or training changes. Original Guide/GT shared across quality and Oracle; each geometric pose counted once. Test split untouched.']
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('OUTPUT',out)
    print(json.dumps({split:summary['splits'][split]['all'] for split in ('train','val')},indent=2))
    print('MODELS',json.dumps(summary['model_errors'],indent=2))


def plot(out,datasets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta,r,records=datasets['val'];fig,axes=plt.subplots(1,3,figsize=(16,5))
    for nav,color in [('left','#d0675b'),('right','#428bb5'),('straight','#4a9c6c')]:
        idx=[i for i,x in enumerate(meta) if x['navigation']==nav];values=abs(r['ds'][idx,1:]).mean(1)
        axes[0].hist(values,bins=30,alpha=.45,label=nav,color=color)
    axes[0].axvline(.05,color='black',ls='--',label='5cm target');axes[0].set(xlabel='Best possible same-index delta-l-only ADE (m)',ylabel='Samples');axes[0].legend()
    poses=sorted({x['pose'] for x in meta})
    axes[1].bar(poses,[np.mean([x['normal_only_min_ade_m'] for x in records if x['pose']==p]) for p in poses]);axes[1].tick_params(axis='x',rotation=35);axes[1].set_ylabel('Exact same-index floor ADE (m)')
    axes[2].scatter([x['normal_only_min_ade_m'] for x in records],[x['symmetric_polyline_mean_m'] for x in records],s=8,alpha=.5)
    axes[2].set(xlabel='Equal-index floor ADE (m)',ylabel='Symmetric nearest-polyline distance (m)',title='Point alignment vs curve geometry')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'distributions.png',dpi=150);plt.close(fig)
    picks=[]
    for nav in ('right','left','straight'):
        idx=[i for i,x in enumerate(meta) if x['navigation']==nav and x['pose']=='entry']
        idx.sort(key=lambda i:records[i]['normal_only_min_ade_m']);picks.append(idx[len(idx)//2])
    fig,axes=plt.subplots(1,3,figsize=(18,7))
    for ax,i in zip(axes,picks):
        paths=[(r['guide'][i],'Guide','#e99a00','--'),(r['gt'][i],'GT','#159447','-'),(r['ideal_l_only'][i],'Perfect delta l','#db3030','-')]
        for p,label,color,style in paths:ax.plot(-p[:,1],p[:,0],label=label,color=color,ls=style,lw=2)
        for j in range(5,51,5):
            a=r['ideal_l_only'][i,j];b=r['gt'][i,j];ax.plot([-a[1],-b[1]],[a[0],b[0]],color='#888888',lw=.8)
        ax.set_aspect('equal');ax.legend();ax.grid(alpha=.2);ax.set(title=f'{meta[i]["navigation"]}/entry\nExact floor ADE={records[i]["normal_only_min_ade_m"]:.3f}m',xlabel='Right (-y), m',ylabel='Forward (x), m')
    fig.tight_layout();fig.savefig(out/'ideal_projection_examples.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
