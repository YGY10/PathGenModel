"""Small qualitative review of the existing perception simulator; no training.

Fixed geometry/pose/GT/Guide and common FOV isolate observation differences.
Two deterministic draws per quality are kept without selection or rejection.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from dataclasses import asdict,replace
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import LineString,Point
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Scenario.build_dataset_v4 import build_one,observe,read_qualities,sample_world_guide
from Scenario.build_dataset import local_pose,to_ego,sample_path,arclength
from Scenario.build_lane_transition import make_urban_world
from World.intersection_generator import IntersectionConfig
from infer import encode_input,prepare_sample
FIELDS=('lane_markings','road_boundaries','stop_lines','crosswalks')
LABELS={'baseline':'无噪声观测基准','clean':'轻度','medium':'中度','hard':'较强'}
COLORS=('#3679cb','#273b51','#c95959','#31a6ad')

def serial(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 raise TypeError(type(value).__name__)

def write(path,value):
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,default=serial).encode()).hexdigest()

def make_case(name,kind,seed,nav,preset=None):
 if kind=='intersection':
  world,_,routes=build_one(seed,IntersectionConfig(),{})
  route=routes[nav];station=route['entry_s']-5
  title='路口左转（进入前5m）' if nav=='left' else '路口右转（进入前5m）'
  params={'entry_lane':route['entry_lane_index'],'exit_lane':route['exit_lane_index']}
 else:
  cfg,world,routes=make_urban_world(seed,preset,marking_pattern='unmarked')
  assert cfg.lanes_out>cfg.lanes_in
  entry=cfg.lanes_in-1
  route=next(r for r in routes.values() if r['entry_lane_index']==entry and r['navigation']==nav)
  station=float(np.interp(0,np.asarray(route['gt_world'])[:,0],arclength(np.asarray(route['gt_world']))))-8
  title=f'{cfg.lanes_in}→{cfg.lanes_out} '+('右侧增道' if cfg.change_side=='right' else '左侧增道')+'（过渡前8m）'
  params=asdict(cfg)
 gt_world=np.asarray(route['gt_world']);ego,heading=local_pose(gt_world,station)
 gt=to_ego(sample_path(gt_world,station),ego,heading)
 guide,_=sample_world_guide(np.asarray(route['guide_world']),ego,heading)
 truth=[]
 for e in world['physical']['elements']:
  truth.append({**e,'points':to_ego(np.asarray(e['points']),ego,heading).tolist()})
 gap=world['metadata'].get('unmarked_core_world_x_m')
 blank=None
 if isinstance(gap,(list,tuple)) and len(gap)==2:
  blank=to_ego(np.array([[gap[0],-40],[gap[1],-40],[gap[1],40],[gap[0],40]]),ego,heading).tolist()
 return {'id':name,'title':title,'kind':kind,'seed':seed,'navigation':nav,'world':world,
  'route':route,'ego':ego,'heading':heading,'gt':gt,'guide':guide,'truth':truth,'blank_polygon':blank,'road_parameters':params}

def row_for(case,q,seed):
 lines,curbs,stops,cross=observe(case['world'],case['ego'],case['heading'],q,np.random.default_rng(seed),max_lines=None)
 row={'id':case['id'],'navigation':case['navigation'],'coordinate_system':'ego_at_start_x_forward_y_left_m',
  'reference_line':case['guide'].tolist(),'gt_path':case['gt'].tolist(),
  'lane_markings':lines,'road_boundaries':curbs,'stop_lines':stops,'crosswalks':cross}
 return prepare_sample(row)

def encoded_lines(encoded):
 out=[]
 for i,valid in enumerate(encoded['polyline_mask']):
  if valid:
   attr=encoded['polyline_attr'][i];out.append({'kind':int(attr[:4].argmax()),'points':encoded['bev_points'][i,encoded['point_mask'][i],:2].numpy(),'attr':attr.numpy()})
 return out

def stats(row,encoded,truth):
 curves=encoded_lines(encoded);distances=[]
 physical={e['id']:LineString(e['points']) for e in truth if e['type'] in ('line','curb')}
 source_ids=set()
 for key in FIELDS[:2]:
  for e in row[key]:
   id=e['source_id'];source_ids.add(id)
   distances.extend(physical[id].distance(Point(p)) for p in e['points'])
 return {'supplied_objects':{k:len(row[k]) for k in FIELDS},
  'encoded_objects':{k:sum(e['kind']==i for e in curves) for i,k in enumerate(FIELDS)},
  'raw_source_ids':sorted(source_ids),'retained_objects':len(curves),
  'discarded_due_to_capacity':sum(len(row[k]) for k in FIELDS)-len(curves),
  'open_segment_length_m':float(sum(LineString(e['points']).length for e in curves if e['kind']<2)),
  'raw_point_deviation_mean_m':float(np.mean(distances)) if distances else None,
  'raw_point_deviation_p90_m':float(np.quantile(distances,.9)) if distances else None}

def setup_plot():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib import font_manager
 for path in font_manager.findSystemFonts():
  if 'NotoSansCJK' in path:
   prop=font_manager.FontProperties(fname=path);plt.rcParams['font.family']=prop.get_name();break
 plt.rcParams['axes.unicode_minus']=False
 return plt

def panel(ax,case,item,reference=False,limits=(-27,27,-18,90)):
 from matplotlib.patches import Rectangle,Polygon as Patch
 # This grey context is physical annotation, never an extra model input.
 for e in case['truth']:
  p=np.asarray(e['points']);closed=e['type'] in ('crosswalk','stop_line')
  p=np.r_[p,p[:1]] if closed else p
  ax.plot(-p[:,1],p[:,0],c='#d4dbe3',ls='--',lw=.75,zorder=1)
 if case['blank_polygon'] is not None:
  p=np.asarray(case['blank_polygon']);ax.add_patch(Patch(np.c_[-p[:,1],p[:,0]],color='#f7d989',alpha=.18,zorder=0))
 if reference:
  for e in case['truth']:
   p=np.asarray(e['points']);closed=e['type'] in ('crosswalk','stop_line');p=np.r_[p,p[:1]] if closed else p
   kind={'line':0,'curb':1,'stop_line':2,'crosswalk':3}[e['type']]
   ax.plot(-p[:,1],p[:,0],c=COLORS[kind],lw=1.25,zorder=2)
 else:
  for e in encoded_lines(item['encoded']):
   p=e['points'];p=np.r_[p,p[:1]] if e['kind']>=2 else p
   ax.plot(-p[:,1],p[:,0],c=COLORS[e['kind']],lw=1.65,zorder=3)
 q=item['quality'];ax.add_patch(Rectangle((-q.right_m,-q.rear_m),q.right_m+q.left_m,q.forward_m+q.rear_m,fill=False,edgecolor='#9a71b7',ls=':',lw=1.2,zorder=4))
 for p,color,style in [(case['gt'],'#1a995e','-'),(case['guide'],'#e4a320','--')]:ax.plot(-p[:,1],p[:,0],c=color,ls=style,lw=1.4,zorder=5)
 ax.scatter([0],[0],c='#1a2a3c',marker='^',s=25,zorder=6)
 ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='向右 −y（m）',ylabel='向前 x（m）')
 ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.13)
 text=item['title']
 if not reference:
  s=item['stats'];text+=f"\n保留 {s['retained_objects']} 个元素；线段总长 {s['open_segment_length_m']:.0f}m"
 ax.set_title(text,fontsize=10)

def draw_case(case,items,folder,plt):
 from matplotlib.lines import Line2D
 reference={**items[0],'title':'物理真值（当前视窗，非感知）'}
 panels=[(reference,True)]+[(item,False) for item in items]
 fig,axes=plt.subplots(2,4,figsize=(19,13))
 for ax,(item,is_ref) in zip(axes.flat,panels):panel(ax,case,item,is_ref)
 legend=[Line2D([0],[0],color=c,label=l,lw=2) for c,l in zip(COLORS,['车道线','路沿','停止线','斑马线'])]
 legend += [Line2D([0],[0],color='#1a995e',label='目标参考'),Line2D([0],[0],color='#e4a320',ls='--',label='Guide'),Line2D([0],[0],color='#d4dbe3',ls='--',label='物理真值背景（不是输入）')]
 fig.suptitle(case['title']+'；固定道路、位置、Guide、目标和FOV',fontsize=15)
 fig.legend(handles=legend,loc='lower center',ncol=7,fontsize=9)
 fig.tight_layout(rect=(0,.045,1,.965));fig.savefig(folder/'comparison.png',dpi=140);plt.close(fig)
 for item,is_ref in panels:
  fig,ax=plt.subplots(figsize=(6,8));panel(ax,case,item,is_ref)
  fig.tight_layout();name='truth' if is_ref else item['key'];fig.savefig(folder/(name+'.png'),dpi=130);plt.close(fig)
 # A close view makes centimetre-scale position noise visible without changing inputs.
 g=case['gt'];stop=35 if case['kind']=='intersection' else 40
 near=g[:stop];xmin=max(-27,float((-near[:,1]).min())-4);xmax=min(27,float((-near[:,1]).max())+4)
 ymin=max(-18,float(near[:,0].min())-2);ymax=min(90,float(near[:,0].max())+5)
 chosen=[items[0],items[1],items[3],items[5]]
 fig,axes=plt.subplots(1,4,figsize=(16,7))
 for ax,item in zip(axes,chosen):panel(ax,case,item,limits=(xmin,xmax,ymin,ymax))
 fig.suptitle(case['title']+'：局部放大（基准 / 轻A / 中A / 强A）',fontsize=13)
 fig.tight_layout(rect=(0,0,1,.95));fig.savefig(folder/'detail.png',dpi=140);plt.close(fig)

def gallery(output,cases):
 import html
 escape=html.escape
 text=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>感知观测审核</title><style>body{margin:24px auto;max-width:1600px;padding:0 24px;font:15px system-ui;color:#26394c;background:#f7f9fc}p{line-height:1.7}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:15px}.card,section{padding:16px;background:white;border:1px solid #dae2ed;border-radius:9px;margin-bottom:20px}img{width:100%;height:auto}label{display:block;margin:5px 0}button{padding:10px 18px;cursor:pointer}textarea{width:96%;min-height:60px}small{color:#6c7e90}summary{cursor:pointer}a{color:#275ea9}</style>',
 '<h1>同一道路的多种模拟感知观测</h1><p>用途：审核当前感知器的视觉假设，不是模型预测，也不是已标定的真实感知分布。每个场景保持道路、自车、Guide和目标不变。主对照固定前85m、后18m、左右22m；仅改变漏检、断裂和误差。轻/中/强使用当前训练参数，A/B为未筛选的两次随机抽样。</p>',
 '<p><b>蓝：车道线；深灰：路沿；红：停止线；青：斑马线。绿：目标参考；橙虚线：Guide。浅灰虚线只是物理真值背景，不是输入；浅黄区域表示道路本来缺内部标线，不能算漏检。</b></p>',
 '<p>观测面板画的是实际编码器保留的最多32个元素、每个32点。框为名义FOV；感知器在裁剪后添加位置噪声，少量点可能越过框边。单次“较强”不保证一定比“中度”少线。</p>',
 '<button id="export">下载我的反馈 JSON</button><span id="status"></span><p>每张可选“常见 / 偶尔有 / 不合理”，也可填写具体原因。反馈只存本浏览器，下载后可交给我，不自动发送。</p>']
 for case in cases:
  id=case['id'];text += [f'<section><h2>{escape(case["title"])}</h2><p>场景编号：{id}；nav={case["navigation"]}；仅增道，无减道样例。</p>' if case['kind']!='intersection' else f'<section><h2>{escape(case["title"])}</h2><p>场景编号：{id}；nav={case["navigation"]}。</p>',
   f'<a href="{id}/comparison.png">整组大图</a> · <a href="{id}/detail.png">局部放大</a> · <a href="{id}/truth.png">物理真值</a><details><summary>展开整组对照</summary><img src="{id}/comparison.png"></details><div class="grid">']
  for item in case['items']:
   key=id+'/'+item['key'];s=item['stats']
   text += [f'<div class="card"><h3>{escape(item["title"])}</h3><a href="{key}.png"><img loading="lazy" src="{key}.png"></a><p><small>保留{s["retained_objects"]}个元素；容量丢弃{s["discarded_due_to_capacity"]}；线段总长{s["open_segment_length_m"]:.1f}m</small></p><a href="{key}.json">模型输入 JSON</a>',f'<div data-review="{key}">']
   for value in ['常见','偶尔有','不合理']:text += [f'<label><input type="radio" name="{key}" value="{value}">{value}</label>']
   text += ['<textarea placeholder="可选：哪条线、哪个位置不合理，或希望缺失怎样变化"></textarea></div></div>']
  text += ['</div></section>']
 text += ['<h2>另外：当前训练三档可见范围</h2><p>下面只改变可见范围，不叠噪声，便于单独评价“能看到多远”。这与上面的固定范围误差对照分开。</p><img src="range_comparison.png"><p>参数与校验见 <a href="manifest.json">manifest.json</a>。</p>',
 '''<script>
const storeKey='pathgen_perception_review_20261105_v1';let feedback={};try{feedback=JSON.parse(localStorage.getItem(storeKey)||'{}')}catch(e){}
function save(){try{localStorage.setItem(storeKey,JSON.stringify(feedback));document.getElementById('status').textContent=' 已保存在本浏览器'}catch(e){document.getElementById('status').textContent=' 浏览器存储不可用，请下载反馈'}}
for(const box of document.querySelectorAll('[data-review]')){const key=box.dataset.review,old=feedback[key]||{};for(const radio of box.querySelectorAll('input'))radio.checked=radio.value===old.rating;box.querySelector('textarea').value=old.note||'';box.addEventListener('change',()=>{feedback[key]={rating:box.querySelector('input:checked')?.value||null,note:box.querySelector('textarea').value};save()});box.querySelector('textarea').addEventListener('input',()=>{feedback[key]={rating:box.querySelector('input:checked')?.value||null,note:box.querySelector('textarea').value};save()})}
document.getElementById('export').onclick=()=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify({version:1,review:'20261105_v1',feedback},null,2)],{type:'application/json'}));a.href=url;a.download='perception_review_feedback.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>''']
 (output/'gallery.html').write_text('\n'.join(text))

def build(output,seed=551105):
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 qualities=read_qualities(ROOT/'perception_v4.example.json');common=qualities['clean']
 baseline=replace(common,line_miss_rate=0,curb_miss_rate=0,box_miss_rate=0,break_rate=0,sigma_m=0,angle_sigma_deg=0,offset_sigma_m=0)
 cases=[make_case('A_intersection_left','intersection',30000000,'left'),
  make_case('B_intersection_right','intersection',30007919,'right'),
  make_case('C_add_right_2to3','addition',44000000,'right',(2,3,'right')),
  make_case('D_add_left_3to4','addition',44000004,'left',(3,4,'left'))]
 plt=setup_plot();report={'version':1,'seed':seed,'policy':'Fixed geometry/pose/GT/Guide/FOV. Two unrejected independent draws per existing quality. Not calibrated to real perception.',
  'original_training_qualities':{k:asdict(v) for k,v in qualities.items()},'cases':{}}
 for ci,case in enumerate(cases):
  folder=output/case['id'];folder.mkdir();items=[]
  specifications=[('baseline','基准：无噪声、无随机漏检',baseline,0)]
  for qi,(label,q) in enumerate(qualities.items()):
   fixed=replace(q,forward_m=common.forward_m,rear_m=common.rear_m,left_m=common.left_m,right_m=common.right_m)
   for draw in range(2):specifications.append((f'{label}_{"AB"[draw]}',f'{LABELS[label]}观测 {"AB"[draw]}',fixed,1+qi*2+draw))
  common_hash=digest({'gt':case['gt'],'guide':case['guide']});records={}
  for key,title,q,slot in specifications:
   draw_seed=[seed,ci,slot];row=row_for(case,q,draw_seed);row['id']+='_'+key
   row['metadata']={'review_case':case['id'],'quality':key,'fov_m':{'forward':q.forward_m,'rear':q.rear_m,'left':q.left_m,'right':q.right_m},'seed_sequence':draw_seed}
   assert digest({'gt':np.asarray(row['gt_path']),'guide':np.asarray(row['reference_line'])})==common_hash
   encoded=encode_input(row,32,32);s=stats(row,encoded,case['truth'])
   assert np.isfinite(encoded['bev_points'].numpy()).all()
   item={'key':key,'title':title,'quality':q,'row':row,'encoded':encoded,'stats':s}
   items.append(item);write(folder/(key+'.json'),row);records[key]={'parameters':asdict(q),'seed':draw_seed,'stats':s}
  write(folder/'world.json',case['world']);write(folder/'truth_ego.json',case['truth'])
  write(folder/'context.json',{'navigation':case['navigation'],'ego_world':case['ego'],'heading':case['heading'],
   'gt':case['gt'],'guide':case['guide'],'road_parameters':case['road_parameters'],'blank_polygon':case['blank_polygon']})
  case['items']=items;draw_case(case,items,folder,plt)
  report['cases'][case['id']]={'title':case['title'],'world_id':case['world']['world_id'],'world_sha256':digest(case['world']),
   'road_parameters':case['road_parameters'],'fixed_gt_guide_sha256':common_hash,'all_variants_same_gt_guide':True,'observations':records}
 # Separately show only the existing quality-dependent field-of-view differences.
 fig,axes=plt.subplots(len(cases),3,figsize=(15,18))
 for ci,case in enumerate(cases):
  for qi,(label,q) in enumerate(qualities.items()):
   exact=replace(baseline,forward_m=q.forward_m,rear_m=q.rear_m,left_m=q.left_m,right_m=q.right_m)
   row=row_for(case,exact,[seed,ci,90+qi]);encoded=encode_input(row,32,32)
   item={'title':f'{case["id"]}\n{label}范围：前{q.forward_m:g} / 左右{q.left_m:g}m','quality':exact,'encoded':encoded,'stats':stats(row,encoded,case['truth'])}
   panel(axes[ci,qi],case,item)
 fig.suptitle('仅比较当前训练FOV：不叠加漏检、断裂或位置误差',fontsize=16)
 fig.tight_layout(rect=(0,0,1,.97));fig.savefig(output/'range_comparison.png',dpi=120);plt.close(fig)
 # Compact overview: reference crop and first draw of each quality.
 fig,axes=plt.subplots(len(cases),4,figsize=(18,18))
 for ci,case in enumerate(cases):
  for j,item in enumerate([case['items'][k] for k in [0,1,3,5]]):panel(axes[ci,j],case,item)
 fig.suptitle('固定道路与FOV：基准 / 轻度A / 中度A / 较强A',fontsize=16)
 fig.tight_layout(rect=(0,0,1,.97));fig.savefig(output/'overview.png',dpi=130);plt.close(fig)
 write(output/'manifest.json',report);gallery(output,cases)
 print(json.dumps({'output':str(output),'cases':len(cases),'observation_inputs':sum(len(c['items']) for c in cases),
  'all_gt_guides_fixed':True,'addition_only':True,'max_discarded_tokens':max(v['stats']['discarded_due_to_capacity'] for c in report['cases'].values() for v in c['observations'].values())},ensure_ascii=False,indent=2))
 return report

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=Path('outputs/perception_review/20261105_v1'))
 parser.add_argument('--seed',type=int,default=551105)
 args=parser.parse_args();build(args.output_dir,args.seed)
