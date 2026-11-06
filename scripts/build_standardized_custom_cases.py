"""Build curated, annotated custom diagnostics, independently of training generators.

No training or inference is started. Full geometry/target annotations are separate
from FOV-cropped perception. Reference paths are designed answers, not measured GT.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import LineString, Polygon, box
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FIELDS=('lane_markings','road_boundaries','stop_lines','crosswalks')
FOV={'forward':85.,'rear':18.,'left':22.,'right':22.}
VIEW=box(-18,-22,85,22)
WIDTH=3.5

def sample_path(p,n=81):
 p=np.asarray(p,float)
 p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-9]]
 arc=np.r_[0.,np.linalg.norm(np.diff(p,axis=0),axis=1).cumsum()]
 if arc[-1]<n-1:raise ValueError('Path is too short')
 return np.column_stack([np.interp(np.arange(n),arc,p[:,k]) for k in range(2)])

def smooth(t):
 t=np.clip(t,0,1)
 return t**3*(10-15*t+6*t*t)

def base(name,nav):
 return {'id':name,'navigation':nav,'coordinate_system':'ego_at_start_x_forward_y_left_m',**{k:[] for k in FIELDS}}

def add(row,key,id,p,**kw):
 row[key].append({'id':id,'points':np.asarray(p,float).tolist(),'confidence':.99,**kw})

def edge(row,id,p,style='solid',color='white'):
 add(row,'lane_markings',id,p,line_style=style,color=color)

def rect(row,key,id,x0,x1,y0,y1):
 add(row,key,id,[[x0,y0],[x1,y0],[x1,y1],[x0,y1]])

def strip(left,right):
 return Polygon(np.r_[left,right[::-1]])

def straight():
 x=np.linspace(-18,125,1431);center=np.c_[x,np.zeros_like(x)]
 row=base('standard_straight','straight')
 for id,y in [('left_edge',1.75),('separator_right',-1.75),('right_edge',-5.25)]:
  edge(row,id,np.c_[x,np.full_like(x,y)],'dashed' if 'separator' in id else 'solid')
 for id,y in [('left_curb',2.),('right_curb',-5.5)]:add(row,'road_boundaries',id,np.c_[x,np.full_like(x,y)])
 xx=np.linspace(0,125,1251)
 row['editor_guide_points']=np.c_[xx,-.8*smooth(xx/25)].tolist()
 row['reference_line']=sample_path(row['editor_guide_points']).tolist()
 target=sample_path(center[center[:,0]>=0])
 l=center+[0,1.75];r=center-[0,1.75]
 return row,target,strip(l,r),strip(center+[0,2],center-[0,5.5]),{
  'title':'直行基准：两车道，保持当前左侧车道','source_case':'20261105_185446_664301',
  'connection':{'entry':'in_left','exit':'out_left','cross_existing_separator':False},
  'navigation_semantics':'当前路径直行；不包含后续左转意图',
  'guide_error':'起点零偏差，前25m渐偏右0.8m；之后保持偏差',
  'target_boundaries':['left_edge','separator_right'],'exit_station_xy_m':[70.,0.],
  'changes':'保留近直行测试意图，重建等宽直道；nav改straight；去掉无对应路口的停止线/斑马线。'}

def turn_path(entry=12.,radius=14.,end_y=110.):
 a=np.linspace(-18,entry,int((entry+18)*10)+1)
 theta=np.linspace(0,np.pi/2,501)
 approach=np.c_[a,np.zeros_like(a)]
 turn=np.c_[entry+radius*np.sin(theta),radius*(1-np.cos(theta))]
 yy=np.linspace(radius,end_y,int((end_y-radius)*10)+1)
 departure=np.c_[np.full_like(yy,entry+radius),yy]
 p=np.r_[approach[:-1],turn[:-1],departure]
 tangent=np.r_[np.tile([1.,0.],(len(approach)-1,1)),np.c_[np.cos(theta[:-1]),np.sin(theta[:-1])],np.tile([0.,1.],(len(departure),1))]
 return p,np.c_[-tangent[:,1],tangent[:,0]]

def left_turn():
 center,normal=turn_path()
 row=base('standard_left_turn','left')
 # Two parallel left-turn connections; no crossing between them.
 for id,offset in [('left_edge',1.75),('separator_right',-1.75),('right_edge',-5.25)]:
  full=center+offset*normal
  # Physical lane markings stop at the junction; do not interpolate over it.
  approach=center[:,1]<1e-8;approach&=center[:,0]<=12.+1e-8
  departure=center[:,1]>=14.-1e-8;departure&=center[:,0]>=26.-1e-8
  for section,mask in [('approach',approach),('departure',departure)]:
   edge(row,id+'_'+section,full[mask],'dashed' if 'separator' in id else 'solid')
 for id,offset in [('left_curb',2.),('right_curb',-5.5)]:add(row,'road_boundaries',id,center+offset*normal)
 rect(row,'stop_lines','approach_stop',6.3,6.7,-5.25,1.75)
 rect(row,'crosswalks','approach_crosswalk',8.,11.,-5.25,1.75)
 rect(row,'crosswalks','departure_crosswalk',24.25,31.25,17.,20.)
 guide,_=turn_path(entry=9.5,radius=14.)
 guide=guide[guide[:,0]>=0] # Approach starts at the ego, without a forward gap.
 row['editor_guide_points']=guide.tolist()
 row['reference_line']=sample_path(guide).tolist()
 target=sample_path(center[center[:,0]>=0])
 return row,target,strip(center+1.75*normal,center-1.75*normal),strip(center+2*normal,center-5.5*normal),{
  'title':'左转连接诊断：当前左侧车道→出口内侧车道','source_case':'20261105_183404_453045',
  'connection':{'entry':'in_left','exit':'out_inner','cross_existing_separator':False,'other_connection':'in_right -> out_outer'},
  'navigation_semantics':'当前80m路径内完成左转，不是直行路段的后续准备指令',
  'guide_error':'比目标提前2.5m转弯，出口向内偏2.5m；有意保留Guide/路沿冲突',
  'target_boundaries':['left_edge_approach','separator_right_approach','left_edge_departure','separator_right_departure'],
  'exit_station_xy_m':[26.,30.],
  'changes':'简化的双车道左转连接，非完整四臂路口复刻；保留提前左转/路沿冲突意图，重建3.5m等宽、切向连续的双车道连接和完整路沿；路口内部不画虚构标线。'}

def split_right():
 x=np.linspace(-18,125,1431);q=smooth((x-12)/40)
 center=np.c_[x,-3.5*q]
 # Normals for a 3.5m target connector envelope, independent of Guide.
 dx=np.gradient(center[:,1],x);normal=np.c_[-dx,np.ones_like(x)];normal/=np.linalg.norm(normal,axis=1)[:,None]
 row=base('standard_right_split','right')
 edge(row,'left_edge',np.c_[x,np.full_like(x,5.25)])
 edge(row,'right_edge',np.c_[x,-1.75-3.5*q])
 add(row,'road_boundaries','left_curb',np.c_[x,np.full_like(x,5.5)])
 add(row,'road_boundaries','right_curb',np.c_[x,-2.-3.5*q])
 # Common missing-marking zone [16,46]; no segment bridges this gap.
 for label,mask in [('approach',x<=16),('departure',x>=46)]:
  edge(row,'retained_separator_'+label,np.c_[x[mask],np.full(mask.sum(),1.75)],'dashed')
 mask=x>=46;edge(row,'new_separator_departure',np.c_[x[mask],np.full(mask.sum(),-1.75)],'dashed')
 xx=np.linspace(0,125,1251)
 guide_y=-3.5*smooth((xx-20)/40)+.8*smooth((xx-12)/40)
 row['editor_guide_points']=np.c_[xx,guide_y].tolist()
 row['reference_line']=sample_path(row['editor_guide_points']).tolist()
 target=sample_path(center[x>=0])
 return row,target,strip(center+1.75*normal,center-1.75*normal),strip(np.c_[x,np.full_like(x,5.5)],np.c_[x,-2.-3.5*q]),{
  'title':'右侧2→3增道：当前右车道自然分出新增右车道','source_case':'20261105_133929_954093',
  'connection':{'entry':'in_right','exit':'out_new_right','cross_existing_separator':False,
                'all_connections':['in_left -> out_left','in_right -> out_middle','in_right -> out_new_right']},
  'navigation_semantics':'下一路口右转准备；当前道路是增道，选择可达的新增右车道',
  'guide_error':'分流横移滞后8m，出口偏左0.8m；保持目标出口身份',
  'target_boundaries':['right_edge','new_separator_departure'],'exit_station_xy_m':[70.,-3.5],
  'changes':'保留右侧增道意图，明确2→3连接；过渡x12–52，公共无内部标线区x16–46；完整出口宽3.5m。'}

def crop(row):
 result=copy.deepcopy(row)
 result.pop('editor_guide_points',None)
 for key in FIELDS:
  result[key]=[]
  for obj in row[key]:
   p=np.asarray(obj['points'])
   if key in ('stop_lines','crosswalks'):
    # Designed boxes are fully inside the FOV; never export a distorted partial box.
    if VIEW.covers(Polygon(p)):result[key].append(copy.deepcopy(obj))
    continue
   inter=LineString(p).intersection(VIEW)
   parts=[] if inter.is_empty else ([inter] if inter.geom_type=='LineString' else list(inter.geoms))
   for i,part in enumerate(parts):
    if part.geom_type!='LineString' or part.length<.5:continue
    item=copy.deepcopy(obj);item['source_id']=obj['id'];item['id']=obj['id']+f'_visible_{i}'
    item['points']=np.asarray(part.coords).tolist();result[key].append(item)
 return result

def write(path,value):
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def validate(full,row,target,corridor,road):
 from infer import encode_input,prepare_sample
 from scene_editor import import_project
 g=np.asarray(row['reference_line']);t=np.asarray(target)
 assert g.shape==t.shape==(81,2) and np.isfinite(g).all() and np.isfinite(t).all()
 assert np.linalg.norm(t[0])<1e-6 and np.linalg.norm(g[0])<1e-6
 assert np.linalg.norm(np.diff(t,axis=0),axis=1).max()<=1.001
 assert np.linalg.norm(np.diff(g,axis=0),axis=1).max()<=1.001
 assert road.is_valid and corridor.is_valid
 assert road.buffer(1e-5).covers(LineString(t))
 assert road.buffer(1e-5).covers(corridor)
 assert all(VIEW.buffer(1e-7).covers(LineString(e['points'])) for k in FIELDS for e in row[k])
 actual=encode_input(prepare_sample(row),32,32)
 without=copy.deepcopy(row);without.pop('gt_path')
 unlabelled=encode_input(prepare_sample(without),32,32)
 import torch
 for k in actual:torch.testing.assert_close(actual[k],unlabelled[k],rtol=0,atol=0)
 count=sum(len(row[k]) for k in FIELDS)
 assert int(actual['polyline_mask'].sum())==count
 return {'gt_start_m':t[0].tolist(),'guide_start_m':g[0].tolist(),
  'max_gt_step_m':float(np.linalg.norm(np.diff(t,axis=0),axis=1).max()),
  'target_within_road':True,'target_corridor_within_road':True,'all_inputs_within_fov':True,
  'objects_retained':count,'annotation_does_not_change_model_input':True,
  'guide_outside_road_length_m':float(LineString(g).difference(road).length),
  'guide_curb_intersections':[e['id'] for e in full['road_boundaries'] if LineString(g).intersects(LineString(e['points']))]}

def plot_cases(cases,output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Polygon as Patch,Rectangle
 colors={'lane_markings':'#6380a4','road_boundaries':'#253951','stop_lines':'#d65f5f','crosswalks':'#33a9bb'}
 def panel(ax,case,visible):
  full,row,t,corridor,road,ann=case
  poly=np.asarray(road.exterior.coords);ax.add_patch(Patch(np.c_[-poly[:,1],poly[:,0]],color='#eef2f6',zorder=0))
  poly=np.asarray(corridor.exterior.coords);ax.add_patch(Patch(np.c_[-poly[:,1],poly[:,0]],color='#bce8d2',alpha=.75,zorder=1))
  source=row if visible else full
  for key,col in colors.items():
   for e in source[key]:
    p=np.asarray(e['points']);p=np.r_[p,p[:1]] if key in FIELDS[2:] else p
    ax.plot(-p[:,1],p[:,0],color=col,lw=1.3,zorder=2)
  g=np.asarray(row['reference_line']);ax.plot(-g[:,1],g[:,0],color='#e69a15',ls='--',lw=2,label='Guide',zorder=4)
  ax.plot(-t[:,1],t[:,0],color='#138558',lw=2,label='Designed target (not sensor GT)',zorder=3)
  ax.scatter([0],[0],c='#15283e',marker='^',zorder=5)
  ax.annotate('Entry: '+ann['connection']['entry'],(0,0),xytext=(8,-12),textcoords='offset points',fontsize=7,color='#253951')
  exit_=np.asarray(ann['exit_station_xy_m']);ax.annotate('Target exit: '+ann['connection']['exit'],(-exit_[1],exit_[0]),xytext=(10,8),textcoords='offset points',fontsize=8,color='#138558')
  ax.add_patch(Rectangle((-22,-18),44,103,fill=False,edgecolor='#9867bf',ls=':',lw=1.5,label='Model FOV'))
  if visible:
   if row['navigation']=='left':ax.set_xlim(-24,9);ax.set_ylim(-3,38)
   else:ax.set_xlim(-9,10);ax.set_ylim(-3,87)
  else:
   ax.set_xlim((-88 if row['navigation']=='left' else -10),10);ax.set_ylim(-5,86 if row['navigation']!='left' else 36)
  ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15);ax.set_xlabel('Right (-ego y), m');ax.set_ylabel('Forward (ego x), m')
  ax.set_title(row['id']+(' | cropped model input' if visible else ' | full geometry / target'),fontsize=10)
  # Legend is placed outside the road panels so narrow lane geometry stays visible.
 fig,axes=plt.subplots(len(cases),2,figsize=(14,16))
 for i,case in enumerate(cases):
  for j in range(2):panel(axes[i,j],case,j==1)
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
 fig.tight_layout(rect=(0,.025,1,1));fig.savefig(output/'overview.png',dpi=140);plt.close(fig)
 for case in cases:
  fig,axes=plt.subplots(1,2,figsize=(13,6))
  for j in range(2):panel(axes[j],case,j==1)
  fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
  fig.tight_layout(rect=(0,.06,1,1));fig.savefig(output/case[1]['id']/'preview.png',dpi=140);plt.close(fig)

def build(output):
 from scene_editor import import_project
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 cases=[];report={}
 for fn in [straight,left_turn,split_right]:
  full,target,corridor,road,ann=fn();row=crop(full);row['gt_path']=target.tolist()
  ann.update(reference_kind='人工设计的标准答案，不是真实采集GT；不声称是唯一舒适轨迹',
    lane_width_m=WIDTH,ego_pose={'xy_m':[0.,0.],'heading_rad':0.},fov_m=FOV,
    corridor_interpretation='在无标线路段为人工声明的虚拟连接走廊，不作为感知输入或声称物理标线',
    input_policy='确定性FOV裁剪，不加随机漏检/噪声；置信度0.99；框仅完整可见时保留',
    model_input_fields='Guide、导航、可见BEV；目标标签不传入模型',
    target_path_81=target.tolist(),target_corridor_polygon=np.asarray(corridor.exterior.coords).tolist(),
    drivable_polygon=np.asarray(road.exterior.coords).tolist())
  report[row['id']]=validate(full,row,target,corridor,road)
  folder=output/row['id'];folder.mkdir()
  write(folder/'input.json',row)
  scene=import_project(row)
  next(e for e in scene['entities'] if e['kind']=='guide')['points']=full['editor_guide_points']
  from scene_editor import export_project
  roundtrip,_,_=export_project(scene)
  np.testing.assert_allclose(roundtrip['reference_line'],row['reference_line'],atol=1e-8,rtol=0)
  write(folder/'scene.json',scene)
  write(folder/'annotation.json',ann);write(folder/'world.json',full)
  cases.append((full,row,target,corridor,road,ann))
 plot_cases(cases,output)
 write(output/'validation.json',report)
 html=['<!doctype html><meta charset="utf-8"><title>标准化自定义场景</title><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto;color:#24374d}img{width:100%}article{margin:30px 0;border-top:1px solid #ddd}p{line-height:1.7}a{margin-right:20px}</style>',
 '<h1>标准化自定义场景：连接、感知范围、目标明确</h1><p>绿色：人工目标路径；浅绿色：目标连接走廊；橙虚线：待修正Guide；紫色框：模型FOV。左图完整几何，右图仅显示可见BEV，绿色标签仅供核对，不进入模型。</p>']
 for _,row,_,_,_,ann in cases:
  name=row['id'];html+=[f'<article><h2>{ann["title"]}</h2><p>{ann["changes"]}<br>导航：{ann["navigation_semantics"]}<br>Guide误差：{ann["guide_error"]}</p>',
   f'<a href="{name}/input.json">模型输入+参考标签</a><a href="{name}/scene.json">编辑器工程</a><a href="{name}/annotation.json">连接与标注</a><img src="{name}/preview.png"></article>']
 (output/'gallery.html').write_text('\n'.join(html))
 print(json.dumps(report,ensure_ascii=False,indent=2))
 return report

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=Path('outputs/custom_standardized/20261105_v2'))
 build(parser.parse_args().output_dir)
