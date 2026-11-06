'use strict';
const $ = id => document.getElementById(id);
const svg=$('canvas');
const names={guide:'Guide line',line:'BEV line',curb:'BEV curb',stop:'停止线',cross:'斑马线'};
const colors={guide:'#e69a17',line:'#407fe8',curb:'#475569',stop:'#d75f61',cross:'#2baca9'};
const clone=x=>JSON.parse(JSON.stringify(x));
const escape=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const empty=()=>({version:1,name:'custom_scene',navigation:'straight',entities:[]});
let project=empty(), selected=null, undo=[], redo=[], draft=null, drag=null, space=false, sampled=null, lastSave=null, dirty=0;
let camera={x:-50,y:-95,w:100,h:110};
let toastTimer,localTimer,measurementProbe=null;
let rotationEntity=null;
function toast(message,error=false){$('toast').textContent=message;$('toast').classList.toggle('error',error);$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,error?7000:3500);}
function current(){return project.entities.find(e=>e.id===selected);}
function remember(){undo.push(clone(project));if(undo.length>100)undo.shift();redo=[];}
function changed(){dirty++;sampled=null;lastSave=null;$('result').hidden=true;clearTimeout(localTimer);localTimer=setTimeout(()=>{try{localStorage.setItem('pathgen_scene_editor_v1',JSON.stringify(project));}catch(e){/* Storage can be disabled; file saves still work. */}},300);}
function undoAction(){if(!undo.length)return;redo.push(clone(project));project=undo.pop();if(!current())selected=null;draft=null;changed();render();}
function redoAction(){if(!redo.length)return;undo.push(clone(project));project=redo.pop();if(!current())selected=null;changed();render();}
function geometry(e){
 if(e.mode!=='bezier')return e.points;
 const p=e.points,out=[],n=Math.min(8192,Math.max(256,Math.ceil(length(p)/.1)));
 for(let i=0;i<=n;i++){const t=i/n,u=1-t,w=[u*u*u,3*u*u*t,3*u*t*t,t*t*t];out.push([0,1].map(k=>p.reduce((s,q,j)=>s+w[j]*q[k],0)));}return out;
}
function length(p){return p.slice(1).reduce((s,v,i)=>s+Math.hypot(v[0]-p[i][0],v[1]-p[i][1]),0);}
function path(p,closed=false){return p.map((v,i)=>`${i?'L':'M'}${-v[1]},${-v[0]}`).join(' ')+(closed?' Z':'');}
function unit(){return camera.h/(svg.clientHeight||700);}
function metric(event,snap=false){const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;const v=p.matrixTransform(svg.getScreenCTM().inverse());let q=[-v.y,-v.x];if(snap&&$('snap').checked)q=q.map(v=>Math.round(v*2)/2);return q;}
function newId(kind){let n=1;while(project.entities.some(e=>e.id===`${kind}_${n}`))n++;return `${kind}_${n}`;}
function create(kind,mode,p){
 if(project.entities.length>=256){toast('最多支持256个元素',true);return false;}
 if(kind==='guide'&&project.entities.some(e=>e.kind==='guide')){toast('已有 Guide，请选择它编辑或删除后重画。',true);return false;}
 if(p.length<2||Math.hypot(p[0][0]-p.at(-1)[0],p[0][1]-p.at(-1)[1])<.01){toast('起点和终点不能重合',true);return false;}
 if(kind==='stop'||kind==='cross'){
  if(mode==='rectangle'){const [a,b]=p;if(Math.abs(a[0]-b[0])<.05||Math.abs(a[1]-b[1])<.05){toast('区域需要宽度和长度，请绘制两个不同的对角点',true);return false;}p=[a,[b[0],a[1]],b,[a[0],b[1]]];}mode='polygon';
 } else if(mode==='bezier'){
  const [a,b]=p;p=[a,[a[0]+(b[0]-a[0])/3,a[1]+(b[1]-a[1])/3],[a[0]+2*(b[0]-a[0])/3,a[1]+2*(b[1]-a[1])/3],b];
 }
 remember();const e={id:newId(kind),kind,mode,points:clone(p),confidence:1,color:'white',line_style:'dashed'};project.entities.push(e);selected=e.id;measurementProbe=null;changed();render();return true;
}
function changeType(){const area=['stop','cross'].includes($('kind').value);$('mode-wrap').hidden=area;$('draw').textContent=area?'画矩形区域':'在画布绘制';if(draft)cancelDraw();}
function startDraw(){
 const kind=$('kind').value;if(kind==='guide'&&project.entities.some(e=>e.kind==='guide')){toast('已有 Guide，请选中编辑或先删除。',true);return;}
 draft={kind,mode:['stop','cross'].includes(kind)?'rectangle':$('mode').value,points:[],hover:null};selected=null;$('finish').hidden=draft.mode!=='polyline';render();
}
function cancelDraw(){draft=null;$('finish').hidden=true;render();}
function finishDraw(){if(!draft)return;if(draft.points.length<2){toast('至少需要两个点',true);return;}const d=draft;if(create(d.kind,d.mode,d.points)){draft=null;$('finish').hidden=true;render();}}
function deleteSelected(){const e=current();if(!e)return;remember();project.entities=project.entities.filter(v=>v!==e);selected=null;changed();render();}
function fit(){
 const p=project.entities.flatMap(geometry);if(!p.length){resetView();return;}
 const xs=p.map(v=>-v[1]),ys=p.map(v=>-v[0]);xs.push(0);ys.push(0);
 let w=Math.max(...xs)-Math.min(...xs)+12,h=Math.max(...ys)-Math.min(...ys)+12,ratio=(svg.clientWidth||800)/(svg.clientHeight||700);
 h=Math.max(20,h,w/ratio);w=h*ratio;camera={x:(Math.max(...xs)+Math.min(...xs)-w)/2,y:(Math.max(...ys)+Math.min(...ys)-h)/2,w,h};renderCanvas();
}
function resetView(){const ratio=(svg.clientWidth||800)/(svg.clientHeight||700);camera={x:-110*ratio/2,y:-95,w:110*ratio,h:110};renderCanvas();}
function grid(){
 const u=unit(),step=camera.h>180?20:camera.h>65?10:camera.h>25?5:1;let out='';
 for(let x=Math.ceil(camera.x/step)*step;x<camera.x+camera.w;x+=step){out+=`<path d="M${x},${camera.y} V${camera.y+camera.h}" stroke="${x===0?'#c9d5e8':'#eaf0f7'}" stroke-width="1" vector-effect="non-scaling-stroke"/>`;out+=`<text x="${x+.4*u}" y="${camera.y+camera.h-12*u}" font-size="${10*u}" fill="#96a5b9">${Number((-x).toFixed(1))}</text>`;}
 for(let y=Math.ceil(camera.y/step)*step;y<camera.y+camera.h;y+=step){out+=`<path d="M${camera.x},${y} H${camera.x+camera.w}" stroke="${y===0?'#c9d5e8':'#eaf0f7'}" stroke-width="1" vector-effect="non-scaling-stroke"/>`;out+=`<text x="${camera.x+5*u}" y="${y-4*u}" font-size="${10*u}" fill="#96a5b9">${Number((-y).toFixed(1))}</text>`;}
 $('grid').innerHTML=out;
}
function measurementCandidates(){
 const e=current(),filter=$('measure-kind').value;
 return project.entities.filter(v=>v.id!==e?.id&&(filter==='all'?['guide','line','curb'].includes(v.kind):filter==='line'?v.kind==='line':['line','curb'].includes(v.kind)));
}
function renderMeasurementOptions(){
 const target=$('measure-target'),previous=target.value;
 const options='<option value="">自动：最近的线</option>'+measurementCandidates().map(v=>`<option value="${escape(v.id)}">${escape(v.id)} · ${names[v.kind]}</option>`).join('');
 if(target.innerHTML!==options){target.innerHTML=options;target.value=[...target.options].some(o=>o.value===previous)?previous:'';}
}
function renderMeasurement(){
 const e=current(),layer=$('measurement'),info=$('distance-info');layer.innerHTML='';renderMeasurementOptions();
 if(!e||!['guide','line','curb'].includes(e.kind)){info.textContent='选择 Guide、车道线或路沿进行测量';return;}
 const points=geometry(e),probe=measurementProbe||points[Math.floor(points.length/2)],target=$('measure-target').value;
 const candidates=measurementCandidates().filter(v=>!target||v.id===target).map(v=>({id:v.id,points:geometry(v)}));
 const m=LineMeasurement.measure(points,probe,candidates);
 if(!m){info.textContent='没有可测量的邻线';return;}
 info.textContent=`${target?'参照':'最近'}：${m.id} · ${m.distance.toFixed(2)} m（测点 x=${m.anchor[0].toFixed(1)}, y=${m.anchor[1].toFixed(1)}）`;
 const u=unit(),x=-(m.anchor[1]+m.point[1])/2,y=-(m.anchor[0]+m.point[0])/2;
 const halo=target?`<path d="${path(candidates[0].points)}" fill="none" stroke="#159182" stroke-width="6" opacity=".25" vector-effect="non-scaling-stroke"/>`:'';
 layer.innerHTML=halo+`<path d="${path([m.anchor,m.point])}" class="dimension"/><circle cx="${-m.anchor[1]}" cy="${-m.anchor[0]}" r="${3*u}" fill="#159182"/><circle cx="${-m.point[1]}" cy="${-m.point[0]}" r="${3*u}" fill="#159182"/><text x="${x+6*u}" y="${y-7*u}" font-size="${12*u}" class="dimension-label">${escape(m.id)} · ${m.distance.toFixed(2)} m</text>`;
}
$('measure-kind').onchange=renderMeasurement;$('measure-target').onchange=renderMeasurement;
function rotateSelected(degrees){
 const e=current();if(!e)return;
 if(!Number.isFinite(degrees)){toast('请输入有效旋转角度',true);return;}
 if(degrees%360===0)return;
 const mode=$('rotate-pivot').value,pivot=mode==='ego'?[0,0]:mode==='start'?e.points[0]:mode==='end'?e.points.at(-1):SceneGeometry.center(geometry(e),e.mode==='polygon');
 const next=SceneGeometry.rotate(e.points,degrees,pivot);
 if(next.flat().some(v=>!Number.isFinite(v)||Math.abs(v)>10000)){toast('旋转后坐标超出支持范围',true);return;}
 remember();e.points=next;measurementProbe=null;changed();render();
}
$('rotate-minus').onclick=()=>rotateSelected(-5);$('rotate-plus').onclick=()=>rotateSelected(5);
$('rotate-apply').onclick=()=>{if(!$('rotate-angle').value.trim()){toast('请输入旋转角度',true);return;}rotateSelected(Number($('rotate-angle').value));$('rotate-angle').value='0';};
function renderCanvas(){
 svg.setAttribute('viewBox',`${camera.x} ${camera.y} ${camera.w} ${camera.h}`);grid();const u=unit();let out='';
 if(project.gt_path)out+=`<path d="${path(project.gt_path)}" class="path" stroke="#16975d" opacity=".7"/>`;
 for(const e of project.entities){const p=geometry(e),closed=['stop','cross'].includes(e.kind),d=path(p,closed),col=e.kind==='line'&&e.color==='yellow'?'#c8a516':colors[e.kind];
  const dash=e.kind==='guide'?'7 5':e.kind==='line'&&e.line_style==='dashed'?'6 5':'';
  if(e.id===selected)out+=`<path d="${d}" fill="${closed?col+'18':'none'}" stroke="${col}" stroke-width="8" opacity=".13" vector-effect="non-scaling-stroke"/>`;
  out+=`<path d="${d}" class="path" stroke="${col}" ${dash?`stroke-dasharray="${dash}"`:''} style="${closed?'fill:'+col+'18;':''}"/><path d="${d}" class="hit" data-entity="${escape(e.id)}"/>`;
 }
 out+=`<path d="M0,${-7*u} L${-5*u},${5*u} L${5*u},${5*u} Z" fill="#253951"/><text x="${8*u}" y="${4*u}" font-size="${10*u}" fill="#73849c">自车 (0, 0)</text>`;$('scene').innerHTML=out;
 let handles='';const e=current();if(e){if(e.mode==='bezier')handles+=`<path d="${path(e.points.slice(0,2))} ${path(e.points.slice(2))}" class="control-line"/>`;
  e.points.forEach((p,i)=>{const c=e.mode==='bezier'&&(i===1||i===2);handles+=`<circle class="handle" data-node="${i}" cx="${-p[1]}" cy="${-p[0]}" r="${(c?5:6)*u}" fill="${c?'#fff':colors[e.kind]}" stroke="${colors[e.kind]}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`;});}
 $('handles').innerHTML=handles;
 let draw='';if(draft){const p=[...draft.points,...(draft.hover?[draft.hover]:[])];if(p.length>1){if(draft.mode==='rectangle'){const [a,b]=[p[0],p.at(-1)];draw=`<path d="${path([a,[b[0],a[1]],b,[a[0],b[1]]],true)}" fill="#577dec12" stroke="#6285d3" stroke-dasharray="5 4" stroke-width="1" vector-effect="non-scaling-stroke"/>`;}else draw=`<path d="${path(p)}" fill="none" stroke="${colors[draft.kind]}" stroke-dasharray="5 4" stroke-width="2" vector-effect="non-scaling-stroke"/>`;}
  draft.points.forEach(p=>draw+=`<circle cx="${-p[1]}" cy="${-p[0]}" r="${4*u}" fill="${colors[draft.kind]}"/>`);}
 $('drawing').innerHTML=draw;
 let samples='';if(sampled){for(const p of sampled.guide)samples+=`<circle cx="${-p[1]}" cy="${-p[0]}" r="${2.2*u}" fill="#d18400"/>`;for(const e of sampled.lines)for(const p of e.points)samples+=`<circle cx="${-p[1]}" cy="${-p[0]}" r="${1.8*u}" fill="#315fea"/>`;}$('samples').innerHTML=samples;
 renderMeasurement();svg.classList.toggle('drawing',!!draft);svg.classList.toggle('panning',!!drag&&drag.mode==='pan');
 $('canvas-tip').textContent=draft?(draft.points.length===0?'点击起点':draft.mode==='polyline'?'继续点击添加节点，Enter /「完成绘制」结束':'点击终点；创建后拖动控制点调整形状'):'选择 / 拖动控制点 · 空格拖动画布 · 滚轮缩放';
 $('zoom-label').textContent=`网格 ${camera.h>180?20:camera.h>65?10:camera.h>25?5:1} m`;
 const guide=project.entities.find(e=>e.kind==='guide'),glen=guide?length(geometry(guide)):0;
 $('guide-status').textContent=guide?`Guide ${glen.toFixed(2)} m · ${glen>=80-1e-6?'可导出 81 点':'需延长至 ≥80 m'}`:'尚未绘制 Guide';$('guide-status').className=guide?(glen>=80-1e-6?'good':'bad'):'';
}
function renderProperties(){
 const e=current();$('properties').hidden=!e;$('no-selection').hidden=!!e;if(!e)return;
 if(rotationEntity!==e.id){rotationEntity=e.id;$('rotate-pivot').value=e.kind==='guide'?'start':'center';$('rotate-angle').value='0';}
 $('prop-id').value=e.id;$('semantics').hidden=e.kind==='guide';$('line-attrs').hidden=e.kind!=='line';$('confidence').value=e.confidence??1;$('confidence-label').textContent=Number(e.confidence??1).toFixed(2);$('color').value=e.color||'unknown';$('style').value=e.line_style||'unknown';$('prop-kind').textContent=names[e.kind];$('prop-length').textContent=length(geometry(e)).toFixed(2)+' m';
 $('control-help').textContent=e.mode==='bezier'?'实心点是端点，空心点控制曲线形状；两种点都可拖动或输入坐标。':'拖动节点调整形状；折线双击线段可插入节点。';
 $('nodes').innerHTML=e.points.map((p,i)=>`<div class="node"><div><span>${e.mode==='bezier'?['起点','形状控制点 1','形状控制点 2','终点'][i]:i===0?'起点':i===e.points.length-1?'终点':`节点 ${i}`}</span>${e.mode!=='bezier'&&e.points.length>(e.mode==='polygon'?3:2)?`<button data-remove-node="${i}">移除</button>`:''}</div><div class="coords"><label>x 向前<input type="number" step=".1" value="${Number(p[0].toFixed(4))}" data-coordinate="${i},0"></label><label>y 向左<input type="number" step=".1" value="${Number(p[1].toFixed(4))}" data-coordinate="${i},1"></label></div></div>`).join('');
 $('add-node').hidden=e.mode==='bezier';$('duplicate').disabled=e.kind==='guide';
}
function render(){
 $('name').value=project.name;$('nav').value=project.navigation;$('count').textContent=project.entities.length;$('empty-list').hidden=!!project.entities.length;
 $('entities').innerHTML=project.entities.map(e=>`<button class="entity ${e.id===selected?'selected':''}" data-select="${escape(e.id)}"><span class="swatch" style="background:${colors[e.kind]}"></span><span class="name">${escape(e.id)}</span><small>${e.kind==='guide'?'Guide':e.kind}</small></button>`).join('');
 $('undo').disabled=!undo.length;$('redo').disabled=!redo.length;renderProperties();renderCanvas();
}
function moveNode(e,index,p){const old=e.points[index],dx=p[0]-old[0],dy=p[1]-old[1];e.points[index]=p;if(e.mode==='bezier'&&(index===0||index===3)){const j=index===0?1:2;e.points[j]=[e.points[j][0]+dx,e.points[j][1]+dy];}}
svg.addEventListener('pointerdown',event=>{
 if(event.button===1||space){drag={mode:'pan',x:event.clientX,y:event.clientY};svg.setPointerCapture(event.pointerId);event.preventDefault();renderCanvas();return;}
 if(event.button!==0)return;
 if(draft){if(event.detail>1&&draft.mode==='polyline')return;draft.points.push(metric(event,true));if(draft.mode!=='polyline'&&draft.points.length===2){finishDraw();}else renderCanvas();return;}
 const node=event.target.closest('[data-node]');if(node&&current()){remember();drag={mode:'node',index:Number(node.dataset.node)};svg.setPointerCapture(event.pointerId);event.preventDefault();return;}
 const target=event.target.closest('[data-entity]');selected=target?target.dataset.entity:null;measurementProbe=null;render();
});
svg.addEventListener('pointermove',event=>{
 const p=metric(event);$('pointer').textContent=`x ${p[0].toFixed(2)} · y ${p[1].toFixed(2)} m`;
 if(drag?.mode==='pan'){const u=unit();camera.x-=(event.clientX-drag.x)*u;camera.y-=(event.clientY-drag.y)*u;drag.x=event.clientX;drag.y=event.clientY;renderCanvas();}
 else if(drag?.mode==='node'&&current()){moveNode(current(),drag.index,metric(event,true));measurementProbe=p;sampled=null;renderCanvas();}
 else if(draft){draft.hover=metric(event,true);renderCanvas();}
 else if(current()){measurementProbe=p;renderMeasurement();}
});
function release(){if(drag?.mode==='node'){changed();render();}drag=null;renderCanvas();}
svg.addEventListener('pointerup',release);svg.addEventListener('pointercancel',release);
svg.addEventListener('wheel',event=>{event.preventDefault();const before=metric(event),px=-before[1],py=-before[0],factor=Math.exp(Math.max(-1,Math.min(1,event.deltaY*.001)));const next=Math.max(8,Math.min(600,camera.h*factor)),scale=next/camera.h;camera={x:px+(camera.x-px)*scale,y:py+(camera.y-py)*scale,w:camera.w*scale,h:next};renderCanvas();},{passive:false});
svg.addEventListener('dblclick',event=>{
 if(draft){if(draft.mode==='polyline')finishDraw();return;}
 const e=current();if(!e||e.mode==='bezier')return;const p=metric(event,true);let best=Infinity,at=0;
 const n=e.mode==='polygon'?e.points.length:e.points.length-1;
 for(let i=0;i<n;i++){const a=e.points[i],b=e.points[(i+1)%e.points.length],v=[b[0]-a[0],b[1]-a[1]],t=Math.max(0,Math.min(1,((p[0]-a[0])*v[0]+(p[1]-a[1])*v[1])/(v[0]**2+v[1]**2||1))),d=Math.hypot(p[0]-a[0]-t*v[0],p[1]-a[1]-t*v[1]);if(d<best){best=d;at=i;}}
 if(best>15*unit())return;remember();e.points.splice(at+1,0,p);changed();render();
});
$('entities').addEventListener('click',event=>{const b=event.target.closest('[data-select]');if(b){selected=b.dataset.select;measurementProbe=null;draft=null;$('finish').hidden=true;render();}});
$('nodes').addEventListener('change',event=>{if(!event.target.dataset.coordinate)return;const [i,k]=event.target.dataset.coordinate.split(',').map(Number),v=Number(event.target.value);if(!Number.isFinite(v)){toast('请输入有效坐标',true);renderProperties();return;}const e=current();remember();const p=[...e.points[i]];p[k]=v;moveNode(e,i,p);measurementProbe=p;changed();render();});
$('nodes').addEventListener('click',event=>{const b=event.target.closest('[data-remove-node]');if(!b)return;remember();current().points.splice(Number(b.dataset.removeNode),1);changed();render();});
$('add-node').onclick=()=>{const e=current();if(!e)return;let at=0,best=-1,n=e.mode==='polygon'?e.points.length:e.points.length-1;for(let i=0;i<n;i++){const a=e.points[i],b=e.points[(i+1)%e.points.length],d=Math.hypot(a[0]-b[0],a[1]-b[1]);if(d>best){best=d;at=i;}}const a=e.points[at],b=e.points[(at+1)%e.points.length];remember();e.points.splice(at+1,0,[(a[0]+b[0])/2,(a[1]+b[1])/2]);changed();render();};
$('prop-id').onchange=()=>{const e=current(),id=$('prop-id').value.trim();if(!id||project.entities.some(v=>v!==e&&v.id===id)){toast('名称不能为空或重复',true);renderProperties();return;}remember();e.id=id;selected=id;changed();render();};
for(const [id,key] of [['confidence','confidence'],['color','color'],['style','line_style']])$(id).onchange=()=>{const e=current(),v=id==='confidence'?Number($(id).value):$(id).value;if(id==='confidence'&&(!Number.isFinite(v)||v<0||v>1)){toast('置信度需在0–1之间',true);renderProperties();return;}remember();e[key]=v;changed();render();};
$('duplicate').onclick=()=>{const e=current();if(!e||e.kind==='guide')return;remember();const c=clone(e);c.id=newId(c.kind);c.points=c.points.map(p=>[p[0],p[1]+.5]);project.entities.push(c);selected=c.id;changed();render();};
$('delete').onclick=deleteSelected;$('draw').onclick=startDraw;$('finish').onclick=finishDraw;$('kind').onchange=changeType;
$('add-numeric').onclick=()=>{const p=[['sx','sy'],['ex','ey']].map(ids=>ids.map(id=>Number($(id).value)));if(p.flat().some(v=>!Number.isFinite(v))){toast('请输入有效坐标',true);return;}const kind=$('kind').value;create(kind,['stop','cross'].includes(kind)?'rectangle':$('mode').value,p);};
$('undo').onclick=undoAction;$('redo').onclick=redoAction;$('fit').onclick=fit;$('reset-view').onclick=resetView;
$('name').onchange=()=>{const name=$('name').value.trim();if(!name){toast('场景名称不能为空',true);$('name').value=project.name;return;}remember();project.name=name;changed();};
$('nav').onchange=()=>{remember();project.navigation=$('nav').value;changed();render();};
function persistDraft(){try{localStorage.setItem('pathgen_scene_editor_v1',JSON.stringify(project));return true;}catch(e){return false;}}
window.addEventListener('pagehide',persistDraft);
function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function connection(online){$('connection').textContent=online?'本地服务已连接':'服务已断开 · 可下载草稿';$('connection').classList.toggle('offline',!online);}
async function checkConnection(){try{const r=await fetch('/api/health',{cache:'no-store',signal:AbortSignal.timeout(3000)});const data=await r.json();connection(r.ok&&data.ok===true);if(r.ok&&data.ok&&data.editor_version!==3){$('connection').textContent='后台版本较旧 · 请重启服务';$('connection').classList.add('offline');}}catch(e){connection(false);}}
async function request(route,value){
 const revision=dirty;let response;
 try{response=await fetch(route,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});}
 catch(e){connection(false);const saved=persistDraft();throw Error('无法连接本地服务。'+(saved?'草稿已保存在本浏览器；':'请先下载编辑草稿；')+'请启动 python scene_editor.py --background，然后重试保存。');}
 connection(true);let result;try{result=await response.json();}catch(e){throw Error('服务没有返回有效 JSON，请检查启动终端');}if(!response.ok)throw Error(result.error||'请求失败');return {result,revision};
}
checkConnection();setInterval(checkConnection,10000);
$('save').onclick=async()=>{const b=$('save');b.disabled=true;b.textContent='正在采样并保存…';try{const {result,revision}=await request('/api/save',clone(project));if(revision===dirty){lastSave=result;$('result').hidden=false;$('saved-path').textContent=result.input_path;$('command').value=result.command;$('saved-preview').hidden=!result.preview_url;if(result.preview_url)$('saved-preview').href=result.preview_url;}toast(`已保存：Guide 81 点，BEV 保留 ${result.summary.retained}/${result.summary.supplied} 个元素`+(result.summary.retained<result.summary.supplied?'。超出32个的元素按模型规则筛选。':''));}catch(e){toast(e.message,true);}finally{b.disabled=false;b.textContent='保存模型输入';}};
$('preview').onclick=async()=>{const b=$('preview');b.disabled=true;try{const {result,revision}=await request('/api/preview',clone(project));if(revision!==dirty)return;sampled=result;renderCanvas();toast(`圆点为实际模型采样：Guide 81 点；BEV ${result.summary.retained} 个元素，每个32点`);}catch(e){toast(e.message,true);}finally{b.disabled=false;}};
$('copy-command').onclick=async()=>{try{await navigator.clipboard.writeText($('command').value);toast('推理命令已复制');}catch(e){$('command').select();toast('请按 Ctrl+C 复制命令');}};
$('download-input').onclick=()=>{if(lastSave)download(lastSave.input,'input.json');};
$('draft').onclick=()=>download(project,(project.name||'scene')+'.scene.json');
function setProject(next){remember();project=next;measurementProbe=null;selected=project.entities[0]?.id??null;draft=null;$('finish').hidden=true;changed();render();fit();}
$('import').onclick=()=>$('file').click();$('file').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{const {result}=await request('/api/import',JSON.parse(await file.text()));setProject(result);toast('已导入，可继续拖动编辑。');}catch(e){toast('导入失败：'+e.message,true);}finally{event.target.value='';}};
$('example').onclick=async()=>{try{const response=await fetch('/api/example');if(!response.ok)throw Error('读取示例失败');setProject(await response.json());toast('已载入现有分流示例；原始文件不会被修改。');}catch(e){toast(e.message,true);}};
$('new').onclick=()=>{if(project.entities.length&&!confirm('新建空场景？当前内容可用撤销恢复。'))return;setProject(empty());resetView();};
window.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;if(event.code==='Space'){space=true;event.preventDefault();}if(event.key==='Escape'){cancelDraw();}else if(event.key==='Enter'&&draft)finishDraw();else if(event.key==='Delete'||event.key==='Backspace'){event.preventDefault();deleteSelected();}else if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();event.shiftKey?redoAction():undoAction();}else if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='y'){event.preventDefault();redoAction();}});
window.addEventListener('keyup',event=>{if(event.code==='Space')space=false;});window.addEventListener('blur',()=>{space=false;release();});
new ResizeObserver(()=>{const ratio=svg.clientWidth/(svg.clientHeight||1),cx=camera.x+camera.w/2;camera.w=camera.h*ratio;camera.x=cx-camera.w/2;renderCanvas();}).observe($('canvas-wrap'));
try{const saved=localStorage.getItem('pathgen_scene_editor_v1');if(saved){const value=JSON.parse(saved);if(value.version===1&&Array.isArray(value.entities)&&value.entities.length<=256){project=value;selected=project.entities[0]?.id??null;}}}catch(e){/* A corrupt draft does not prevent starting the editor. */}
render();resetView();
