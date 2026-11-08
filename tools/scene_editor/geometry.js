/* Rigid transforms in ego metres: positive angle turns +x towards +y (left). */
(function(root){
 'use strict';
 function center(points,closed=false){
  let x=0,y=0,total=0;
  const count=closed?points.length:points.length-1;
  for(let i=0;i<count;i++){
   const a=points[i],b=points[(i+1)%points.length],weight=Math.hypot(b[0]-a[0],b[1]-a[1]);
   x+=(a[0]+b[0])*.5*weight;y+=(a[1]+b[1])*.5*weight;total+=weight;
  }
  return total?[x/total,y/total]:[...points[0]];
 }
 function rotate(points,degrees,pivot){
  if(!Number.isFinite(degrees)||!pivot.every(Number.isFinite))throw Error('旋转角度和支点必须为有限数值');
  const angle=(degrees%360)*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle);
  return points.map(p=>{const x=p[0]-pivot[0],y=p[1]-pivot[1];return [pivot[0]+c*x-s*y,pivot[1]+s*x+c*y];});
 }
 function arrowDelta(key,shift=false,alt=false){
  const step=alt ? .01 : shift ? 1 : .1;
  return ({ArrowUp:[step,0],ArrowDown:[-step,0],ArrowLeft:[0,step],ArrowRight:[0,-step]})[key]||null;
 }
 function translate(points,delta){return points.map(p=>p.map((v,k)=>v+delta[k]));}
 function samplePath(points){
  const clean=points.filter((p,i)=>!i||Math.hypot(p[0]-points[i-1][0],p[1]-points[i-1][1])>1e-8);
  const arc=[0];for(let i=1;i<clean.length;i++)arc.push(arc.at(-1)+Math.hypot(clean[i][0]-clean[i-1][0],clean[i][1]-clean[i-1][1]));
  if(clean.length<2)throw Error('路径长度不足');
  function at(s){s=Math.max(0,Math.min(arc.at(-1),s));let lo=0,hi=arc.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(arc[mid]<=s)lo=mid;else hi=mid;}const t=(s-arc[lo])/(arc[hi]-arc[lo]);return clean[lo].map((v,k)=>v+t*(clean[hi][k]-v));}
  return {arc,at,total:arc.at(-1)};
 }
 function localCenter(gt,left,right,start,end,blend=5,options={}){
  const route=samplePath(gt),mode=options.mode||'both',offset=options.offset??1.75;
  if(!['both','left','right'].includes(mode))throw Error('无效参考模式');
  if(mode!=='both'&&(!Number.isFinite(offset)||offset<=0||offset>8))throw Error('单侧偏移距离须大于0且不超过8m');
  const group=lines=>!lines?.length?[]:typeof lines[0][0]==='number'?[lines]:lines;
  left=group(left);right=group(right);
  if(mode!=='right'&&!left.length||mode!=='left'&&!right.length)throw Error('请选择所需边界片段');
  if(![start,end,blend].every(Number.isFinite)||start<0||end>route.total+1e-6||end<=start||blend<=0||2*blend>end-start)
   throw Error('区间须在GT长度内，结束大于开始，过渡长度须大于0且不超过区间一半');
  const cross=(a,b)=>a[0]*b[1]-a[1]*b[0];
  function hit(p,n,lines,sign){
   const hits=[];
   for(const line of lines)for(let i=1;i<line.length;i++){
    const a=line[i-1],v=line[i].map((x,k)=>x-a[k]),q=a.map((x,k)=>x-p[k]),den=cross(n,v);
    if(Math.abs(den)<1e-9)continue;
    const d=cross(q,v)/den,u=cross(q,n)/den;
    if(u>=-1e-8&&u<=1+1e-8&&(sign===0||d*sign>1e-5)&&!hits.some(x=>Math.abs(x-d)<1e-5))hits.push(d);
   }
   const nearby=hits.filter(d=>Math.abs(d)<=8);
   if(nearby.length!==1)throw Error(nearby.length?'边界与局部横截面多次相交，请缩短区间或更换边界':'两侧缺少有效横截面交点：请检查边界方向、缺口或区间');
   return nearby[0];
  }
  // Offset each fragment independently along its own normal. Never join gaps.
  function offsetGroup(lines,side){
   const mid=route.at((start+end)/2),a=route.at((start+end)/2-.5),b=route.at((start+end)/2+.5),direction=[b[0]-a[0],b[1]-a[1]];
   return lines.map(line=>{
    let path=samplePath(line),near=0,best=Infinity;
    for(const station of path.arc){const p=path.at(station),d=Math.hypot(p[0]-mid[0],p[1]-mid[1]);if(d<best){best=d;near=station;}}
    const u=path.at(near-.5),v=path.at(near+.5);
    if((v[0]-u[0])*direction[0]+(v[1]-u[1])*direction[1]<0)path=samplePath([...line].reverse());
    const samples=[...path.arc],step=Math.max(.25,path.total/6000);
    for(let s=0;s<path.total;s+=step)samples.push(s);
    samples.sort((a,b)=>a-b);
    return samples.filter((s,i)=>!i||s-samples[i-1]>1e-8).map(s=>{
     const p=path.at(s),a=path.at(s-.1),b=path.at(s+.1),dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy);
     if(len<1e-8)throw Error('边界局部方向退化');
     return [p[0]+side*offset*dy/len,p[1]-side*offset*dx/len];
    });
   });
  }
  const shifted=mode==='both'?null:offsetGroup(mode==='left'?left:right,mode==='left'?1:-1);
  const stations=[...route.arc,start,end];const step=Math.max(.25,route.total/6000);
  for(let s=0;s<route.total;s+=step)stations.push(s);
  stations.sort((a,b)=>a-b);
  const unique=stations.filter((s,i)=>!i||s-stations[i-1]>1e-8),reference=[],widths=[];let maxShift=0;
  const smooth=t=>t*t*t*(10+t*(-15+6*t));
  const result=unique.map(s=>{
   const p=route.at(s);if(s<start-1e-8||s>end+1e-8)return p;
   const a=route.at(s-.5),b=route.at(s+.5),dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);
   if(length<1e-8)throw Error('GT局部方向退化');
   const n=[-dy/length,dx/length];let shift;
   if(mode==='both'){
    const l=hit(p,n,left,1),r=hit(p,n,right,-1),width=l-r;
    if(width<1||width>8)throw Error('候选车道宽度不在1–8m内，请检查所选边界');
    shift=(l+r)/2;widths.push(width);
   }else{
    hit(p,n,mode==='left'?left:right,mode==='left'?1:-1);
    shift=hit(p,n,shifted,0);
   }
   const weight=smooth(Math.max(0,Math.min(1,(s-start)/blend,(end-s)/blend)));
   const c=p.map((v,k)=>v+shift*n[k]);reference.push(c);maxShift=Math.max(maxShift,Math.abs(shift*weight));
   return p.map((v,k)=>v+shift*weight*n[k]);
  });
  if(result.length>8192)throw Error('GT节点过多，请简化路径后重试');
  return {points:result,reference,maxShift,widthMin:widths.length?Math.min(...widths):null,widthMax:widths.length?Math.max(...widths):null,mode,offset:mode==='both'?null:offset};
 }
 const api={center,rotate,arrowDelta,translate,localCenter};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.SceneGeometry=api;
})(globalThis);
