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
 const api={center,rotate};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.SceneGeometry=api;
})(globalThis);
