/* Metre-space point-to-polyline measurements; also usable by Node tests. */
(function(root){
 'use strict';
 function closest(point, points){
  let best=null;
  for(let i=1;i<points.length;i++){
   const a=points[i-1],b=points[i],dx=b[0]-a[0],dy=b[1]-a[1];
   const t=Math.max(0,Math.min(1,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy||1)));
   const q=[a[0]+t*dx,a[1]+t*dy],distance=Math.hypot(q[0]-point[0],q[1]-point[1]);
   if(!best||distance<best.distance)best={point:q,distance};
  }
  return best;
 }
 function measure(source,probe,candidates){
  const anchor=closest(probe,source);if(!anchor)return null;
  let best=null;
  for(const candidate of candidates){const hit=closest(anchor.point,candidate.points);if(hit&&(!best||hit.distance<best.distance))best={...hit,id:candidate.id,anchor:anchor.point};}
  return best;
 }
 const api={closest,measure};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.LineMeasurement=api;
})(globalThis);
