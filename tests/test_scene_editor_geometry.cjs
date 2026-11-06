const assert=require('node:assert/strict');
const {rotate,center}=require('../tools/scene_editor/geometry.js');
const {measure}=require('../tools/scene_editor/measurement.js');
const close=(a,b)=>a.forEach((p,i)=>p.forEach((v,k)=>assert(Math.abs(v-b[i][k])<1e-10)));
const line=[[0,1],[10,1],[20,2]],copy=JSON.stringify(line),pivot=line[0];
close(rotate([[0,0],[10,0]],90,[0,0]),[[0,0],[0,10]]);
const turned=rotate(line,37,pivot);close([turned[0]],[pivot]);close(rotate(turned,-37,pivot),line);assert.equal(JSON.stringify(line),copy);
const rect=[[10,-2],[12,-2],[12,2],[10,2]],c=center(rect,true),r=rotate(rect,31,c);
close([center(rotate(rect,90,c),true)],[c]);
for(let i=0;i<4;i++){const edge=(p,j)=>[p[(j+1)%4][0]-p[j][0],p[(j+1)%4][1]-p[j][1]],a=edge(rect,i),b=edge(r,i),next=edge(r,(i+1)%4);assert(Math.abs(Math.hypot(...a)-Math.hypot(...b))<1e-10);assert(Math.abs(b[0]*next[0]+b[1]*next[1])<1e-10);}
// A Bezier curve and all its handles undergo the same rigid transform.
const controls=[[0,0],[20,0],[40,15],[60,20]],t=.37,w=[(1-t)**3,3*(1-t)**2*t,3*(1-t)*t*t,t**3];
const point=p=>[0,1].map(k=>p.reduce((s,v,i)=>s+w[i]*v[k],0));
close([point(rotate(controls,28,[3,4]))],rotate([point(controls)],28,[3,4]));
close([center(rotate(controls,28,center(controls)))],[center(controls)]);
const candidates=[{id:'curb',points:[[0,.2],[80,.2]]},{id:'other_lane',points:[[0,3.5],[80,3.5]]}];
assert.equal(measure([[0,0],[80,0]],[40,0],candidates).id,'curb');
assert.equal(measure([[0,0],[80,0]],[40,0],candidates.filter(v=>v.id==='other_lane')).distance,3.5);
console.log('PASS: rotation direction, fixed pivot, inverse, rectangle shape, Bezier affine invariance, explicit farther reference despite nearby curb.');
