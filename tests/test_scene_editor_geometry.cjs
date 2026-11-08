const assert=require('node:assert/strict');
const {rotate,center}=require('../tools/scene_editor/geometry.js');
const {measure}=require('../tools/scene_editor/measurement.js');
const {translate,arrowDelta}=require('../tools/scene_editor/geometry.js');
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
// Screen arrows map to x-forward/y-left, and move every vertex/control point.
assert.deepEqual(arrowDelta('ArrowUp'),[.1,0]);
assert.deepEqual(arrowDelta('ArrowRight',true),[0,-1]);
assert.deepEqual(arrowDelta('ArrowLeft',false,true),[0,.01]);
assert.deepEqual(arrowDelta('ArrowDown',true,true),[-.01,0]);
for(const shape of [line,rect,controls]){
 const before=JSON.stringify(shape),delta=arrowDelta('ArrowRight',true);
 close(translate(shape,delta),shape.map(([x,y])=>[x,y-1]));
 assert.equal(JSON.stringify(shape),before);
 close(translate(translate(shape,delta),[0,1]),shape);
}
console.log('PASS: arrow directions/steps and rigid translation of line, box and Bezier handles.');
const {localCenter}=require('../tools/scene_editor/geometry.js');
// Unequal sampling and reversed boundary point order do not change the centre.
const hand=[[0,0],[10,.7],[85,.7]],left=[[0,2],[85,2]],right=[[85,-2],[60,-2],[25,-2],[0,-2]];
const savedHand=JSON.stringify(hand),centred=localCenter(hand,left,right,20,70,5);
const near=centred.points.filter(p=>p[0]>30&&p[0]<60);
assert(near.length>30);assert(near.every(p=>Math.abs(p[1])<1e-9));
assert.equal(JSON.stringify(hand),savedHand);close([centred.points[0]],[[0,0]]);
assert(centred.points.filter(p=>p[0]>71).every(p=>Math.abs(p[1]-.7)<1e-9));
assert.throws(()=>localCenter(hand,left,right,20,70,30),/过渡/);
assert.throws(()=>localCenter(hand,right,left,20,70,5),/交点/);
assert.throws(()=>localCenter(hand,[[40,2],[85,2]],right,20,70,5),/交点/);
assert.throws(()=>localCenter(hand,left,[[0,-20],[85,-20]],20,70,5),/交点/);
// Circular path: local normals recover the concentric lane centre, not matched indices.
const arc=(r,count)=>Array.from({length:count},(_,i)=>{const t=(i/(count-1))*.8;return [r*Math.sin(t),50-r*Math.cos(t)];});
const curved=localCenter(arc(49.5,801),arc(48,901),arc(52,701).reverse(),5,30,3);
assert(curved.points.filter(p=>p[0]>10&&p[0]<24).every(p=>Math.abs(Math.hypot(p[0],p[1]-50)-50)<.001));
console.log('PASS: local normal centering, uneven/reversed boundary samples, curved lanes, unchanged outside interval, invalid/missing boundary rejection.');
const pieces=[[[0,-2],[45,-2]],[[45,-2],[85,-2]]];
const grouped=localCenter(hand,[left],pieces,20,70,5);
assert(grouped.points.filter(p=>p[0]>30&&p[0]<60).every(p=>Math.abs(p[1])<1e-9));
assert.throws(()=>localCenter(hand,[left],[[[0,-2],[44,-2]],[[46,-2],[85,-2]]],20,70,5),/交点/);
assert.throws(()=>localCenter(hand,[left],[right,[[0,-3],[85,-3]]],20,70,5),/多次/);
for(const mode of ['left','right']){
 const single=localCenter(hand,mode==='left'?[left]:[],mode==='right'?pieces:[],20,70,5,{mode,offset:1.75});
 const expected=mode==='left'?.25:-.25;
 assert(single.widthMin===null && single.widthMax===null);
 assert(single.points.filter(p=>p[0]>30&&p[0]<60).every(p=>Math.abs(p[1]-expected)<1e-9));
}
const reversedSingle=localCenter(hand,[[...left].reverse()],[],20,70,5,{mode:'left',offset:1.75});
assert(reversedSingle.points.filter(p=>p[0]>30&&p[0]<60).every(p=>Math.abs(p[1]-.25)<1e-9));
const curveSingle=localCenter(arc(49.5,801),[arc(48,901).reverse()],[],5,30,3,{mode:'left',offset:1.75});
assert(curveSingle.points.filter(p=>p[0]>10&&p[0]<24).every(p=>Math.abs(Math.hypot(p[0],p[1]-50)-49.75)<.002));
// Oblique line: distance is perpendicular to the boundary, not a constant ego-y shift.
const oblique=localCenter([[0,0],[85,0]],[[[0,3],[85,5]]],[],20,70,5,{mode:'left',offset:1.75});
assert(oblique.reference.every(([x,y])=>Math.abs((3+(2/85)*x-y)/Math.hypot(1,2/85)-1.75)<1e-9));
for(const offset of [0,-1,9,NaN])assert.throws(()=>localCenter(hand,[left],[],20,70,5,{mode:'left',offset}),/偏移距离/);
console.log('PASS: multiple fragments, shared endpoints, gap/conflicting overlap rejection; left/right offsets, reverse order, curved/oblique normal distances.');
