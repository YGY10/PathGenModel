// Optional browser smoke test: run the editor, then node tests/scene_editor_browser_smoke.cjs
const {spawn}=require('child_process'),fs=require('fs'),os=require('os'),path=require('path'),assert=require('assert');
const root=path.resolve(__dirname,'..');
const base=path.join(root,'outputs/scene_editor');fs.mkdirSync(base,{recursive:true});
const out=fs.mkdtempSync(path.join(base,'browser_validation_'));
const browser=spawn(process.env.CHROME_BIN||'/usr/bin/google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-first-run','--no-default-browser-check','--remote-debugging-pipe','--user-data-dir='+fs.mkdtempSync(path.join(os.tmpdir(),'bev-editor-chrome-'))],{stdio:['ignore','ignore','pipe','pipe','pipe']});
let buffer=Buffer.alloc(0),seq=0,pending=new Map(),errors=[];
browser.stderr.on('data',()=>{});
browser.stdio[4].on('data',data=>{buffer=Buffer.concat([buffer,data]);let at;while((at=buffer.indexOf(0))>=0){let msg=JSON.parse(buffer.subarray(0,at));buffer=buffer.subarray(at+1);if(msg.id&&pending.has(msg.id)){const p=pending.get(msg.id);pending.delete(msg.id);msg.error?p.reject(Error(JSON.stringify(msg.error))):p.resolve(msg.result);}if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params.exceptionDetails.text+': '+JSON.stringify(msg.params.exceptionDetails.exception));}});
function send(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});browser.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');});}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));let session;
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},session);if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function wait(expression,limit=25000){const start=Date.now();while(Date.now()-start<limit){if(await evaluate(expression))return;await sleep(100);}throw Error('Timed out: '+expression);}
async function screenshot(name){const r=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false},session);fs.writeFileSync(path.join(out,name),Buffer.from(r.data,'base64'));}
async function click(x,y,count=1){await send('Input.dispatchMouseEvent',{type:'mousePressed',x,y,button:'left',clickCount:count},session);await send('Input.dispatchMouseEvent',{type:'mouseReleased',x,y,button:'left',clickCount:count},session);}
async function set(id,value,change=false){await evaluate(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(value)};${change?`document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event('change',{bubbles:true}));`:''}`);}
async function button(id){await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);}
const timer=setTimeout(()=>{browser.kill('SIGKILL');console.error('Browser test timeout');process.exit(1);},90000);
(async()=>{try{
 const target=await send('Target.createTarget',{url:'about:blank'});session=(await send('Target.attachToTarget',{targetId:target.targetId,flatten:true})).sessionId;
 await send('Page.enable',{},session);await send('Runtime.enable',{},session);await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1050,deviceScaleFactor:1,mobile:false},session);
 await send('Page.navigate',{url:process.env.BEV_EDITOR_URL||'http://127.0.0.1:8765'},session);await wait(`document.getElementById('add-numeric') && document.getElementById('guide-status').textContent.length>0`);
 await wait(`document.getElementById('connection').textContent.includes('已连接')`);
 await button('add-numeric');
 await set('kind','line',true);await set('sy','1.75');await set('ey','1.75');await button('add-numeric');
 assert((await evaluate(`document.getElementById('distance-info').textContent`)).includes('没有可测量'));
 await evaluate(`document.getElementById('measure-kind').value='all';document.getElementById('measure-kind').dispatchEvent(new Event('change'));`);
 assert((await evaluate(`document.getElementById('distance-info').textContent`)).includes('1.75 m'));
 await set('confidence','.63',true);await set('color','yellow',true);await set('style','solid',true);
 await set('kind','curb',true);await set('sy','-2.3');await set('ey','-5');await button('add-numeric');await set('confidence','.81',true);
 await set('nav','right',true);await set('name','browser_validation',true);
 await evaluate(`document.querySelector('[data-select="guide_1"]').click()`);
 const priorDistance=await evaluate(`document.getElementById('distance-info').textContent`);
 const handle=await evaluate(`(()=>{const b=document.querySelector('[data-node="1"]').getBoundingClientRect();return {x:b.x+b.width/2,y:b.y+b.height/2}})()`);
 await send('Input.dispatchMouseEvent',{type:'mousePressed',x:handle.x,y:handle.y,button:'left',clickCount:1},session);
 await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:handle.x+24,y:handle.y-14,buttons:1},session);
 await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:handle.x+24,y:handle.y-14,button:'left',clickCount:1},session);
 assert(await evaluate(`document.querySelectorAll('#measurement path').length===1`));
 assert.notEqual(await evaluate(`document.getElementById('distance-info').textContent`),priorDistance,'Dragging must update the local distance');
 const moved=await evaluate(`document.querySelector('[data-coordinate="1,1"]').value`);assert.notEqual(Number(moved),0,'Bezier handle must move');
 await button('undo');assert.equal(Number(await evaluate(`document.querySelector('[data-coordinate="1,1"]').value`)),0);await button('redo');
 // Actual mouse-drawn polyline, completed with Enter.
 await set('kind','line',true);await set('mode','polyline');await button('draw');
 for(const [x,y] of [[0,-8],[30,-9],[80,-8]]){const screen=await evaluate(`(()=>{const p=document.getElementById('canvas').createSVGPoint();p.x=${-y};p.y=${-x};const q=p.matrixTransform(document.getElementById('canvas').getScreenCTM());return {x:q.x,y:q.y}})()`);await click(screen.x,screen.y);}
 await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13},session);await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13},session);
 assert.equal(await evaluate(`document.getElementById('count').textContent`),'4');assert.equal(await evaluate(`document.querySelectorAll('#nodes .node').length`),3);
 await button('preview');await wait(`document.querySelectorAll('#samples circle').length>81`);
 await button('save');await wait(`!document.getElementById('result').hidden`);
 const input=await evaluate(`document.getElementById('saved-path').textContent`);const row=JSON.parse(fs.readFileSync(input));assert.equal(row.navigation,'right');assert.equal(row.reference_line.length,81);assert.equal(row.lane_markings[0].confidence,.63);assert.equal(row.lane_markings[0].color,'yellow');assert.equal(row.lane_markings[0].line_style,'solid');assert.equal(row.road_boundaries[0].confidence,.81);
 fs.writeFileSync(path.join(out,'saved_input_path.txt'),input);
 await evaluate(`document.querySelector('[data-select="guide_1"]').click()`);await screenshot('editor.png');
 // Load example, then round-trip the exact editable project through file import.
 await button('example');await wait(`document.getElementById('name').value!=='browser_validation' && Number(document.getElementById('count').textContent)>0`);
 const doc=await send('DOM.getDocument',{},session),file=await send('DOM.querySelector',{nodeId:doc.root.nodeId,selector:'#file'},session);
 await send('DOM.setFileInputFiles',{files:[path.join(path.dirname(input),'scene.json')],nodeId:file.nodeId},session);
 await wait(`document.getElementById('name').value==='browser_validation'`);
 assert.equal(await evaluate(`document.getElementById('count').textContent`),'4');
 // Failed connection must preserve the draft, explain recovery, then allow saving again.
 await send('Network.enable',{},session);
 await send('Network.emulateNetworkConditions',{offline:true,latency:0,downloadThroughput:0,uploadThroughput:0},session);
 await button('save');await wait(`!document.getElementById('save').disabled`);
 assert(await evaluate(`document.getElementById('toast').textContent.includes('无法连接本地服务')`));
 assert.equal(await evaluate(`JSON.parse(localStorage.getItem('pathgen_scene_editor_v1')).name`),'browser_validation');
 assert(await evaluate(`document.getElementById('connection').classList.contains('offline')`));
 await send('Network.emulateNetworkConditions',{offline:false,latency:0,downloadThroughput:-1,uploadThroughput:-1},session);
 await button('save');await wait(`!document.getElementById('save').disabled`);
 assert(await evaluate(`!document.getElementById('result').hidden && !document.getElementById('connection').classList.contains('offline')`));
 // Rigid curve rotation: fixed start, preserved shape, undo/redo.
 await evaluate(`document.querySelector('[data-select="line_1"]').click()`);
 await set('rotate-pivot','start');await set('rotate-angle','30');await button('rotate-apply');
 assert(Math.abs(Number(await evaluate(`document.querySelector('[data-coordinate="0,1"]').value`))-1.75)<1e-8);
 assert(Math.abs(Number(await evaluate(`document.querySelector('[data-coordinate="3,1"]').value`))-44.25)<1e-3);
 await button('undo');assert.equal(Number(await evaluate(`document.querySelector('[data-coordinate="3,1"]').value`)),1.75);
 await button('redo');assert(Math.abs(Number(await evaluate(`document.querySelector('[data-coordinate="3,1"]').value`))-44.25)<1e-3);await button('undo');
 // A curb 20cm away must not prevent locking a farther lane reference.
 await set('kind','curb',true);await set('sx','0');await set('ex','85');await set('sy','1.95');await set('ey','1.95');await button('add-numeric');
 await evaluate(`document.querySelector('[data-select="line_1"]').click()`);await set('measure-kind','boundaries',true);await set('measure-target','',true);
 assert((await evaluate(`document.getElementById('distance-info').textContent`)).includes('0.20 m'));
 await set('measure-target','line_2',true);
 assert((await evaluate(`document.getElementById('distance-info').textContent`)).includes('参照：line_2'));
 assert(!(await evaluate(`document.getElementById('distance-info').textContent`)).includes('0.20 m'));
 await button('rotate-plus');assert.equal(await evaluate(`document.getElementById('measure-target').value`),'line_2');await button('undo');
 // Rotate a box about its centre, then save its real polygon and PNG snapshot.
 await set('kind','stop',true);await set('sx','20');await set('ex','20.4');await set('sy','-2');await set('ey','2');await button('add-numeric');
 await set('rotate-pivot','center');await set('rotate-angle','90');await button('rotate-apply');
 assert(Math.abs(Number(await evaluate(`document.querySelector('[data-coordinate="0,0"]').value`))-22.2)<1e-6);
 assert(Math.abs(Number(await evaluate(`document.querySelector('[data-coordinate="0,1"]').value`))+.2)<1e-6);
 await button('undo');assert.equal(Number(await evaluate(`document.querySelector('[data-coordinate="0,0"]').value`)),20);await button('redo');
 await button('save');await wait(`!document.getElementById('save').disabled && !document.getElementById('result').hidden`);
 const imageLink=await evaluate(`document.getElementById('saved-preview').href`);
 const signature=await evaluate(`(async()=>{const r=await fetch(${JSON.stringify(imageLink)});if(!r.ok)throw Error('PNG endpoint failed');return Array.from(new Uint8Array(await r.arrayBuffer()).slice(0,8));})()`);
 assert.deepEqual(signature,[137,80,78,71,13,10,26,10]);
 const rotatedPath=await evaluate(`document.getElementById('saved-path').textContent`),rotated=JSON.parse(fs.readFileSync(rotatedPath));
 assert(Math.abs(rotated.stop_lines[0].points[0][0]-22.2)<1e-8);assert(fs.existsSync(path.join(path.dirname(rotatedPath),'scene.png')));
 assert(fs.readFileSync(path.join(path.dirname(path.dirname(rotatedPath)),'gallery.html'),'utf8').includes(path.basename(path.dirname(rotatedPath))));
 await evaluate(`document.querySelector('[data-select="line_1"]').click();document.getElementById('measure-kind').value='line';document.getElementById('measure-kind').dispatchEvent(new Event('change'));document.getElementById('measure-target').value='line_2';document.getElementById('measure-target').dispatchEvent(new Event('change'));`);
 await screenshot('editor_features.png');
 // Arrows translate the whole selected box in screen directions, including fine steps.
 await evaluate(`document.querySelector('[data-select="stop_1"]').click()`);
 const boxBefore=await evaluate(`JSON.parse(JSON.stringify(current().points))`);
 async function arrow(key,modifiers=0){await send('Input.dispatchKeyEvent',{type:'keyDown',key,code:key,modifiers},session);await send('Input.dispatchKeyEvent',{type:'keyUp',key,code:key},session);}
 await arrow('ArrowUp');await arrow('ArrowRight',8);await arrow('ArrowLeft',1);
 const boxAfter=await evaluate(`current().points`);
 boxAfter.forEach((p,i)=>{assert(Math.abs(p[0]-boxBefore[i][0]-.1)<1e-9);assert(Math.abs(p[1]-boxBefore[i][1]+.99)<1e-9);});
 await button('undo');await button('undo');await button('undo');
 assert.deepEqual(await evaluate(`current().points`),boxBefore);await button('redo');await button('undo');
 // Input fields retain normal arrow behaviour and must not translate the object.
 await evaluate(`document.getElementById('rotate-angle').focus()`);await arrow('ArrowUp');
 assert.deepEqual(await evaluate(`current().points`),boxBefore);
 await evaluate(`document.activeElement.blur()`);
 // Create editable GT from Guide; duplicate is blocked; all model feature arrays stay identical.
 const guideBefore=await evaluate(`JSON.stringify(project.entities.find(e=>e.kind==='guide'))`);
 await button('guide-to-gt');assert.equal(await evaluate(`current().kind`),'gt');
 assert.deepEqual(await evaluate(`current().points[0]`),[0,0]);
 assert.equal(await evaluate(`JSON.stringify(project.entities.find(e=>e.kind==='guide'))`),guideBefore);
 await button('guide-to-gt');assert.equal(await evaluate(`project.entities.filter(e=>e.kind==='gt').length`),1);
 await arrow('ArrowUp');assert.equal(await evaluate(`current().points[0][0]`),.1);await button('undo');
 await arrow('ArrowRight');assert.equal(await evaluate(`current().points[0][1]`),-.1);await button('undo');
 await button('preview');await wait(`sampled && sampled.gt && sampled.gt.length===81`);
 await button('save');await wait(`!document.getElementById('save').disabled && !document.getElementById('result').hidden`);
 const gtPath=await evaluate(`document.getElementById('saved-path').textContent`),gtInput=JSON.parse(fs.readFileSync(gtPath));
 assert.equal(gtInput.gt_path.length,81);assert.deepEqual(gtInput.gt_path[0],[0,0]);
 assert.equal(await evaluate(`sampled.summary.supplied`),gtInput.lane_markings.length+gtInput.road_boundaries.length+gtInput.stop_lines.length+gtInput.crosswalks.length);
 const savedProject=JSON.parse(fs.readFileSync(path.join(path.dirname(gtPath),'scene.json')));
 assert.equal(savedProject.entities.filter(e=>e.kind==='gt').length,1);
 await screenshot('editor_gt_arrows.png');
 // Saved input GT imports as an editable line. Direct drawing of GT works after deletion.
 await evaluate(`setProject(${JSON.stringify(savedProject)})`);
 await evaluate(`selected=project.entities.find(e=>e.kind==='gt').id;render()`);await button('delete');
 await set('kind','gt',true);await set('mode','polyline');await set('sx','0');await set('sy','0');await set('ex','85');await set('ey','0');await button('add-numeric');
 assert.equal(await evaluate(`current().kind`),'gt');assert.equal(await evaluate(`current().mode`),'polyline');
 console.log('PASS: GT copy/edit/save/sampling and manual creation; duplicate guard; arrows for box/GT, modifier steps, undo/redo, input focus guard.');
 // Short Guide must produce a visible error and never a false success.
 await evaluate(`document.querySelector('[data-select="guide_1"]').click();const f=document.querySelector('[data-coordinate="3,0"]');f.value='50';f.dispatchEvent(new Event('change',{bubbles:true}));`);
 await button('save');await wait(`!document.getElementById('save').disabled`);
 assert(await evaluate(`document.getElementById('toast').textContent.includes('至少 80')`));assert(await evaluate(`document.getElementById('result').hidden`));
 // Local centering is preview-only until applied, archives hand GT, and invalidates stale previews.
 const centerProject={version:1,name:'centering_validation',navigation:'straight',entities:[
 {id:'guide',kind:'guide',mode:'polyline',points:[[0,0],[85,0]]},
 {id:'gt',kind:'gt',mode:'polyline',points:[[0,0],[10,.7],[85,.7]]},
 {id:'left',kind:'line',mode:'polyline',points:[[0,2],[85,2]],color:'white',line_style:'dashed'},
 {id:'right',kind:'line',mode:'polyline',points:[[85,-2],[25,-2],[0,-2]],color:'white',line_style:'solid'}]};
 await evaluate(`setProject(${JSON.stringify(centerProject)});selected='gt';render()`);
 await set('center-left','left');await set('center-right','right');await set('center-start','20');await set('center-end','70');await set('center-blend','5');
 await button('center-preview');assert(await evaluate(`!centerPreview && document.getElementById('toast').textContent.includes('可信')`));
 await evaluate(`document.getElementById('center-trusted').checked=true`);
 const originalCenterGT=await evaluate(`JSON.stringify(current().points)`);
 await button('center-preview');assert(await evaluate(`!!centerPreview && !document.getElementById('center-apply').disabled`));
 assert.equal(await evaluate(`JSON.stringify(current().points)`),originalCenterGT);
 await button('center-cancel');assert(await evaluate(`!centerPreview`));
 await button('center-preview');await set('center-end','65');await evaluate(`document.getElementById('center-end').dispatchEvent(new Event('input',{bubbles:true}))`);
 assert(await evaluate(`!centerPreview && document.getElementById('center-apply').disabled`));
 await set('center-end','70');await button('center-preview');await screenshot('centering_preview.png');await button('center-apply');
 assert(await evaluate(`current().mode==='polyline' && current().points.length>200 && !!current().original_handdrawn`));
 assert.equal(await evaluate(`JSON.stringify(current().original_handdrawn.points)`),originalCenterGT);
 assert(await evaluate(`current().points.filter(p=>p[0]>30&&p[0]<60).every(p=>Math.abs(p[1])<1e-9)`));
 await button('undo');assert.equal(await evaluate(`JSON.stringify(current().points)`),originalCenterGT);await button('redo');
 await button('save');await wait(`!document.getElementById('save').disabled && !document.getElementById('result').hidden`);
 const centeredSave=await evaluate(`document.getElementById('saved-path').textContent`),centerScene=JSON.parse(fs.readFileSync(path.join(path.dirname(centeredSave),'scene.json')));
 assert(centerScene.entities.find(e=>e.kind==='gt').original_handdrawn);
 await evaluate(`setProject(${JSON.stringify(centerScene)});selected='gt';render()`);
 assert.equal(await evaluate(`current().centering_history.length`),1);
 await button('center-restore');assert.equal(await evaluate(`JSON.stringify(current().points)`),originalCenterGT);await button('undo');assert(await evaluate(`current().points.length>200`));
 // Multiple disjoint objects representing the same continuous boundary, plus single-side modes.
 const groupedProject=JSON.parse(JSON.stringify(centerProject));groupedProject.entities=groupedProject.entities.filter(e=>e.id!=='right');
 groupedProject.entities.push({id:'right_a',kind:'line',mode:'polyline',points:[[0,-2],[45,-2]]},{id:'right_b',kind:'line',mode:'polyline',points:[[85,-2],[45,-2]]});
 await evaluate(`setProject(${JSON.stringify(groupedProject)});selected='gt';render()`);
 async function chooseMany(id,values){await evaluate(`(()=>{const s=document.getElementById(${JSON.stringify(id)});for(const o of s.options)o.selected=${JSON.stringify(values)}.includes(o.value);s.dispatchEvent(new Event('input',{bubbles:true}));})()`);}
 await chooseMany('center-left',['left']);await chooseMany('center-right',['right_a','right_b']);
 await set('center-mode','both',true);await button('center-preview');assert(await evaluate(`!!centerPreview && centerPreview.result.points.filter(p=>p[0]>30&&p[0]<60).every(p=>Math.abs(p[1])<1e-9)`));
 await screenshot('boundary_groups_preview.png');
 await set('center-mode','left',true);assert(await evaluate(`!centerPreview && document.getElementById('center-right-wrap').hidden`));
 await set('center-width','3.5');await button('center-half-width');assert.equal(await evaluate(`document.getElementById('center-offset').value`),'1.75');
 await button('center-preview');assert(await evaluate(`!!centerPreview && centerPreview.result.reference.every(p=>Math.abs(p[1]-.25)<1e-9)`));
 await screenshot('single_left_preview.png');await button('center-apply');
 assert(await evaluate(`current().centering_history.at(-1).mode==='left' && current().centering_history.at(-1).left[0]==='left' && current().centering_history.at(-1).right.length===0`));
 await button('save');await wait(`!document.getElementById('save').disabled && !document.getElementById('result').hidden`);
 const singleSave=await evaluate(`document.getElementById('saved-path').textContent`),singleScene=JSON.parse(fs.readFileSync(path.join(path.dirname(singleSave),'scene.json')));
 assert.equal(singleScene.entities.find(e=>e.kind==='gt').centering_history.at(-1).offset,1.75);
 await button('undo');await set('center-mode','right',true);await button('center-preview');
 assert(await evaluate(`!!centerPreview && centerPreview.result.reference.every(p=>Math.abs(p[1]+.25)<1e-9)`));
 await set('center-offset','0');await evaluate(`document.getElementById('center-offset').dispatchEvent(new Event('input',{bubbles:true}))`);await button('center-preview');assert(await evaluate(`!centerPreview && document.getElementById('toast').textContent.includes('偏移距离')`));
 await set('center-mode','both',true);
 console.log('PASS: multi-select fragments, left/right reference, width-to-offset, metadata persistence and invalid offset guard.');
 // Preview the user's straight case, without applying or saving any changes to it.
 const actual=path.join(root,'outputs/scene_editor/20261107_230623_690212/scene.json');
 if(fs.existsSync(actual)){
  const real=JSON.parse(fs.readFileSync(actual));await evaluate(`setProject(${JSON.stringify(real)});selected=project.entities.find(e=>e.kind==='gt').id;render()`);
  await set('center-left','line_1');await set('center-right','line_2');await set('center-start','30');await set('center-end','65');await set('center-blend','5');
  await button('center-preview');assert(await evaluate(`!!centerPreview`));await screenshot('centering_user_straight_preview.png');
 }
 console.log('PASS: local centre preview/cancel/apply, confirmation, invalidation, preservation, undo/redo and save/reimport.');
 assert.deepEqual(errors,[],'No uncaught browser errors');
 console.log('PASS: numeric creation; mouse Bezier drag; undo/redo; mouse polyline; semantics/nav editing; sampling preview; distance display; save; import round trip; offline draft and reconnect save; short-Guide error; curve/box rotation; explicit farther reference; saved PNG endpoint.');console.log('Saved input:',input);console.log('Screenshot:',path.join(out,'editor.png'));
 }finally{clearTimeout(timer);try{await send('Browser.close');}catch(e){}browser.kill();}})().catch(e=>{console.error(e.stack);process.exitCode=1;browser.kill('SIGKILL');});
