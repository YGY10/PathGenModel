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
 assert(fs.readFileSync(path.join(root,'outputs/scene_editor/gallery.html'),'utf8').includes(path.basename(path.dirname(rotatedPath))));
 await evaluate(`document.querySelector('[data-select="line_1"]').click();document.getElementById('measure-kind').value='line';document.getElementById('measure-kind').dispatchEvent(new Event('change'));document.getElementById('measure-target').value='line_2';document.getElementById('measure-target').dispatchEvent(new Event('change'));`);
 await screenshot('editor_features.png');
 // Short Guide must produce a visible error and never a false success.
 await evaluate(`document.querySelector('[data-select="guide_1"]').click();const f=document.querySelector('[data-coordinate="3,0"]');f.value='50';f.dispatchEvent(new Event('change',{bubbles:true}));`);
 await button('save');await wait(`!document.getElementById('save').disabled`);
 assert(await evaluate(`document.getElementById('toast').textContent.includes('至少 80')`));assert(await evaluate(`document.getElementById('result').hidden`));
 assert.deepEqual(errors,[],'No uncaught browser errors');
 console.log('PASS: numeric creation; mouse Bezier drag; undo/redo; mouse polyline; semantics/nav editing; sampling preview; distance display; save; import round trip; offline draft and reconnect save; short-Guide error; curve/box rotation; explicit farther reference; saved PNG endpoint.');console.log('Saved input:',input);console.log('Screenshot:',path.join(out,'editor.png'));
 }finally{clearTimeout(timer);try{await send('Browser.close');}catch(e){}browser.kill();}})().catch(e=>{console.error(e.stack);process.exitCode=1;browser.kill('SIGKILL');});
