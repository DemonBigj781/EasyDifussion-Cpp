#!/usr/bin/env node
/* Focused state/request tests for the real portable UI scripts using a small
 * mocked DOM, fetch transport, and manually advanced progress timers.
 * No browser, server, model, or image decoder is exercised here. The PNG-shaped
 * fixture checks the preview/download URI only; real PNG validation belongs to
 * verify_inference.py and verify_inference_api.py.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert/strict');
const {webcrypto} = require('crypto');
const scripts = path.resolve(__dirname, '../resources/cpp-ui/scripts');
const generate = fs.readFileSync(path.join(scripts, 'generate.js'), 'utf8');
assert(generate.startsWith('// BEGIN VERBATIM SHARED EDITOR MASK TOOLS (source equality tested)\n'+fs.readFileSync(path.resolve(__dirname,'../../ui/media/js/editor-mask-tools.js'),'utf8')+'// END VERBATIM SHARED EDITOR MASK TOOLS\n'), 'Shared wand bundle must exactly match canonical source');
const kiosk = fs.readFileSync(path.join(scripts, 'kiosk.js'), 'utf8');
const caps = {protocol:1,mode:'native-single-user',kiosk_supported:false,kiosk_enabled:false,txt2img:true,img2img_ui:true,uploaded_mask_inpainting_ui:true,advanced_inpaint_ui:true,max_concurrent_generations:1};
const flush = async () => { for (let i=0;i<12;i++) await new Promise(r=>setImmediate(r)); };
function harness(mode='success', compute={backend:'cpu',provider:'auto',device:'auto'}, savedCheckpoint='') {
    const ids = new Map(), timers = new Map(), calls = [];
    let canvasAlpha=128, imageWidth=8, delayImages=false; const pendingImages=[]; const drawingCalls=[],canvasExports=[];
    let timerId=0, resolveGeneration, progressState={current_step:0,total_steps:4,completed:false};
    class Element {
        constructor(tag='div') { this.tag=tag;this.children=[];this.handlers={};this.dataset={};this._value='';this.disabled=false;this.attributes={};this.style={}; }
        set id(v) { this._id=v;ids.set(v,this); }
        get id() { return this._id; }
        set value(v) { this._value=String(v); }
        get value() { return this._value; }
        get options() { return this.children; }
        add(option) { this.children.push(option);if(this.children.length===1)this.value=option.value; }
        append(...children) { this.children.push(...children); }
        prepend(...children) { this.children.unshift(...children); }
        after() {}
        replaceWith(other) { ids.set(this.id,other); }
        replaceChildren(...children) { this.children=[...children];this.value=children[0]?.value??''; }
        addEventListener(name,handler) { this.handlers[name]=handler; }
        setAttribute(name,value) { this.attributes[name]=value; }
        removeAttribute(name) { delete this.attributes[name]; }
        remove() { this.removed=true; }
        getContext() { const element=this; return {drawImage(image){drawingCalls.push(['drawImage',image?._pixels?Array.from(image._pixels.data):null,this.globalCompositeOperation]);},getImageData(x,y,w=1,h=1){const data=new Uint8ClampedArray(w*h*4);for(let i=3;i<data.length;i+=4)data[i]=canvasAlpha;return {data};},createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};},putImageData(pixels){element._pixels=pixels;drawingCalls.push('undo');},clearRect(){drawingCalls.push('clear');},beginPath(){},moveTo(){},lineTo(){},stroke(){drawingCalls.push(['stroke',this.globalAlpha,this.globalCompositeOperation]);},arc(){},fill(){drawingCalls.push(['fill',this.globalAlpha,this.globalCompositeOperation]);}}; }
        toDataURL() {canvasExports.push({width:this.width,height:this.height,pixels:this._pixels?Array.from(this._pixels.data):null});const bytes=Buffer.alloc(26);bytes.write('PNG',1);bytes[25]=6;return 'data:image/png;base64,'+bytes.toString('base64');}
        getBoundingClientRect() { return {left:0,top:0,width:8,height:8}; }
        setPointerCapture() {}
    }
    const names=['width-value','height-value','makeImage','stopImage','generation-queue-status','generation-progress','preview-content','generation-model-status','show-download-popup','output_format','num_images','vae_model','text_encoder_model','generation-companion-status','cpp-modifier-status','cpp-modifier-grid','lora_model','generate-plugin-panels','cpp-image-modifiers','stable_diffusion_model','sampler','width','height','prompt','negative_prompt','seed','steps','guidance_scale','clear-all-previews','initial-text','kiosk-status-banner','kiosk-mode-enabled','kiosk-mode-save','kiosk-mode-save-status'];
    for(const id of names){const e=new Element();e.id=id;e.name=id;ids.set(id,e);}
    const scope = new Element();scope.querySelectorAll=()=>[...ids.values()];
    const values={width:128,height:128,steps:4,seed:42,guidance_scale:7,prompt:'a red cube',negative_prompt:''};
    let saved = {options:{sd_model_checkpoint:savedCheckpoint},inference:{image:{
        width:128,height:128,steps:4,seed:42,cfg_scale:7,prompt:'a red cube',negative_prompt:'',sampler_name:'euler_a',scheduler:'discrete',
    }}};
    for(const [id,value] of Object.entries(values))ids.get(id).value=value;
    const document={createTextNode:text=>({textContent:text}),getElementById:id=>ids.get(id),querySelector:()=>scope,createElement:tag=>new Element(tag),documentElement:new Element()};
    const response = (body,status=200) => ({ok:status<400,status,text:async()=>JSON.stringify(body),json:async()=>body});
    async function fetch(path,options={}) {
        const body=options.body?JSON.parse(options.body):undefined;calls.push({path,body});
        if(path.endsWith('cosmopolitan-capabilities')) {
            if(mode==='unavailable-capabilities')return response({message:'Unavailable'},503);
            return response(mode==='bad-capabilities'?{...caps,kiosk_enabled:true}:mode==='old-capabilities'?{...caps,img2img_ui:false}:mode==='img2img-only'?{...caps,uploaded_mask_inpainting_ui:false,advanced_inpaint_ui:false}:mode==='no-advanced'?{...caps,advanced_inpaint_ui:false}:caps);
        }
        if(path.endsWith('/config'))return response({saved,effective:{compute,...saved}});
        if(path.endsWith('/options'))return response(saved.options);
        if(path.endsWith('/settings')) {
            if(mode==='save-conflict')return response({message:'A native generation request is active'},409);
            saved=JSON.parse(JSON.stringify(body));
            return response({saved,effective:{compute,...saved}});
        }
        if(path.endsWith('backend-devices'))return response({default_webgpu_selector:'WebGPU0',devices:[
            {backend:'WebGPU',selector:'WebGPU0',provider:'native',software:false,type:'gpu',description:'First physical GPU',stable_id:'uuid:first',stable_id_available:true,memory_known:false,memory_total:null},
            {backend:'WebGPU',selector:'WebGPU1',provider:'native',software:false,type:'integrated-gpu',description:'Second physical GPU',stable_id:'uuid:second',stable_id_available:true,memory_known:false,memory_total:null},
            {backend:'WebGPU',selector:'WebGPU2',provider:'embedded',software:true,type:'software',description:'llvmpipe',stable_id:'',stable_id_available:false,memory_known:false,memory_total:null},
            {backend:'CPU',selector:'CPU',provider:'builtin',software:true,type:'cpu',memory_known:true,memory_total:8589934592},
        ]});
        if(path.endsWith('checkpoints'))return response({models:[{name:'model.safetensors'}]});
        if(path.endsWith('txt2img')||path.endsWith('img2img')) {
            if(mode==='conflict')return response({message:'Another generation is active'},409);
            if(mode==='network')throw new TypeError('network disconnected');
            return new Promise(resolve=>{resolveGeneration=value=>resolve(response(value));});
        }
        if(path.endsWith('/progress'))return response(progressState);
        if(path.endsWith('/interrupt'))return response({});
        throw Error('Unexpected endpoint: '+path);
    }
    const window={addEventListener(){}};
    const context=vm.createContext({window,document,fetch,crypto:webcrypto,Uint32Array,Uint8Array,console,atob:s=>Buffer.from(s,"base64").toString("binary"),
        File:class{constructor(parts,name,options){this.parts=parts;this.name=name;this.type=options.type;this.size=parts[0].length;}read(reader){reader.result='data:image/png;base64,'+Buffer.from(this.parts[0]).toString('base64');setImmediate(()=>reader.onload());}},
        FileReader:class{readAsDataURL(file){file.read(this);}},
        Image:class{set src(value){this.naturalWidth=imageWidth;this.naturalHeight=8;const complete=()=>value.includes('broken')?this.onerror():this.onload();if(delayImages)pendingImages.push(complete);else setImmediate(complete);}},
        Option:class extends Element{constructor(text,value){super('option');this.textContent=text;this.value=value;}},
        localStorage:{getItem(){throw Error('Generate must read native settings');},setItem(){throw Error('Generate must save native settings');}},
        setTimeout(fn){const id=++timerId;timers.set(id,fn);return id;},clearTimeout(id){timers.delete(id);}
    });
    vm.runInContext(kiosk,context);
    vm.runInContext(generate,context);
    return {ids,calls,window,drawingCalls,canvasExports,delayImages(value){delayImages=value;},resolveImages(){for(const complete of pendingImages.splice(0))complete();},setAlpha(value){canvasAlpha=value;},setWidth(value){imageWidth=value;},async upload(kind, data='data:image/png;base64,iVBORw0KGgo=') {
            const input=ids.get(`cosmo-${kind}-file`);
            input.files=[{type:'image/png',size:32,read(reader){reader.result=data;setImmediate(()=>reader.onload());}}];
            await input.handlers.change();await flush();
        }, beginUpload(kind) {
            let complete;const input=ids.get(`cosmo-${kind}-file`);
            input.files=[{type:'image/png',size:32,read(reader){complete=data=>{reader.result=data;reader.onload();};}}];
            input.handlers.change();return complete;
        },async click(id){await ids.get(id).handlers.click?.();await flush();},
        async start(){const promise=ids.get('makeImage').handlers.click();await flush();return {promise};},
        async tick(state){if(state)progressState=state;const first=timers.entries().next().value;if(first){timers.delete(first[0]);await first[1]();}await flush();},
        async finish(value){resolveGeneration(value);await flush();}};
}
const watchdog = setTimeout(() => {
    console.error('FAIL: asynchronous UI verification did not finish within 10 seconds');
    process.exit(1);
}, 10000);
(async()=>{
    const normal=harness();await flush();
    assert.equal(normal.ids.get('makeImage').disabled,false);
    for(const id of ['num_images','output_format','vae_model','text_encoder_model'])assert.equal(normal.ids.get(id).disabled,true,id);
    const running=await normal.start();
    assert.equal(normal.ids.get('makeImage').disabled,true);
    assert.equal(normal.ids.get('stopImage').disabled,true);
    assert.equal(normal.ids.get('cosmo-save-generation-defaults').disabled,true,'saving disabled during generation');
    const call=normal.calls.find(c=>c.path.endsWith('txt2img'));
    assert.equal(call.body.backend,'CPU');
    assert.equal(call.body.override_settings.sd_model_checkpoint,'model.safetensors');
    assert.equal(call.body.batch_size,1);
    await normal.tick({current_step:1,total_steps:4,completed:false});
    assert.equal(normal.ids.get('stopImage').disabled,false);
    await normal.click('stopImage');
    assert.equal(normal.calls.find(c=>c.path.endsWith('/interrupt')).body.id_task,call.body.force_task_id);
    assert.equal(normal.ids.get('makeImage').disabled,true);
    await normal.finish({images:[]});await running.promise;
    assert.equal(normal.ids.get('makeImage').disabled,false);
    assert.match(normal.ids.get('generation-queue-status').textContent,/stopped/);

    const png=harness();await flush();png.ids.get('cosmo-generation-backend').value='WebGPU2';const imageJob=await png.start();
    assert.equal(png.calls.find(c=>c.path.endsWith('txt2img')).body.backend,'WebGPU2');
    await png.finish({images:['iVBORw0KGgoAAAANSUhEUg==']});await imageJob.promise;
    const card=png.ids.get('preview-content').children[0];
    assert.equal(card.children[0].src,'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==');
    assert.equal(card.children[2].download,'image-42.png');
    assert(!png.calls.some(c=>c.path.endsWith('/settings')),'ordinary generation must not silently persist request overrides');

    const defaults=harness('success',{backend:'cpu',provider:'auto',device:'auto'},'model.safetensors');await flush();
    assert.equal(defaults.ids.get('stable_diffusion_model').value,'model.safetensors','native saved checkpoint loads without browser storage');
    assert.equal(defaults.ids.get('prompt').value,'a red cube','native saved recipe populates the form');
    defaults.ids.get('prompt').value='a blue sphere';defaults.ids.get('steps').value='8';defaults.ids.get('seed').value='-1';
    await defaults.click('cosmo-save-generation-defaults');
    const savedCall=defaults.calls.filter(c=>c.path.endsWith('/settings'));
    assert.equal(savedCall.length,1,'checkpoint and recipe use one atomic native mutation');
    assert.equal(savedCall[0].body.options.sd_model_checkpoint,'model.safetensors');
    assert.equal(savedCall[0].body.inference.image.prompt,'a blue sphere');
    assert.equal(savedCall[0].body.inference.image.steps,8);
    assert.equal(savedCall[0].body.inference.image.seed,-1,'random seed preference is saved without converting it into a one-request seed');
    assert.match(defaults.ids.get('generation-queue-status').textContent,/defaults saved/);
    assert.equal(defaults.ids.get('cosmo-save-generation-defaults').disabled,false);
    const missingCheckpoint=harness('success',{backend:'cpu',provider:'auto',device:'auto'},'missing.safetensors');await flush();
    assert.equal(missingCheckpoint.ids.get('makeImage').disabled,true,'missing saved checkpoint does not select a different model');
    assert.match(missingCheckpoint.ids.get('generation-model-status').textContent,/missing.safetensors.*unavailable/);
    const saveConflict=harness('save-conflict');await flush();await saveConflict.click('cosmo-save-generation-defaults');
    assert.match(saveConflict.ids.get('generation-queue-status').textContent,/Could not save.*active/);
    assert.equal(saveConflict.ids.get('makeImage').disabled,false,'failed persistence restores editable controls');

    const hardware=harness('success',{backend:'webgpu',provider:'auto',device:'uuid:second'});await flush();
    assert.equal(hardware.ids.get('cosmo-generation-backend').value,'WebGPU1','stable ID resolves the correct current selector');
    const choices=hardware.ids.get('cosmo-generation-backend').options;
    assert.equal(choices.length,4,'physical adapters must not collapse into one backend option');
    assert.match(choices.find(option=>option.value==='WebGPU2').textContent,/Software CPU/);
    assert.doesNotMatch(choices.find(option=>option.value==='WebGPU2').textContent,/Discrete GPU|Integrated GPU|GiB/);
    assert.match(choices.find(option=>option.value==='WebGPU1').textContent,/Integrated GPU/);
    const hardwareJob=await hardware.start();
    assert.equal(hardware.calls.find(c=>c.path.endsWith('txt2img')).body.backend,'WebGPU1');
    await hardware.finish({images:['iVBORw0KGgoAAAANSUhEUg==']});await hardwareJob.promise;
    const absent=harness('success',{backend:'webgpu',provider:'native',device:'uuid:missing'});await flush();
    assert.equal(absent.ids.get('makeImage').disabled,true,'missing saved device must not silently choose another');
    assert.equal(absent.ids.get('cosmo-generation-backend').value,'');
    assert.match(absent.ids.get('generation-queue-status').textContent,/unavailable/);
    absent.ids.get('cosmo-generation-backend').value='WebGPU0';
    absent.ids.get('cosmo-generation-backend').handlers.change();
    assert.equal(absent.ids.get('makeImage').disabled,false,'explicit alternate selection restores generation');

    const network=harness('network');await flush();await network.start();
    assert.equal(network.ids.get('makeImage').disabled,true,'uncertain request must stay blocked');
    await network.tick({current_step:1,total_steps:4,completed:false});
    assert.equal(network.ids.get('makeImage').disabled,true,'lost connection must stay blocked while the server is sampling');
    await network.tick({current_step:4,total_steps:4,completed:true});
    assert.equal(network.ids.get('makeImage').disabled,false);
    assert.match(network.ids.get('generation-queue-status').textContent,/not received/);

    const conflict=harness('conflict');await flush();await conflict.start();
    assert.equal(conflict.ids.get('makeImage').disabled,false);
    assert.match(conflict.ids.get('generation-queue-status').textContent,/Another generation/);

    const uploaded=harness();await flush();
    await uploaded.upload('initial');await uploaded.upload('mask');
    uploaded.ids.get('cosmo-image-strength').value='0.45';
    uploaded.ids.get('cosmo-mask-blur').value='2';uploaded.ids.get('cosmo-mask-invert').checked=true;
    await uploaded.click('cosmo-save-generation-defaults');
    const imageDefaults=uploaded.calls.find(c=>c.path.endsWith('/settings'));
    assert(!JSON.stringify(imageDefaults.body).includes('iVBOR'),'uploaded pixels persisted in recipe');
    const masked=await uploaded.start();
    const maskedCall=uploaded.calls.find(c=>c.path.endsWith('/img2img'));
    assert.equal(maskedCall.body.init_images[0],'iVBORw0KGgo=');
    assert.equal(maskedCall.body.mask,'iVBORw0KGgo=');
    assert.equal(maskedCall.body.denoising_strength,.45);assert.equal(maskedCall.body.mask_blur,2);
    assert.equal(maskedCall.body.inpainting_mask_invert,1);
    assert(uploaded.ids.get('cosmo-initial-file').disabled);
    await uploaded.finish({images:['iVBORw0KGgoAAAANSUhEUg==']});await masked.promise;
    await uploaded.click('cosmo-initial-clear');assert(uploaded.ids.get('cosmo-mask-preview').hidden);
    assert(uploaded.ids.get('cosmo-mask-file').disabled);
    const stale=uploaded.beginUpload('initial');assert(uploaded.ids.get('makeImage').disabled);
    await uploaded.upload('initial','data:image/png;base64,newimage');
    stale('data:image/png;base64,oldimage');await flush();
    assert.equal(uploaded.ids.get('cosmo-initial-preview').src,'data:image/png;base64,newimage');
    const cleared=uploaded.beginUpload('initial');await uploaded.click('cosmo-initial-clear');
    cleared('data:image/png;base64,oldimage');await flush();
    assert(uploaded.ids.get('cosmo-initial-preview').hidden,'cleared upload resurrected');
    await uploaded.upload('initial','data:image/png;base64,broken');
    assert(uploaded.ids.get('cosmo-initial-preview').hidden);
    assert.match(uploaded.ids.get('generation-queue-status').textContent,/not a decodable/);
    const plain=await uploaded.start();assert(uploaded.calls.some(c=>c.path.endsWith('/txt2img')));
    await uploaded.finish({images:[]});await plain.promise;
    const imgOnly=harness('img2img-only');await flush();await imgOnly.upload('initial');
    assert(imgOnly.ids.get('cosmo-mask-file').disabled);
    assert(imgOnly.ids.get('cosmo-mask-blur-label').hidden);
    const oldServer=harness('old-capabilities');await flush();
    assert(oldServer.ids.get('cosmo-image-inputs').hidden);
    assert(oldServer.ids.get('cosmo-initial-file').disabled);

    // Explicit-alpha guides and ordinary masks stay independent. Canvas and
    // decoding here are mocks; real PNG/editor coverage is in browser suite.
    const guidePng=Buffer.alloc(26);guidePng.write('PNG',1);guidePng[25]=6;
    const guideData='data:image/png;base64,'+guidePng.toString('base64');
    const advanced=harness();await flush();await advanced.upload('initial');await advanced.upload('mask');await advanced.upload('guide',guideData);
    assert(!advanced.ids.get('cosmo-guide-preview').hidden);
    await advanced.click('cosmo-save-generation-defaults');
    assert(!JSON.stringify(advanced.calls.find(c=>c.path.endsWith('/settings')).body).includes('inpaint_color'));
    advanced.ids.get('cosmo-image-strength').value='1';await advanced.start();
    assert(!advanced.calls.some(c=>c.path.endsWith('/img2img')));assert.match(advanced.ids.get('generation-queue-status').textContent,/strictly between/);
    advanced.ids.get('cosmo-image-strength').value='.75';
    advanced.ids.get('cosmo-mask-enabled').checked=false;
    const guideJob=await advanced.start();const guideCall=advanced.calls.find(c=>c.path.endsWith('/img2img'));
    assert.equal(guideCall.body.inpaint_color_guide,guidePng.toString('base64'));assert(!('mask' in guideCall.body));
    assert(advanced.ids.get('cosmo-guide-enabled').disabled);assert(advanced.ids.get('cosmo-guide-edit').disabled);const busyDraws=advanced.drawingCalls.length;advanced.ids.get('cosmo-guide-canvas').handlers.pointerdown({pointerId:1,button:0,clientX:1,clientY:1});assert.equal(advanced.drawingCalls.length,busyDraws);
    await advanced.finish({images:[]});await guideJob.promise;
    advanced.ids.get('cosmo-guide-enabled').checked=false;advanced.ids.get('cosmo-mask-enabled').checked=true;
    const ordinaryJob=await advanced.start();const ordinaryCall=advanced.calls.filter(c=>c.path.endsWith('/img2img')).at(-1);
    assert('mask' in ordinaryCall.body);assert(!('inpaint_color_guide' in ordinaryCall.body));
    assert(!advanced.ids.get('cosmo-guide-preview').hidden,'disable must preserve guide');
    await advanced.finish({images:[]});await ordinaryJob.promise;
    advanced.ids.get('cosmo-guide-enabled').checked=true;
    await advanced.click('cosmo-guide-edit');assert(advanced.ids.get('makeImage').disabled);assert(!advanced.ids.get('cosmo-guide-editor').hidden);
    await flush();
    const guideCanvas=advanced.ids.get('cosmo-guide-canvas');assert.equal(guideCanvas.width,8);assert.equal(guideCanvas.height,8);
    const initialBefore=advanced.ids.get('cosmo-initial-preview').src;
    guideCanvas.handlers.pointerdown({pointerId:1,clientX:2,clientY:2});guideCanvas.handlers.pointermove({clientX:5,clientY:5});guideCanvas.handlers.pointerup();
    assert(advanced.drawingCalls.some(x=>Array.isArray(x)&&x[0]==='stroke'&&x[1]===.5&&x[2]==='source-over'));
    advanced.ids.get('cosmo-guide-erase').checked=true;guideCanvas.handlers.pointerdown({pointerId:1,clientX:2,clientY:2});guideCanvas.handlers.pointercancel();
    assert(advanced.drawingCalls.some(x=>Array.isArray(x)&&x[2]==='destination-out'));
    await advanced.click('cosmo-guide-undo');assert(advanced.drawingCalls.includes('undo'));
    // Wand samples only the separate source snapshot, paints a discrete color
    // selection once, supports erase and preserves source/mask/guide inputs.
    const maskBefore=advanced.ids.get('cosmo-mask-preview').src, guideBefore=advanced.ids.get('cosmo-guide-preview').src;
    advanced.ids.get('cosmo-guide-tool').value='wand';advanced.ids.get('cosmo-guide-erase').checked=false;
    const wandBefore=advanced.drawingCalls.length;
    guideCanvas.handlers.pointerdown({pointerId:3,button:0,clientX:2,clientY:2});guideCanvas.handlers.pointermove({clientX:5,clientY:5});guideCanvas.handlers.pointerup();
    const wandDraw=advanced.drawingCalls.slice(wandBefore).find(x=>Array.isArray(x)&&x[0]==='drawImage'&&x[1]);
    assert(wandDraw);assert.deepEqual(wandDraw[1].slice(0,4),[51,102,255,128]);assert.equal(wandDraw[1].length,8*8*4);assert.equal(wandDraw[2],'source-over');
    assert.equal(advanced.drawingCalls.slice(wandBefore).filter(x=>Array.isArray(x)&&x[0]==='stroke').length,0,'wand drag must not brush');
    advanced.ids.get('cosmo-guide-erase').checked=true;guideCanvas.handlers.pointerdown({pointerId:3,button:0,clientX:2,clientY:2});
    assert(advanced.drawingCalls.some(x=>Array.isArray(x)&&x[0]==='drawImage'&&x[1]&&x[2]==='destination-out'));
    let n=advanced.drawingCalls.length;advanced.ids.get('cosmo-guide-buffer').value='1.5';guideCanvas.handlers.pointerdown({pointerId:3,button:0,clientX:2,clientY:2});assert.equal(advanced.drawingCalls.length,n);assert.match(advanced.ids.get('generation-queue-status').textContent,/whole number/);
    advanced.ids.get('cosmo-guide-buffer').value='0';guideCanvas.handlers.pointerdown({pointerId:3,button:2,clientX:2,clientY:2});assert.equal(advanced.drawingCalls.length,n);
    assert.equal(advanced.ids.get('cosmo-mask-preview').src,maskBefore);assert.equal(advanced.ids.get('cosmo-guide-preview').src,guideBefore);
    await advanced.click('cosmo-guide-undo');
    await advanced.click('cosmo-guide-cancel');assert.equal(advanced.ids.get('cosmo-initial-preview').src,initialBefore);assert(advanced.ids.get('cosmo-guide-editor').hidden);assert(!advanced.ids.get('cosmo-guide-preview').hidden);
    await advanced.click('cosmo-guide-edit');await flush();await advanced.click('cosmo-guide-save');await flush();assert(!advanced.ids.get('cosmo-guide-preview').hidden);assert.equal(advanced.ids.get('cosmo-initial-preview').src,initialBefore);
    // Cancel while source/guide decode is pending, then open again. Late
    // callbacks from the canceled editor must not paint over the new session.
    advanced.delayImages(true);const firstOpen=advanced.ids.get('cosmo-guide-edit').handlers.click();await flush();
    assert(advanced.ids.get('cosmo-guide-save').disabled);await advanced.click('cosmo-guide-cancel');
    const beforeLate=advanced.drawingCalls.length;advanced.resolveImages();await firstOpen;await flush();
    assert.equal(advanced.drawingCalls.length,beforeLate);assert(advanced.ids.get('cosmo-guide-editor').hidden);
    advanced.delayImages(false);await advanced.click('cosmo-guide-edit');await advanced.click('cosmo-guide-cancel');
    guidePng[25]=2;await advanced.upload('guide','data:image/png;base64,'+guidePng.toString('base64'));
    assert(advanced.ids.get('cosmo-guide-preview').hidden);assert.match(advanced.ids.get('generation-queue-status').textContent,/alpha channel/);
    advanced.setAlpha(0);await advanced.upload('guide',guideData);assert.match(advanced.ids.get('generation-queue-status').textContent,/no painted pixels/);
    advanced.setAlpha(128);advanced.setWidth(16);await advanced.upload('guide',guideData);assert.match(advanced.ids.get('generation-queue-status').textContent,/matching dimensions/);
    advanced.setWidth(8);await advanced.upload('guide',guideData);await advanced.click('cosmo-initial-clear');assert(advanced.ids.get('cosmo-guide-preview').hidden);
    const noAdvanced=harness('no-advanced');await flush();await noAdvanced.upload('initial');assert(noAdvanced.ids.get('cosmo-guide-file').disabled);assert(noAdvanced.ids.get('cosmo-guide-edit').hidden);await noAdvanced.click('cosmo-guide-edit');assert(noAdvanced.ids.get('cosmo-guide-editor').hidden);

    // Ordinary masks have their own grayscale storage/editor. All canvas/PNG
    // decoding is mocked; production exported pixel arrays are asserted here.
    const maskUi=harness();await flush();maskUi.setAlpha(255);await maskUi.upload('initial');await maskUi.upload('guide',guideData);await maskUi.upload('mask');
    const sourceUnchanged=maskUi.ids.get('cosmo-initial-preview').src,guideUnchanged=maskUi.ids.get('cosmo-guide-preview').src,maskUnchanged=maskUi.ids.get('cosmo-mask-preview').src;
    maskUi.ids.get('cosmo-mask-invert').checked=true;maskUi.ids.get('cosmo-mask-blur').value='2';
    await maskUi.click('cosmo-mask-edit');assert(!maskUi.ids.get('cosmo-mask-editor').hidden);assert(maskUi.ids.get('makeImage').disabled);assert(maskUi.ids.get('cosmo-guide-edit').disabled);
    await maskUi.click('cosmo-guide-edit');assert(maskUi.ids.get('cosmo-guide-editor').hidden,'ordinary editor must not open guide editor');
    const maskCanvas=maskUi.ids.get('cosmo-mask-canvas');maskUi.ids.get('cosmo-mask-size').value='1';
    maskCanvas.handlers.pointerdown({pointerId:7,button:0,clientX:.5,clientY:.5});assert(maskUi.ids.get('cosmo-mask-save').disabled);
    maskCanvas.handlers.pointermove({pointerId:7,clientX:1.5,clientY:.5});maskCanvas.handlers.pointercancel({pointerId:7});assert(!maskUi.ids.get('cosmo-mask-save').disabled);
    await maskUi.click('cosmo-mask-save');let exported=maskUi.canvasExports.at(-1);assert.equal(exported.width,8);assert.equal(exported.height,8);assert(exported.pixels.every((v,i)=>v===(i%4===3?255:0)),'canceled stroke/empty mask must export opaque black');
    assert.equal(maskUi.ids.get('cosmo-initial-preview').src,sourceUnchanged);assert.equal(maskUi.ids.get('cosmo-guide-preview').src,guideUnchanged);
    let job=await maskUi.start();let req=maskUi.calls.filter(c=>c.path.endsWith('/img2img')).at(-1);assert.equal(req.body.inpainting_mask_invert,1);assert.equal(req.body.mask_blur,2);assert(req.body.mask);assert.equal(req.body.inpaint_color_guide,guideData.split(',')[1]);
    assert(maskUi.ids.get('cosmo-mask-edit').disabled);await maskUi.click('cosmo-mask-edit');assert(maskUi.ids.get('cosmo-mask-editor').hidden);await maskUi.finish({images:[]});await job.promise;
    await maskUi.click('cosmo-mask-edit');maskUi.ids.get('cosmo-mask-tool').value='wand';maskCanvas.handlers.pointerdown({pointerId:7,button:0,clientX:.5,clientY:.5});await maskUi.click('cosmo-mask-save');exported=maskUi.canvasExports.at(-1);assert(exported.pixels.every(v=>v===255),'full source-color selection must export opaque white despite invert request flag');
    // Erase after selection, undo, clear+undo, then cancel: input layers stay put.
    const savedMask=maskUi.ids.get('cosmo-mask-preview').src;await maskUi.click('cosmo-mask-edit');maskCanvas.handlers.pointerdown({pointerId:7,button:0,clientX:.5,clientY:.5});maskUi.ids.get('cosmo-mask-erase').checked=true;maskCanvas.handlers.pointerdown({pointerId:7,button:0,clientX:.5,clientY:.5});await maskUi.click('cosmo-mask-undo');await maskUi.click('cosmo-mask-reset');await maskUi.click('cosmo-mask-undo');await maskUi.click('cosmo-mask-save');assert(maskUi.canvasExports.at(-1).pixels.every(v=>v===255));
    await maskUi.click('cosmo-mask-edit');await maskUi.click('cosmo-mask-reset');await maskUi.click('cosmo-mask-cancel');assert.equal(maskUi.ids.get('cosmo-mask-preview').src,savedMask);assert.equal(maskUi.ids.get('cosmo-guide-preview').src,guideUnchanged);
    // A whole pointer stroke applies gray strength ONCE over its coverage,
    // independent of repeated pointermove rate; canceled strokes do not evict
    // the oldest of five completed operations from undo history.
    await maskUi.click('cosmo-mask-edit');maskUi.ids.get('cosmo-mask-tool').value='brush';maskUi.ids.get('cosmo-mask-erase').checked=false;maskUi.ids.get('cosmo-mask-strength').value='.5';
    maskCanvas.handlers.pointerdown({pointerId:9,button:0,clientX:.5,clientY:.5});for(let i=0;i<12;i++)maskCanvas.handlers.pointermove({pointerId:9,clientX:.5,clientY:.5});maskCanvas.handlers.pointerup({pointerId:9});await maskUi.click('cosmo-mask-save');exported=maskUi.canvasExports.at(-1);assert.deepEqual(exported.pixels.slice(0,4),[128,128,128,255]);assert(exported.pixels.slice(4).every((v,i)=>v===(i%4===3?255:0)));
    await maskUi.click('cosmo-mask-edit');maskUi.ids.get('cosmo-mask-tool').value='wand';maskUi.ids.get('cosmo-mask-strength').value='.1';for(let i=0;i<5;i++)maskCanvas.handlers.pointerdown({pointerId:10,button:0,clientX:.5,clientY:.5});
    maskUi.ids.get('cosmo-mask-tool').value='brush';maskCanvas.handlers.pointerdown({pointerId:10,button:0,clientX:2.5,clientY:2.5});maskCanvas.handlers.pointercancel({pointerId:10});
    for(let i=0;i<5;i++)await maskUi.click('cosmo-mask-undo');await maskUi.click('cosmo-mask-save');assert(maskUi.canvasExports.at(-1).pixels.every((v,i)=>v===(i%4===3?255:0)));
    // Transparency and dimension problems reject editing, not existing uploads.
    maskUi.setAlpha(128);await maskUi.click('cosmo-mask-edit');assert(maskUi.ids.get('cosmo-mask-editor').hidden);assert.match(maskUi.ids.get('generation-queue-status').textContent,/opaque/);assert.equal(maskUi.ids.get('cosmo-mask-preview').src,savedMask);
    maskUi.setAlpha(255);maskUi.setWidth(16);await maskUi.click('cosmo-mask-edit');assert(maskUi.ids.get('cosmo-mask-editor').hidden);assert.match(maskUi.ids.get('generation-queue-status').textContent,/dimensions must match/);maskUi.setWidth(8);
    maskUi.delayImages(true);const delayedMask=maskUi.ids.get('cosmo-mask-edit').handlers.click();await flush();await maskUi.click('cosmo-mask-cancel');const noLateDraw=maskUi.drawingCalls.length;maskUi.resolveImages();await delayedMask;await flush();assert.equal(maskUi.drawingCalls.length,noLateDraw);assert(maskUi.ids.get('cosmo-mask-editor').hidden);maskUi.delayImages(false);
    await maskUi.click('cosmo-initial-clear');assert(maskUi.ids.get('cosmo-mask-preview').hidden);assert(maskUi.ids.get('cosmo-guide-preview').hidden);assert(maskUi.ids.get('cosmo-mask-edit').disabled);
    assert(imgOnly.ids.get('cosmo-mask-edit').hidden);await imgOnly.click('cosmo-mask-edit');assert(imgOnly.ids.get('cosmo-mask-editor').hidden);

    // Whole-mask actions operate on stored gray pixels, never source/guide or
    // the separate request-level invert flag, brush erase, or brush strength.
    const actions=harness();await flush();actions.setAlpha(255);await actions.upload('initial');await actions.upload('mask');await actions.upload('guide',guideData);
    const actionSource=actions.ids.get('cosmo-initial-preview').src,actionGuide=actions.ids.get('cosmo-guide-preview').src;
    actions.ids.get('cosmo-mask-enabled').checked=false;actions.ids.get('cosmo-mask-invert').checked=true;actions.ids.get('cosmo-mask-blur').value='2';actions.ids.get('cosmo-mask-strength').value='.25';actions.ids.get('cosmo-mask-erase').checked=true;
    await actions.click('cosmo-mask-edit');for(let i=0;i<8;i++)await actions.click('cosmo-mask-fill');await actions.click('cosmo-mask-undo');await actions.click('cosmo-mask-save');
    assert(actions.canvasExports.at(-1).pixels.every((v,i)=>v===(i%4===3?255:0)),'no-op fills must not evict original black history');
    assert.equal(actions.ids.get('cosmo-mask-enabled').checked,false);assert.equal(actions.ids.get('cosmo-mask-invert').checked,true);assert.equal(actions.ids.get('cosmo-guide-preview').src,actionGuide);
    await actions.click('cosmo-mask-edit');await actions.click('cosmo-mask-fill');await actions.click('cosmo-mask-invert-pixels');await actions.click('cosmo-mask-undo');await actions.click('cosmo-mask-reset');await actions.click('cosmo-mask-undo');await actions.click('cosmo-mask-save');
    assert(actions.canvasExports.at(-1).pixels.every(v=>v===255),'fill/undo must produce white independent of brush erase or .25 strength');
    await actions.click('cosmo-mask-edit');actions.ids.get('cosmo-mask-erase').checked=false;actions.ids.get('cosmo-mask-strength').value='.5';actions.ids.get('cosmo-mask-size').value='1';const actionCanvas=actions.ids.get('cosmo-mask-canvas');
    actionCanvas.handlers.pointerdown({pointerId:1,button:0,clientX:0,clientY:0});let draws=actions.drawingCalls.length;assert(actions.ids.get('cosmo-mask-invert-pixels').disabled);await actions.click('cosmo-mask-invert-pixels');await actions.click('cosmo-mask-fill');assert.equal(actions.drawingCalls.length,draws,'whole actions must not interrupt a stroke');actionCanvas.handlers.pointerup({pointerId:1});
    await actions.click('cosmo-mask-invert-pixels');await actions.click('cosmo-mask-save');let actionExport=actions.canvasExports.at(-1).pixels;assert.deepEqual(actionExport.slice(0,4),[127,127,127,255]);assert(actionExport.slice(4).every(v=>v===255));
    assert.equal(actions.ids.get('cosmo-mask-invert').checked,true);assert.equal(actions.ids.get('cosmo-initial-preview').src,actionSource);assert.equal(actions.ids.get('cosmo-guide-preview').src,actionGuide);
    const actionMask=actions.ids.get('cosmo-mask-preview').src,exportCount=actions.canvasExports.length;await actions.click('cosmo-mask-edit');await actions.click('cosmo-mask-invert-pixels');await actions.click('cosmo-mask-reset');await actions.click('cosmo-mask-cancel');assert.equal(actions.canvasExports.length,exportCount);assert.equal(actions.ids.get('cosmo-mask-preview').src,actionMask);
    actions.delayImages(true);const actionLoading=actions.ids.get('cosmo-mask-edit').handlers.click();await flush();draws=actions.drawingCalls.length;assert(actions.ids.get('cosmo-mask-fill').disabled);await actions.click('cosmo-mask-fill');await actions.click('cosmo-mask-invert-pixels');assert.equal(actions.drawingCalls.length,draws);await actions.click('cosmo-mask-cancel');actions.resolveImages();await actionLoading;await flush();actions.delayImages(false);
    actions.ids.get('cosmo-mask-enabled').checked=true;const actionJob=await actions.start();const actionRequest=actions.calls.filter(c=>c.path.endsWith('/img2img')).at(-1);assert.equal(actionRequest.body.inpainting_mask_invert,1);assert.equal(actionRequest.body.mask_blur,2);assert.equal(actionRequest.body.inpaint_color_guide,guideData.split(',')[1]);draws=actions.drawingCalls.length;await actions.click('cosmo-mask-invert-pixels');await actions.click('cosmo-mask-reset');assert.equal(actions.drawingCalls.length,draws);await actions.finish({images:[]});await actionJob.promise;

    const denied=harness('bad-capabilities');await flush();
    assert.equal(denied.ids.get('makeImage').disabled,true);
    assert.equal(denied.calls.length,1,'failed policy capability must not fall through to generation setup');
    assert.match(denied.ids.get('generation-queue-status').textContent,/disabled/);
    const unavailable=harness('unavailable-capabilities');await flush();
    assert.equal(unavailable.ids.get('makeImage').disabled,true);
    assert.equal(unavailable.calls.length,1,'HTTP capability failure must not initialize generation');
    assert.equal(normal.ids.get('kiosk-mode-save').disabled,true);
    assert(!normal.calls.some(c=>c.path==='/render'));
    console.log('UI_DOM_MOCK PASS: mask invert/fill/clear exact gray, independent request invert, no-op undo preservation, cancel/loading/stroke/busy guards; ordinary editor opaque export/empty black/white wand, stroke cancel/erase/undo/reset, input isolation, transparent/dimension rejection and stale cancel; shared wand fill/erase/undo, invalid buffer/non-left/busy gating, stale source decode cancel; Advanced guide/mask independence, alpha/dimension/strength checks, guide-only/ordinary-only, local brush/erase/undo/cancel/save handlers, unsupported capability; img2img/mask payload, image input clear/replace/decode-error, stale-read rejection, older-server capability gating, pixel-free defaults; native checkpoint/recipe defaults, atomic explicit save and save conflict, missing saved checkpoint, temporary generation overrides, capability failure, CPU default, distinct hardware/software adapters, stable-ID default, missing-device rejection, exact device request mapping, unsupported controls, progress, cancellation identity, PNG preview URI, lost connection until completion, busy conflict');
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>clearTimeout(watchdog));
