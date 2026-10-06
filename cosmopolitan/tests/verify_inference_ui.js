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
const kiosk = fs.readFileSync(path.join(scripts, 'kiosk.js'), 'utf8');
const caps = {protocol:1,mode:'native-single-user',kiosk_supported:false,kiosk_enabled:false,txt2img:true,max_concurrent_generations:1};
const flush = async () => { for (let i=0;i<12;i++) await new Promise(r=>setImmediate(r)); };
function harness(mode='success') {
    const ids = new Map(), timers = new Map(), calls = [];
    let timerId=0, resolveGeneration, progressState={current_step:0,total_steps:4,completed:false};
    class Element {
        constructor(tag='div') { this.tag=tag;this.children=[];this.handlers={};this.dataset={};this._value='';this.disabled=false;this.attributes={}; }
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
    }
    const names=['width-value','height-value','makeImage','stopImage','generation-queue-status','generation-progress','preview-content','generation-model-status','show-download-popup','output_format','num_images','vae_model','text_encoder_model','generation-companion-status','cpp-modifier-status','cpp-modifier-grid','lora_model','generate-plugin-panels','cpp-image-modifiers','stable_diffusion_model','sampler','width','height','prompt','negative_prompt','seed','steps','guidance_scale','clear-all-previews','initial-text','kiosk-status-banner','kiosk-mode-enabled','kiosk-mode-save','kiosk-mode-save-status'];
    for(const id of names){const e=new Element();e.id=id;e.name=id;ids.set(id,e);}
    const scope = new Element();scope.querySelectorAll=()=>[...ids.values()];
    const values={width:128,height:128,steps:4,seed:42,guidance_scale:7,prompt:'a red cube',negative_prompt:''};
    for(const [id,value] of Object.entries(values))ids.get(id).value=value;
    const document={getElementById:id=>ids.get(id),querySelector:()=>scope,createElement:tag=>new Element(tag),documentElement:new Element()};
    const response = (body,status=200) => ({ok:status<400,status,text:async()=>JSON.stringify(body),json:async()=>body});
    async function fetch(path,options={}) {
        const body=options.body?JSON.parse(options.body):undefined;calls.push({path,body});
        if(path.endsWith('cosmopolitan-capabilities')) {
            if(mode==='unavailable-capabilities')return response({message:'Unavailable'},503);
            return response(mode==='bad-capabilities'?{...caps,kiosk_enabled:true}:caps);
        }
        if(path.endsWith('backend-devices'))return response({devices:[{backend:'WebGPU'},{backend:'CPU'}]});
        if(path.endsWith('checkpoints'))return response({models:[{name:'model.safetensors'}]});
        if(path.endsWith('txt2img')) {
            if(mode==='conflict')return response({message:'Another generation is active'},409);
            if(mode==='network')throw new TypeError('network disconnected');
            return new Promise(resolve=>{resolveGeneration=value=>resolve(response(value));});
        }
        if(path.endsWith('/progress'))return response(progressState);
        if(path.endsWith('/interrupt'))return response({});
        throw Error('Unexpected endpoint: '+path);
    }
    const window={addEventListener(){}};
    const context=vm.createContext({window,document,fetch,crypto:webcrypto,Uint32Array,console,
        Option:class extends Element{constructor(text,value){super('option');this.textContent=text;this.value=value;}},
        localStorage:{getItem:()=>null,setItem(){}},
        setTimeout(fn){const id=++timerId;timers.set(id,fn);return id;},clearTimeout(id){timers.delete(id);}
    });
    vm.runInContext(kiosk,context);
    vm.runInContext(generate,context);
    return {ids,calls,window,async click(id){await ids.get(id).handlers.click?.();await flush();},
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
    const call=normal.calls.find(c=>c.path.endsWith('txt2img'));
    assert.equal(call.body.backend,'cpu');
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

    const png=harness();await flush();png.ids.get('cosmo-generation-backend').value='webgpu';const imageJob=await png.start();
    assert.equal(png.calls.find(c=>c.path.endsWith('txt2img')).body.backend,'webgpu');
    await png.finish({images:['iVBORw0KGgoAAAANSUhEUg==']});await imageJob.promise;
    const card=png.ids.get('preview-content').children[0];
    assert.equal(card.children[0].src,'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==');
    assert.equal(card.children[2].download,'image-42.png');

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

    const denied=harness('bad-capabilities');await flush();
    assert.equal(denied.ids.get('makeImage').disabled,true);
    assert.equal(denied.calls.length,1,'failed policy capability must not fall through to generation setup');
    assert.match(denied.ids.get('generation-queue-status').textContent,/disabled/);
    const unavailable=harness('unavailable-capabilities');await flush();
    assert.equal(unavailable.ids.get('makeImage').disabled,true);
    assert.equal(unavailable.calls.length,1,'HTTP capability failure must not initialize generation');
    assert.equal(normal.ids.get('kiosk-mode-save').disabled,true);
    assert(!normal.calls.some(c=>c.path==='/render'));
    console.log('UI_DOM_MOCK PASS: capability failure, CPU default, WebGPU selection, native request mapping, unsupported controls, progress, cancellation identity, PNG preview URI, lost connection until completion, busy conflict');
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>clearTimeout(watchdog));
