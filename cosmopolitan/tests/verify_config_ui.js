#!/usr/bin/env node
/* Executes the portable Settings script in a minimal DOM/fetch mock.
   No browser, HTTP server, or hardware is used. */
const fs = require('fs'), path = require('path'), vm = require('vm'), assert = require('assert/strict');
const source = fs.readFileSync(path.resolve(__dirname, '../resources/cpp-ui/scripts/backend-platform.js'), 'utf8');
const flush = async () => { for (let i = 0; i < 12; i++) await new Promise(resolve => setImmediate(resolve)); };
function harness(mode = 'normal', compute = {backend:'cpu', provider:'auto', device:'auto'}) {
    const ids = new Map(), calls = [];
    class Element {
        constructor(tag='div') {this.tag=tag;this.children=[];this.handlers={};this.disabled=false;this._value='';}
        set id(value) {this._id=value;ids.set(value,this);} get id(){return this._id;}
        set value(value) {this._value=String(value);for(const item of this.children)item.selected=item.value===this._value;}
        get value(){return this._value;}
        get options(){return this.children;}
        appendChild(item){this.children.push(item);if(this.tag==='select' && this.children.length===1)this.value=item.value;return item;}
        replaceChildren(){this.children=[];this.value='';}
        addEventListener(name,fn){this.handlers[name]=fn;}
        setAttribute(){}
    }
    for(const id of ['backend_platform','save-system-settings-btn','system-settings-extra','system-info','native-device-routing-status','native-device-routing-refresh','cpp-input-saving']) {const e=new Element(id==='backend_platform'?'select':'div');e.id=id;}
    const module = new Element('select');
    const document={getElementById:id=>ids.get(id),createElement:tag=>new Element(tag),querySelectorAll:()=>[module]};
    const saved={schema:1,compute,server:{port:8188,log_level:'info'},models:{checkpoint_dir:'models/checkpoints'},options:{}};
    let config={schema:1,config_path:'/settings.json',saved:structuredClone(saved),effective:structuredClone(saved),restart_required:false};
    const response=(body,status=200)=>({ok:status<400,status,json:async()=>structuredClone(body)});
    async function fetch(url, opts={}) {
        calls.push({url,body:opts.body?JSON.parse(opts.body):null});
        if(url==='/get/app_config')return mode==='config-fail'?response({message:'Bad saved config'},500):response(config);
        if(url.endsWith('backend-devices'))return mode==='device-fail'?response({message:'Driver unavailable'},503):response({devices:[
            {backend:'CPU',selector:'CPU',provider:'builtin',software:true,description:'CPU'},
            {backend:'WebGPU',selector:'WebGPU0',provider:'native',software:false,description:'Physical GPU',stable_id_available:true,stable_id:'uuid:physical'},
            {backend:'WebGPU',selector:'WebGPU1',provider:'embedded',software:true,description:'llvmpipe',stable_id_available:false,stable_id:''},
        ]});
        if(url==='/app_config') {
            if(mode==='busy')return response({message:'A native generation request is active'},409);
            const patch=JSON.parse(opts.body);config.saved={...config.saved,...patch};config.restart_required=true;
            return response(config);
        }
        throw Error('Unexpected endpoint '+url);
    }
    const ready=mode==='capability-fail'?Promise.reject(Error('Unknown native policy')):Promise.resolve({});
    ready.catch(()=>{});
    vm.runInNewContext(source,{document,window:{CppKiosk:{ready}},fetch,console});
    return {ids,calls,module,async change(id,value){const e=ids.get(id);e.value=value;await e.handlers.change?.();await flush();},async save(){await ids.get('save-system-settings-btn').handlers.click();await flush();}};
}
(async()=>{
    const cpu=harness();await flush();
    assert.equal(cpu.ids.get('backend_platform').value,'cpu');
    assert.equal(cpu.ids.get('save-system-settings-btn').disabled,false);
    assert.equal(cpu.module.disabled,true);
    assert.deepEqual(cpu.ids.get('backend_platform').options.map(x=>x.value),['cpu','webgpu']);
    assert.match(cpu.ids.get('system-info').textContent,/settings.json/);
    await cpu.change('backend_platform','webgpu');
    const choices=cpu.ids.get('cosmo-config-device').options;
    assert.equal(choices.length,3);
    assert.match(choices.find(x=>x.value==='WebGPU1').textContent,/software \/ CPU/);
    assert.match(choices.find(x=>x.value==='uuid:physical').textContent,/physical hardware/);
    await cpu.change('cosmo-config-device','uuid:physical');await cpu.save();
    const sent=cpu.calls.find(x=>x.url==='/app_config').body;
    assert.deepEqual(sent.compute,{backend:'webgpu',provider:'auto',device:'uuid:physical'});
    assert.match(cpu.ids.get('cosmo-config-status').textContent,/restart/i);
    assert.match(cpu.ids.get('system-info').textContent,/Running: cpu/);
    await cpu.change('cosmo-config-provider','embedded');
    assert.deepEqual(cpu.ids.get('cosmo-config-device').options.map(x=>x.value),['auto'],'new policy has no invented adapter list before restart');
    const missing=harness('normal',{backend:'webgpu',provider:'auto',device:'uuid:missing'});await flush();
    assert.equal(missing.ids.get('cosmo-config-device').value,'uuid:missing');
    assert.equal(missing.ids.get('save-system-settings-btn').disabled,true);
    await missing.change('cosmo-config-device','auto');assert.equal(missing.ids.get('save-system-settings-btn').disabled,false);
    const failing=harness('config-fail');await flush();assert.equal(failing.ids.get('save-system-settings-btn').disabled,true);assert.match(failing.ids.get('cosmo-config-status').textContent,/Bad saved config/);
    const caps=harness('capability-fail');await flush();assert.equal(caps.ids.get('save-system-settings-btn').disabled,true);assert.equal(caps.calls.length,0);
    const unavailable=harness('device-fail');await flush();assert.equal(unavailable.ids.get('save-system-settings-btn').disabled,false);assert.match(unavailable.ids.get('cosmo-config-device-inventory').textContent,/Driver unavailable/);
    const busy=harness('busy');await flush();await busy.save();assert.match(busy.ids.get('cosmo-config-status').textContent,/not saved.*active/);assert.equal(busy.ids.get('save-system-settings-btn').disabled,false);
    console.log(JSON.stringify({status:'PASS',scope:'Settings DOM and request mock; no browser/server/hardware',cases:8}));
})().catch(error=>{console.error(error);process.exitCode=1;});
