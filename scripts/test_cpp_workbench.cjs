const assert=require('node:assert/strict');
const test=require('node:test');
const root='../source/UI.cpp/Pages/src/Plugin/plugin_scripts/';
test('wildcard parsing preserves duplicates and last line, normalizes CRLF',()=>{
    const {parseLines,composePrompt}=require(root+'prompt_workbench.js');
    assert.deepEqual(parseLines(' first  line\r\nsecond\nfirst  line\n'),['first line','second','first line']);
    assert.equal(composePrompt('prefix','main','suffix'),'prefix, main, suffix');
    assert.equal(composePrompt('','main',''),'main');
});
test('LoRA shuttle preserves multi-LoRA shapes and clamps like legacy',()=>{
    const {shuttle}=require(root+'image_workbench.js');
    const req={use_lora_model:['a','b'],lora_alpha:[.05,.95],seed:12};
    assert.deepEqual(shuttle(req,'shift',.1).lora_alpha,[.15,1]);
    assert.deepEqual(shuttle(req,'set',.5).lora_alpha,[.5,.5]);
    assert.deepEqual(req.lora_alpha,[.05,.95]);
    assert.equal(shuttle({use_lora_model:'a',lora_alpha:.5},'shift',-.1).lora_alpha,.4);
});
test('img2img schedules update and reset independently without mutating base',()=>{
    const {schedule}=require(root+'image_workbench.js');
    assert.equal(schedule(7,0,2,.2,0,0,30),7);
    assert.equal(schedule(7,2,2,.2,0,0,30),7.2);
    assert.equal(schedule(7,4,2,.2,4,0,30),7);
    assert.equal(schedule(.9,3,1,.1,0,0,.99),.99);
    assert.throws(()=>schedule(1,1,-1,.1,0,0,10),/interval/i);
});
test('template import accepts legacy records, rejects malformed data and preserves extras',()=>{
    const {validateTemplates}=require(root+'template_manager.js');
    const input=[{name:'test',task:{numOutputsTotal:2,reqBody:{prompt:'cat',seed:42,use_lora_model:['a','b'],lora_alpha:[.2,.3]}},extra:'keep'}];
    assert.deepEqual(validateTemplates(input),input);
    assert.throws(()=>validateTemplates([{name:'bad',task:{reqBody:null}}]),/template/i);
    assert.throws(()=>validateTemplates([{name:'bad',task:{reqBody:{prompt:9}}}]),/prompt/i);
    assert.throws(()=>validateTemplates({}),/array/i);
});
