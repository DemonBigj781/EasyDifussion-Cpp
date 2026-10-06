const assert = require('node:assert/strict');
const test = require('node:test');
const filename = '../source/UI.cpp/Pages/src/Plugin/plugin_scripts/creative_plugins.js';
test('Rabbit Hole builds bounded, independent parameter combinations', () => {
    const {rabbitRequests} = require(filename);
    const base = {prompt:'cat',seed:12,num_inference_steps:20,guidance_scale:7,num_outputs:4,lora_alpha:[.5,.8]};
    const jobs = rabbitRequests(base,{max:6,seeds:2,stepsCount:3,stepsStep:5,stepsMid:20,cfgCount:1});
    assert.equal(jobs.length,6);
    assert.deepEqual([...new Set(jobs.map(job=>job.num_inference_steps))],[15,20,25]);
    assert.ok(jobs.every(job=>job.num_outputs===1));
    assert.equal(base.num_outputs,4);
    assert.throws(()=>rabbitRequests(base,{max:0}),/maximum/i);
    assert.throws(()=>rabbitRequests(base,{max:513}),/maximum/i);
    const models=rabbitRequests(base,{max:8,models:['a','b'],vaes:['x','y'],loras:['l'],loraCount:2,loraStep:.2});
    assert.equal(models.length,8);
    assert.deepEqual([...new Set(models.map(job=>job.use_stable_diffusion_model))],['a','b']);
});
test('Animation validates time ranges and limits before rendering', () => {
    const {frameTimes} = require(filename);
    assert.deepEqual(frameTimes(0,1,4),[0,.25,.5,.75]);
    assert.throws(()=>frameTimes(1,0,5),/range/i);
    assert.throws(()=>frameTimes(0,100,30),/limit/i);
    assert.throws(()=>frameTimes(0,1,0),/FPS/i);
});
