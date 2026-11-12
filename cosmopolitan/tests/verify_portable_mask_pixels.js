#!/usr/bin/env node
// Actual production mask arithmetic and rasterization, not canvas mocks.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.resolve(__dirname,'../resources/cpp-ui/scripts/generate.js'),'utf8');
const begin='// BEGIN PORTABLE ORDINARY MASK PIXELS\n',end='// END PORTABLE ORDINARY MASK PIXELS\n';
const context={window:{}};vm.runInNewContext(source.slice(source.indexOf(begin)+begin.length,source.indexOf(end)),context);const tool=context.window.PortableMaskPixels;let checks=0;
const equal=(a,b)=>{assert.deepEqual(Array.from(a),b);checks++;};
const rgba=Uint8ClampedArray.from([255,0,0,255,0,255,0,255,0,0,255,255,128,128,128,255,0,0,0,255,255,255,255,255]);const original=Array.from(rgba);
equal(tool.fromRgba(rgba,6,1),[76,149,28,128,0,255]);equal(rgba,original);
equal(tool.toRgba(Uint8ClampedArray.from([0,128,255]),3,1),[0,0,0,255,128,128,128,255,255,255,255,255]);
let mask=new Uint8ClampedArray(9);tool.stroke(mask,3,3,[1.5,1.5],[1.5,1.5],1,255,1);equal(mask,[0,0,0,0,255,0,0,0,0]);
tool.stroke(mask,3,3,[1.5,1.5],[1.5,1.5],1,0,1);equal(mask,Array(9).fill(0));
tool.stroke(mask,3,3,[.5,.5],[2.5,2.5],1,255,1);equal(mask,[255,0,0,0,255,0,0,0,255]);
const before=Array.from(mask);tool.stroke(mask,3,3,[-100,-100],[-99,-99],1,255,1);equal(mask,before);
mask=new Uint8ClampedArray(3);const selected=Uint8ClampedArray.from([1,0,1]);tool.select(mask,selected,3,1,255,.5);equal(mask,[128,0,128]);equal(selected,[1,0,1]);
tool.select(mask,selected,3,1,255,.5);equal(mask,[192,0,192]);tool.select(mask,selected,3,1,0,1);equal(mask,[0,0,0]);
// Pixel-center raster rule guarantees size-one integer-coordinate dots,
// erases, corners and short horizontal strokes without claiming canvas AA.
for(const [point,index] of [[[0,0],0],[[1,1],4],[[2.999,2.999],8]]){const dot=new Uint8ClampedArray(9),beforePoint=[...point];tool.stroke(dot,3,3,point,point,1,255,1);equal(dot,Array.from({length:9},(_,i)=>i===index?255:0));tool.stroke(dot,3,3,point,point,1,0,1);equal(dot,Array(9).fill(0));equal(point,beforePoint);}
const horizontal=new Uint8ClampedArray(9);tool.stroke(horizontal,3,3,[0,1],[2,1],1,255,1);equal(horizontal,[0,0,0,255,255,255,0,0,0]);
// A cleared/all-black mask is valid and exports opaque BLACK, not transparent white.
equal(tool.toRgba(mask,3,1),[0,0,0,255,0,0,0,255,0,0,0,255]);
for(const alpha of [0,1,128,254]){const input=Uint8ClampedArray.from([255,255,255,alpha]),old=Array.from(input);assert.throws(()=>tool.fromRgba(input,1,1),/opaque/);equal(input,old);}
for(const run of [()=>tool.fromRgba(rgba,3,1),()=>tool.toRgba(mask,0,1),()=>tool.toRgba(mask,3.5,1),()=>tool.toRgba(mask,4096,4096),()=>tool.select(mask,selected,3,1,128,1),()=>tool.select(mask,selected,3,1,255,NaN),()=>tool.select(mask,selected,3,1,255,0),()=>tool.select(mask,selected,3,1,255,2),()=>tool.stroke(mask,3,1,[NaN,0],[0,0],1,255,1),()=>tool.stroke(mask,3,1,[],[0,0],1,255,1),()=>tool.stroke(mask,3,1,[0,0],[0,0],0,255,1)]){const old=Array.from(mask);assert.throws(run);equal(mask,old);}
// Exhaustive gray and alpha-opaque round trips keep native grayscale values.
const values=Uint8ClampedArray.from({length:256},(_,i)=>i);equal(tool.fromRgba(tool.toRgba(values,256,1),256,1),Array.from(values));
// Whole-image actions are exact grayscale operations, not brush strength or
// request inversion. No-op fill reports false so it need not consume undo.
const whole=Uint8ClampedArray.from([0,1,127,128,254,255]);assert.equal(tool.invert(whole,6,1),true);equal(whole,[255,254,128,127,1,0]);tool.invert(whole,6,1);equal(whole,[0,1,127,128,254,255]);
assert.equal(tool.fill(whole,6,1,255),true);equal(whole,Array(6).fill(255));assert.equal(tool.fill(whole,6,1,255),false);
assert.equal(tool.fill(whole,6,1,0),true);equal(whole,Array(6).fill(0));assert.equal(tool.fill(whole,6,1,0),false);equal(tool.toRgba(whole,6,1),Array.from({length:24},(_,i)=>i%4===3?255:0));
for(const run of [()=>tool.fill(whole,6,1,128),()=>tool.fill(whole,6,1,NaN),()=>tool.fill(whole,3,1,255),()=>tool.invert(whole,1.5,4)]){const before=Array.from(whole);assert.throws(run);equal(whole,before);}
console.log(`PORTABLE_MASK_PIXELS PASS: ${checks} exact checks; native RGB luminance, opaque export including empty black, alpha rejection, raster stroke/erase/clipping, selection blending, bounds and immutable inputs. No browser/model inference.`);
