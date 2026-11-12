#!/usr/bin/env node
// Real shared pixel algorithm; no browser, image decoder, model or SAM.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const canonical=fs.readFileSync(path.join(root,'ui/media/js/editor-mask-tools.js'),'utf8');
const script=fs.readFileSync(path.join(root,'cosmopolitan/resources/cpp-ui/scripts/generate.js'),'utf8');
const start='// BEGIN VERBATIM SHARED EDITOR MASK TOOLS (source equality tested)\n';
const end='// END VERBATIM SHARED EDITOR MASK TOOLS\n';
assert(script.startsWith(start));const body=script.slice(start.length,script.indexOf(end));assert.equal(body,canonical,'Bundled shared selection source drifted');
const context={window:{}};vm.runInNewContext(body,context);const portable=context.window.EditorMaskTools;
const reference=require(path.join(root,'ui/media/js/editor-mask-tools.js'));
const rgba=values=>Uint8ClampedArray.from(values.flatMap(x=>Array.isArray(x)?x:[x,x,x,255]));
let checks=0;
function selected(values,w,h,x,y,t,b,expected){const pixels=rgba(values),before=Array.from(pixels);const got=Array.from(portable.select(pixels,w,h,x,y,t,b));assert.deepEqual(got,expected);assert.deepEqual(got,Array.from(reference.select(pixels,w,h,x,y,t,b)));assert.deepEqual(Array.from(pixels),before,'source mutated');checks++;}
selected([0,255,0],3,1,0,0,0,0,[1,0,0]); // disconnected equal colors
selected([0,255,255,0],2,2,0,0,0,0,[1,0,0,0]); // four, not eight, connected
selected([[0,0,0,255],[2,0,0,255],[4,0,0,255]],3,1,0,0,3,0,[1,1,0]); // seed-relative, not chained
selected([[0,0,0,255],[3,4,0,255]],2,1,0,0,5,0,[1,1]);
selected([[0,0,0,255],[3,4,0,255]],2,1,0,0,4.99,0,[1,0]);
selected([255,0,0,255],2,2,1,0,0,0,[0,1,0,0]); // no row wrapping
selected([255,255,255,255,0,255,255,255,255],3,3,1,1,0,1,[0,1,0,1,1,1,0,1,0]); // Manhattan buffer
selected([0,255,255,255],4,1,0,0,0,2,[1,1,1,0]);
selected([[1,2,3,0],[1,2,3,255]],2,1,0,0,0,0,[1,1]); // RGB ignores source alpha
selected([0,255],2,1,-10,10,0,0,[1,0]);selected([0,255],2,1,10,-10,0,0,[0,1]);
for(const args of [[rgba([0]),0,1,0,0,0,0],[rgba([0]),1.5,1,0,0,0,0],[rgba([0]),2,1,0,0,0,0],[rgba([0]),1,1,NaN,0,0,0],[rgba([0]),1,1,0,Infinity,0,0],[rgba([0]),1,1,0,0,NaN,0],[rgba([0]),1,1,0,0,0,Infinity]]){assert.throws(()=>portable.select(...args));checks++;}
// Deterministic corpus exercises arbitrary seed coordinates, color ranges,
// disconnected regions and expansion while comparing exact existing behavior.
let seed=42;const next=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
for(let k=0;k<200;k++){const w=1+next()%15,h=1+next()%12,pixels=Uint8ClampedArray.from({length:w*h*4},()=>next()%256),before=Array.from(pixels);const args=[pixels,w,h,next()%w,next()%h,next()%256,next()%5];assert.deepEqual(Array.from(portable.select(...args)),Array.from(reference.select(...args)));assert.deepEqual(Array.from(pixels),before);checks++;}
console.log(`PORTABLE_WAND_PIXEL PASS: verbatim shared-source equality; ${checks} exact pixel/differential cases, connected seed-relative RGB threshold, bounds, Manhattan buffer and source immutability. No browser or SAM inference.`);
