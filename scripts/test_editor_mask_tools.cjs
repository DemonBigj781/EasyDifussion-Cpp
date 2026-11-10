const assert = require('node:assert/strict');
const {test} = require('node:test');
const tools = require('../ui/media/js/editor-mask-tools.js');
const pixels = values=>Uint8ClampedArray.from(values.flatMap(value=>[value,value,value,255]));
test('zero sensitivity matches exact color, higher sensitivity expands the region',()=>{
    assert.deepEqual([...tools.select(pixels([100,103,200]),3,1,0,0,0,0)],[1,0,0]);
    assert.deepEqual([...tools.select(pixels([100,103,200]),3,1,0,0,6,0)],[1,1,0]);
});
test('buffer grows by whole pixels without wrapping across image rows',()=>{
    assert.deepEqual([...tools.select(pixels([255,0,255,255]),2,2,1,0,0,1)],[1,1,0,1]);
    assert.deepEqual([...tools.select(pixels([0,255,255]),3,1,0,0,0,2)],[1,1,1]);
});
