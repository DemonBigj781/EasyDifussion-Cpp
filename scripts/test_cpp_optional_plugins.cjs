const assert = require('node:assert/strict');
const test = require('node:test');
const path = require('node:path');
const root = path.join(__dirname, '../source/UI.cpp/Pages/src/Plugin/plugin_scripts');

test('native plugin math: legacy seed parity, bounded prompt products, safe diff', () => {
    const m = require(path.join(root, 'optional_plugins.js'));
    assert.equal(m.crc32('123456789'), 0xcbf43926);
    assert.deepEqual(m.expand('a {red,blue} {cat,dog}'), ['a red cat','a red dog','a blue cat','a blue dog']);
    assert.throws(() => m.expand('{a,b}{c,d}', 3), /limit/i);
    assert.equal(m.seedFor({prompt:'café', negative_prompt:'bad', seed:42}, 'normal'),42);
    assert.equal(m.seedFor({prompt:'café', negative_prompt:'bad', seed:42}, 'high'),m.crc32('café_42'));
    assert.equal(m.seedFor({prompt:'café', negative_prompt:'bad', seed:42}, 'very_high'),m.crc32('café_bad_42'));
    assert.deepEqual(m.diff('red cat', 'blue cat'), [{type:'delete', text:'red'},{type:'insert',text:'blue'},{type:'equal',text:' cat'}]);
});

test('queue preserves FIFO, supports newest first, counts images and survives errors', async () => {
    const {createQueue} = require(path.join(root, 'generation_queue.js'));
    const order = [], releases = [];
    const q = createQueue({run: req => new Promise(resolve => {order.push(req.seed); releases.push(resolve);})});
    const a=q.enqueue({seed:1,num_outputs:2}), b=q.enqueue({seed:2,num_outputs:3}), c=q.enqueue({seed:3,num_outputs:1});
    assert.deepEqual(q.state,{active:true,pending:2,images:6});
    q.newestFirst=true;
    releases.shift()({}); await a; await new Promise(resolve=>setImmediate(resolve));
    assert.deepEqual(order,[1,3]);
    releases.shift()({}); await c; await new Promise(resolve=>setImmediate(resolve));
    releases.shift()({}); await b;
    assert.deepEqual(order,[1,3,2]);
    assert.equal(q.state.images,0);
    const failures=createQueue({run:async req=>{if(req.seed===1) throw Error('bad'); return req;}});
    await assert.rejects(failures.enqueue({seed:1}), /bad/);
    assert.equal((await failures.enqueue({seed:2})).seed,2);
});

test('cancel rejects queued requests and active result without starting another task', async () => {
    const {createQueue} = require(path.join(root, 'generation_queue.js'));
    let release;
    const q=createQueue({run:()=>new Promise(resolve=>{release=resolve;})});
    const a=q.enqueue({seed:1}), b=q.enqueue({seed:2});
    const checks=[assert.rejects(a,/cancel/i),assert.rejects(b,/cancel/i)];
    q.cancel(); release({}); await Promise.all(checks);
    assert.deepEqual(q.state,{active:false,pending:0,images:0});
});

test('FIFO snapshots inputs and rejects overflow without dropping existing work', async () => {
    const {createQueue} = require(path.join(root, 'generation_queue.js'));
    const order=[],releases=[];
    const q=createQueue({limit:1,run:req=>new Promise(resolve=>{order.push(req.seed);releases.push(resolve);})});
    const request={seed:10};
    const a=q.enqueue(request),b=q.enqueue({seed:20});request.seed=99;
    await assert.rejects(q.enqueue({seed:30}),/Queue limit/);
    releases.shift()({});await a;await new Promise(resolve=>setImmediate(resolve));
    releases.shift()({});await b;
    assert.deepEqual(order,[10,20]);
});
