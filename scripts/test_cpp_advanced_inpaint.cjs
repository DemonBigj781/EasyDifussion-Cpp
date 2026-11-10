// Live C++ pages and real canvas controls; generation requests are intercepted.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.CANVAS_TEST_URL || 'http://127.0.0.1:10000';

async function run() {
    const browser = await chromium.launch({headless:true});
    try {
        const page = await browser.newPage({viewport:{width:1280,height:950}}), errors=[], requests=[];
        page.on('pageerror',error=>errors.push(error.message));
        await page.route('**/*',async route=>{
            const request=route.request(), url=new URL(request.url());
            if (process.env.CANVAS_TEST_RENDERER && request.resourceType()==='document' && url.pathname.startsWith('/cpp-ui')) {
                return route.fulfill({contentType:'text/html',body:execFileSync(process.env.CANVAS_TEST_RENDERER,[url.pathname==='/cpp-ui/canvas'?'/canvas':'/'],{encoding:'utf8'})});
            }
            if(request.method()==='POST') {
                assert.equal(url.pathname,'/render','No other mutation is allowed during this test');
                requests.push(request.postDataJSON());
                return route.fulfill({status:409,json:{detail:'Test intercepted: no generation started'}});
            }
            const models={
                '/get/model':{models:[{model:'test/sd15',name:'Test SD 1.5'}]},
                '/get/vae':{models:[]}, '/get/other':{models:[]},
                '/get/lora':{models:[]}, '/get/models':{models:[]},
            };
            if(models[url.pathname])return route.fulfill({json:models[url.pathname]});
            return route.continue();
        });
        await page.goto(base+'/cpp-ui/canvas',{waitUntil:'domcontentloaded'});
        assert.equal(await page.locator('#canvas-advanced-inpaint').count(),1,
            'Advanced Inpaint must be present in the running C++ Canvas, not only legacy HTML');
        await page.waitForFunction(()=>window.CppCanvas?.ready);
        await page.evaluate(()=>CppCanvas.ready);
        assert.equal(await page.locator('#canvas-brush-size').inputValue(),'1');
        assert.equal(await page.locator('#canvas-brush-size').getAttribute('min'),'1');
        assert.equal(await page.locator('#canvas-wand-sensitivity').getAttribute('type'),'range');
        assert.equal(await page.locator('#canvas-wand-buffer').getAttribute('type'),'range');
        const source=await page.evaluate(()=>{
            const canvas=document.createElement('canvas');canvas.width=64;canvas.height=64;
            const ctx=canvas.getContext('2d');ctx.fillStyle='#ffffff';ctx.fillRect(0,0,64,64);
            return canvas.toDataURL();
        });
        await page.setInputFiles('#canvas-source',{name:'source.png',mimeType:'image/png',buffer:Buffer.from(source.split(',')[1],'base64')});
        await page.waitForFunction(()=>document.getElementById('canvas-status').textContent.includes('64 × 64'));
        await page.click('#canvas-advanced-inpaint');
        await page.getByRole('button',{name:'Use #ff0000',exact:true}).click();
        async function opacity(value) {
            await page.locator('#canvas-opacity').evaluate((el,value)=>{el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));
        }
        async function dot(x,y) {
            await page.locator('#image-editor-canvas').scrollIntoViewIfNeeded();
            const box=await page.locator('#image-editor-canvas').boundingBox();
            await page.mouse.move(box.x+x/64*box.width,box.y+y/64*box.height);
            await page.mouse.down();await page.mouse.up();
        }
        await page.locator('#canvas-brush-size').evaluate(el=>{el.value='8';el.dispatchEvent(new Event('input',{bubbles:true}));});
        await opacity(25);await dot(16,32);
        await opacity(75);await dot(48,32);
        await page.click('#canvas-undo');
        await page.click('#canvas-redo');
        await page.getByRole('button',{name:'Use #000000',exact:true}).click();
        await opacity(50);await dot(32,32);
        await page.getByRole('button',{name:'Use #ff0000',exact:true}).click();
        // Other modes have independent layers and cannot erase advanced guidance.
        await page.click('#canvas-draw');await opacity(100);await dot(32,16);
        await page.click('#canvas-inpaint');await dot(32,48);
        await page.click('#canvas-advanced-inpaint');
        await page.screenshot({path:'/tmp/cpp-advanced-inpaint-desktop.png'});
        await page.click('#canvas-generate');
        await page.waitForURL('**/cpp-ui');
        await page.evaluate(()=>CppCanvas.ready);
        await page.waitForSelector('#stable_diffusion_model-model-list',{state:'attached'});
        await page.waitForSelector('#text_encoder_model-model-list',{state:'attached'});
        await page.fill('#stable_diffusion_model','sd15');
        await page.locator('#stable_diffusion_model-model-list [data-path="test/sd15"]').click();
        await page.fill('#prompt','test red color guidance');
        assert.equal(await page.locator('#cpp-canvas-strength').inputValue(),'0.65','Range limits and step must be set before its initial value');
        const payload=await page.evaluate(()=>CppGeneration.buildRequest());
        assert.ok(payload.init_image.startsWith('data:image/png;'));
        assert.ok(payload.init_image_mask.startsWith('data:image/png;'));
        assert.ok(payload.inpaint_color_guide.startsWith('data:image/png;'),'Advanced Inpaint must submit separate RGBA color guidance');
        async function pixels(url) {
            return page.evaluate(async url=>{
                const image=new Image();image.src=url;await image.decode();const c=document.createElement('canvas');c.width=image.width;c.height=image.height;
                const ctx=c.getContext('2d');ctx.drawImage(image,0,0);
                return {size:[c.width,c.height],values:[[16,32],[48,32],[32,16],[32,48],[0,0],[32,32]].map(([x,y])=>[...ctx.getImageData(x,y,1,1).data])};
            },url);
        }
        const mask=await pixels(payload.init_image_mask), original=await pixels(payload.init_image), guide=await pixels(payload.inpaint_color_guide);
        assert.deepEqual(mask.size,[64,64]);assert.deepEqual(guide.size,[64,64]);
        assert.deepEqual(mask.values[0],[0,0,0,255],'Ordinary mask remains independent of color guidance');
        assert.deepEqual(mask.values[3],[255,255,255,255],'Ordinary inpaint influence is retained');
        assert.ok(Math.abs(guide.values[0][3]-64)<=1);assert.ok(Math.abs(guide.values[1][3]-191)<=1);
        assert.ok(Math.abs(guide.values[5][3]-128)<=1,'Black guidance retains its opacity weight');
        assert.deepEqual(original.values[0],[255,255,255,255]);assert.deepEqual(original.values[1],[255,255,255,255]);
        assert.deepEqual(original.values[2],[255,0,0,255],'Only the separate Draw layer may change the base');
        const fields=request=>({init_image:request.init_image,init_image_mask:request.init_image_mask,inpaint_color_guide:request.inpaint_color_guide});
        for(const mode of ['draw','inpaint','advanced']) {
            await page.click(`#cpp-input-preview-tab-${mode}`);
            assert.equal(await page.locator(`#cpp-input-preview-panel-${mode}`).isVisible(),true);
            assert.deepEqual(fields(await page.evaluate(()=>{const req={};CppCanvas.applyToRequest(req);return req;})),fields(payload),'Preview tabs must not change generation inputs');
        }
        await page.locator('#cpp-input-preview-tab-advanced').scrollIntoViewIfNeeded();
        await page.screenshot({path:'/tmp/cpp-color-inpaint-previews.png'});
        await page.click('#makeImage');
        await page.waitForFunction(()=>document.getElementById('generation-queue-status').textContent.includes('Test intercepted'));
        assert.equal(requests.length,1);assert.equal(requests[0].init_image_mask,payload.init_image_mask);
        assert.equal(requests[0].inpaint_color_guide,payload.inpaint_color_guide);
        // Reopen without baking the old paint into its own base; cancel must not alter applied input.
        await page.click('#cpp-canvas-open-advanced');
        await page.waitForURL('**/cpp-ui/canvas?mode=advanced');await page.evaluate(()=>CppCanvas.ready);
        assert.equal(await page.locator('#canvas-brush-size').inputValue(),'1','A new page load starts at 1 px, even for a saved canvas');
        await page.selectOption('#canvas-tool','erase');await opacity(100);await dot(16,32);
        await page.click('#canvas-cancel');await page.waitForURL('**/cpp-ui');await page.evaluate(()=>CppCanvas.ready);
        const unchanged=await page.evaluate(()=>{const req={};CppCanvas.applyToRequest(req);return req;});
        assert.equal(unchanged.init_image_mask,payload.init_image_mask);
        // Check the other modes' exported payloads independently, then return
        // to the advanced layer without losing or baking any guidance twice.
        for(const mode of ['draw','inpaint']) {
            await page.click(`#cpp-canvas-open-${mode}`);await page.waitForURL(`**/cpp-ui/canvas?mode=${mode}`);await page.evaluate(()=>CppCanvas.ready);
            await page.click('#canvas-generate');await page.waitForURL('**/cpp-ui');await page.evaluate(()=>CppCanvas.ready);
            const request=await page.evaluate(()=>{const req={};CppCanvas.applyToRequest(req);return req;});
            assert.deepEqual(fields(request),fields(payload),'The last editor mode must not replace the other influences');
        }
        await page.uncheck('#cpp-canvas-inpaint-use');
        await page.waitForFunction(()=>!document.getElementById('cpp-canvas-inpaint-use').disabled);
        const guideOnly=await page.evaluate(()=>{const req={};CppCanvas.applyToRequest(req);return req;});
        assert.equal(guideOnly.init_image_mask,undefined);assert.equal(guideOnly.inpaint_color_guide,payload.inpaint_color_guide);
        await page.check('#cpp-canvas-inpaint-use');await page.waitForFunction(()=>!document.getElementById('cpp-canvas-inpaint-use').disabled);
        await page.uncheck('#cpp-canvas-guide-use');
        await page.waitForFunction(()=>!document.getElementById('cpp-canvas-guide-use').disabled);
        const normalOnly=await page.evaluate(()=>{const req={};CppCanvas.applyToRequest(req);return req;});
        assert.ok(normalOnly.init_image_mask);assert.equal(normalOnly.inpaint_color_guide,undefined);
        await page.uncheck('#cpp-canvas-inpaint-use');await page.waitForFunction(()=>!document.getElementById('cpp-canvas-inpaint-use').disabled);
        const drawOnly=await page.evaluate(()=>{const req={};CppCanvas.applyToRequest(req);return req;});
        assert.equal(drawOnly.init_image_mask,undefined);assert.equal(drawOnly.inpaint_color_guide,undefined);
        await page.check('#cpp-canvas-inpaint-use');await page.waitForFunction(()=>!document.getElementById('cpp-canvas-inpaint-use').disabled);
        await page.check('#cpp-canvas-guide-use');await page.waitForFunction(()=>!document.getElementById('cpp-canvas-guide-use').disabled);
        await page.click('#cpp-canvas-open-advanced');await page.waitForURL('**/cpp-ui/canvas?mode=advanced');await page.evaluate(()=>CppCanvas.ready);
        await page.click('#canvas-generate');await page.waitForURL('**/cpp-ui');await page.evaluate(()=>CppCanvas.ready);
        assert.equal(await page.evaluate(()=>{const req={};CppCanvas.applyToRequest(req);return req.init_image_mask;}),payload.init_image_mask);
        await page.click('#cpp-canvas-open-advanced');await page.waitForURL('**/cpp-ui/canvas?mode=advanced');await page.evaluate(()=>CppCanvas.ready);
        await page.setViewportSize({width:390,height:844});
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile horizontal overflow');
        await page.screenshot({path:'/tmp/cpp-advanced-inpaint-mobile.png'});
        const split=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=64;const ctx=c.getContext('2d');ctx.fillStyle='rgb(100,100,100)';ctx.fillRect(0,0,32,64);ctx.fillStyle='rgb(103,103,103)';ctx.fillRect(32,0,32,64);return c.toDataURL();});
        await page.setInputFiles('#canvas-source',{name:'split.png',mimeType:'image/png',buffer:Buffer.from(split.split(',')[1],'base64')});
        await page.waitForFunction(()=>document.getElementById('image-editor-canvas').getContext('2d').getImageData(0,0,1,1).data[0]===100);
        await page.selectOption('#canvas-tool','wand');await opacity(100);
        async function range(id,value){await page.locator(id).evaluate((el,value)=>{el.value=String(value);el.dispatchEvent(new Event('input',{bubbles:true}));},value);}
        const sample=()=>page.evaluate(()=>[16,32,33,50].map(x=>document.getElementById('image-editor-canvas').getContext('2d').getImageData(x,32,1,1).data[0]));
        await range('#canvas-wand-sensitivity',0);await range('#canvas-wand-buffer',0);await dot(16,32);assert.deepEqual(await sample(),[255,103,103,103]);
        await range('#canvas-wand-sensitivity',6);await page.click('#canvas-undo');await page.click('#canvas-redo');
        assert.deepEqual(await sample(),[255,103,103,103],'Redo retains the original selection after changing sensitivity');
        await range('#canvas-wand-sensitivity',0);
        await page.click('#canvas-clear');await range('#canvas-wand-buffer',1);await dot(16,32);assert.deepEqual(await sample(),[255,255,103,103]);
        await page.click('#canvas-clear');await range('#canvas-wand-buffer',0);await range('#canvas-wand-sensitivity',6);await dot(16,32);assert.deepEqual(await sample(),[255,255,255,255]);
        await page.click('#canvas-cancel');await page.waitForURL('**/cpp-ui');await page.evaluate(()=>CppCanvas.ready);
        await page.click('#cpp-canvas-clear');
        await page.waitForFunction(()=>document.getElementById('cpp-canvas-status').textContent.includes('No canvas image'));
        const cleared=await page.evaluate(()=>{const req={};CppCanvas.applyToRequest(req);return req;});
        assert.equal(cleared.init_image,undefined);assert.equal(cleared.init_image_mask,undefined);
        assert.deepEqual(errors,[]);
        console.log('PASS: C++ color-weighted inpaint, independent mask/guide controls, unchanged source, view-only Main preview tabs, default strength 0.65, 1px brush, magic sliders, undo/redo, API payload, desktop/mobile; no AI generation');
    } finally {await browser.close();}
}
run().catch(error=>{console.error(error);process.exitCode=1;});
