const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.EDITOR_TEST_URL || 'http://127.0.0.1:10000';

(async()=>{
    const browser=await chromium.launch({headless:true});
    try {
        const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
        page.on('pageerror',error=>errors.push(error.message));
        await page.route('**/*',route=>{
            const pathname=new URL(route.request().url()).pathname;
            if(route.request().method()==='POST' && ['/files/list_lora','/files/list_model','/files/list_vae'].includes(pathname)) {
                return route.fulfill({json:{files:[],meta:[],count:0}});
            }
            assert.equal(route.request().method(),'GET',`Editor test blocked a backend mutation: ${new URL(route.request().url()).pathname}`);
            if(new URL(route.request().url()).pathname==='/get/models') {
                return route.fulfill({json:{models:[{model:'test/sd15',name:'Test SD 1.5',tags:['stable-diffusion','sd_v1']}]}});
            }
            return route.continue();
        });
        await page.goto(base+'/legacy',{waitUntil:'domcontentloaded'});
        await page.waitForSelector('#tab-image-editor-page');
        await page.waitForFunction(()=>typeof modelsDB==='object');
        assert.equal(await page.locator('[data-editor-mode="advanced"]').count(),1,'Legacy Canvas needs its own Advanced Inpaint selection');
        if(!await page.locator('#init_image_button_advanced_inpaint').isVisible())await page.locator('#initial-image-settings h4.collapsible').click();
        assert.equal(await page.locator('#init_image_button_advanced_inpaint').isVisible(),true,'Advanced button is discoverable before loading an image');
        await page.click('#init_image_button_advanced_inpaint');
        await page.waitForSelector('#image-advanced-inpainter.editor-page-pane.active');
        assert.equal(await page.locator('#image-editor-page-content .editor-page-pane.active').count(),1);
        assert.equal(await page.evaluate(()=>advancedInpainter.inputActive),true);
        await page.locator('#image-advanced-inpainter .image-editor-button').filter({hasText:/^Cancel$/}).click();
        assert.equal(await page.evaluate(()=>initImagePreviewContainer.classList.contains('has-image')),false);
        const white=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=128;const x=c.getContext('2d');x.fillStyle='white';x.fillRect(0,0,128,128);return c.toDataURL();});
        await page.setInputFiles('#init_image',{name:'source.png',mimeType:'image/png',buffer:Buffer.from(white.split(',')[1],'base64')});
        await page.waitForFunction(()=>advancedInpainter.width===128 && advancedInpainter.layers.background.ctx.getImageData(0,0,1,1).data[0]===255);
        await page.click('#tab-image-editor-page');await page.click('[data-editor-mode="advanced"]');
        for(const [mode,popup] of [['draw','image-editor'],['inpaint','image-inpainter'],['advanced','image-advanced-inpainter']]) {
            await page.click(`[data-editor-mode="${mode}"]`);
            for(const id of ['brush-size','wand-threshold','selection-buffer'])assert.equal(await page.locator(`#${popup}-${id}`).getAttribute('type'),'range');
            assert.equal(await page.locator(`#${popup}-brush-size`).inputValue(),'1');
            assert.equal(await page.locator(`#${popup}-brush-size`).getAttribute('min'),'1');
        }
        await page.click('[data-editor-mode="advanced"]');
        const popup=page.locator('#image-advanced-inpainter');
        await popup.locator('.image_editor_color .editor-options-container > div').nth(3).click();
        await popup.locator('.image_editor_brush_size input').evaluate(el=>{el.value='10';el.dispatchEvent(new Event('input',{bubbles:true}));});
        async function stroke(x,y,editorPopup=popup){
            const overlay=editorPopup.locator('.editor-canvas-overlay');await overlay.scrollIntoViewIfNeeded();const box=await overlay.boundingBox();
            await page.mouse.move(box.x+x/128*box.width,box.y+y/128*box.height);await page.mouse.down();
            await page.mouse.move(box.x+(x+2)/128*box.width,box.y+y/128*box.height,{steps:2});await page.mouse.up();
        }
        await popup.locator('.image_editor_opacity .editor-options-container > div').nth(4).click();await stroke(32,64);
        await popup.locator('.image_editor_opacity .editor-options-container > div').nth(1).click();await stroke(96,64);
        await page.click('[data-editor-mode="draw"]');await page.click('[data-editor-mode="inpaint"]');await page.click('[data-editor-mode="advanced"]');
        assert.equal(await page.evaluate(()=>imageEditor.inputActive || imageInpainter.inputActive),false);
        await page.click('[data-editor-mode="inpaint"]');
        const normalPopup=page.locator('#image-inpainter');
        await normalPopup.locator('.image_editor_opacity .editor-options-container > div').nth(0).click();
        await normalPopup.locator('.image_editor_brush_size input').evaluate(el=>{el.value='10';el.dispatchEvent(new Event('input',{bubbles:true}));});
        await stroke(64,32,normalPopup);
        await normalPopup.locator('.image-editor-button').filter({hasText:/^Save$/}).click();
        const ordinaryMask=await page.evaluate(()=>imageInpainter.getImg());
        const originalSource=await page.evaluate(()=>initImagePreview.src);
        await page.click('#init_image_button_advanced_inpaint');
        await popup.locator('.image-editor-button').filter({hasText:/^Save$/}).click();
        await page.waitForFunction(()=>document.getElementById('enable_color_guide').checked);
        const request=await page.evaluate(()=>getCurrentUserRequest().reqBody);
        assert.ok(request.init_image.startsWith('data:image/'));assert.ok(request.mask.startsWith('data:image/'));
        assert.equal(request.init_image,originalSource,'Advanced Inpaint must not commit colors into the original');
        assert.equal(request.mask,ordinaryMask,'Normal inpaint mask must not be replaced by Advanced');
        const pixels=await page.evaluate(async url=>{const img=new Image();img.src=url;await img.decode();const c=document.createElement('canvas');c.width=img.width;c.height=img.height;const ctx=c.getContext('2d');ctx.drawImage(img,0,0);return [[32,64],[96,64],[0,0]].map(([x,y])=>[...ctx.getImageData(x,y,1,1).data]);},request.inpaint_color_guide);
        assert.ok(Math.abs(pixels[0][3]-51)<=2);assert.ok(Math.abs(pixels[1][3]-204)<=2);assert.equal(pixels[2][3],0);
        for(const mode of ['draw','inpaint','advanced']) {
            await page.click(`#legacy-input-preview-tab-${mode}`);
            assert.equal(await page.locator(`#legacy-input-preview-panel-${mode}`).isVisible(),true);
            const again=await page.evaluate(()=>getCurrentUserRequest().reqBody);
            assert.equal(again.init_image,request.init_image);assert.equal(again.mask,request.mask);assert.equal(again.inpaint_color_guide,request.inpaint_color_guide);
        }
        await page.locator('#legacy-input-preview-tab-advanced').scrollIntoViewIfNeeded();
        const previewBox=await page.locator('#init_image_wrapper').boundingBox();
        const actionBox=await page.locator('#init_image_buttons').boundingBox();
        assert.ok(previewBox.width>=200,'Preview tabs must not be squeezed into the old thumbnail column');
        assert.ok(previewBox.y+previewBox.height<=actionBox.y+1,'Preview text must not overlap action controls');
        await page.screenshot({path:'/tmp/legacy-color-inpaint-previews.png'});
        await page.setViewportSize({width:390,height:844});
        await page.locator('#legacy-input-preview-tab-advanced').scrollIntoViewIfNeeded();
        const mobilePreview=await page.locator('.image-preview-tabs').boundingBox();
        assert.ok(mobilePreview.x>=0&&mobilePreview.x+mobilePreview.width<=391,'Main preview must fit mobile width');
        await page.screenshot({path:'/tmp/legacy-color-inpaint-previews-mobile.png'});
        await page.setViewportSize({width:1440,height:1000});
        await page.locator('label[for="enable_mask"]').first().click();
        assert.equal(await page.locator('#enable_mask').isChecked(),false);
        const colorOnly=await page.evaluate(()=>getCurrentUserRequest().reqBody);
        assert.equal(colorOnly.mask,undefined);assert.equal(colorOnly.inpaint_color_guide,request.inpaint_color_guide);
        await page.locator('label[for="enable_mask"]').first().click();
        await page.locator('label[for="enable_color_guide"]').last().click();
        assert.equal(await page.locator('#enable_color_guide').isChecked(),false);
        const normalOnly=await page.evaluate(()=>getCurrentUserRequest().reqBody);
        assert.equal(normalOnly.mask,ordinaryMask);assert.equal(normalOnly.inpaint_color_guide,undefined);
        await page.locator('label[for="enable_color_guide"]').last().click();
        // Magic wand uses the same sensitivity/buffer algorithm in both UIs.
        const split=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=64;const ctx=c.getContext('2d');ctx.fillStyle='rgb(100,100,100)';ctx.fillRect(0,0,32,64);ctx.fillStyle='rgb(103,103,103)';ctx.fillRect(32,0,32,64);return c.toDataURL();});
        await page.setInputFiles('#init_image',{name:'split.png',mimeType:'image/png',buffer:Buffer.from(split.split(',')[1],'base64')});
        await page.waitForFunction(()=>advancedInpainter.width===64 && advancedInpainter.layers.background.ctx.getImageData(0,0,1,1).data[0]===100);
        await page.click('#init_image_button_advanced_inpaint');
        await popup.getByText('Magic Wand',{exact:true}).click();
        await popup.locator('.image_editor_opacity .editor-options-container > div').nth(0).click();
        async function range(id,value){await popup.locator(id).evaluate((el,value)=>{el.value=String(value);el.dispatchEvent(new Event('input',{bubbles:true}));},value);}
        async function wand(){const el=popup.locator('.editor-canvas-overlay');await el.scrollIntoViewIfNeeded();const b=await el.boundingBox();await page.mouse.click(b.x+b.width/4,b.y+b.height/2);}
        const sample=()=>page.evaluate(()=>[16,32,33,50].map(x=>advancedInpainter.layers.drawing.ctx.getImageData(x,32,1,1).data[3]));
        const clear=()=>popup.locator('.image-editor-button').filter({hasText:/^Clear$/}).click();
        await range('#image-advanced-inpainter-wand-threshold',0);await range('#image-advanced-inpainter-selection-buffer',0);await wand();
        assert.deepEqual(await sample(),[255,0,0,0]);
        await range('#image-advanced-inpainter-wand-threshold',6);
        await popup.locator('.image-editor-button').filter({hasText:/^Undo$/}).click();
        await popup.locator('.image-editor-button').filter({hasText:/^Redo$/}).click();
        assert.deepEqual(await sample(),[255,0,0,0],'Redo must replay the recorded sensitivity, not the currently selected slider value');
        assert.equal(await popup.locator('#image-advanced-inpainter-wand-threshold').inputValue(),'6','History replay must not change the current slider choice');
        await range('#image-advanced-inpainter-wand-threshold',0);
        await clear();await range('#image-advanced-inpainter-selection-buffer',1);await wand();assert.deepEqual(await sample(),[255,255,0,0]);
        await clear();await range('#image-advanced-inpainter-selection-buffer',0);await range('#image-advanced-inpainter-wand-threshold',6);await wand();assert.deepEqual(await sample(),[255,255,255,255]);
        await popup.locator('.image-editor-button').filter({hasText:/^Save$/}).click();
        const opaqueGuide=await page.evaluate(()=>getCurrentUserRequest().reqBody.inpaint_color_guide);
        assert.equal(Buffer.from(opaqueGuide.split(',')[1],'base64')[25],6,'A fully opaque guide must still export RGBA weights');
        assert.equal(await page.evaluate(()=>initImagePreview.src),split,'Full color weight must not paint into the original');
        await page.click('#init_image_button_advanced_inpaint');
        await page.screenshot({path:'/tmp/legacy-advanced-inpaint-desktop.png'});
        await page.setViewportSize({width:390,height:844});
        for(const id of ['brush-size','wand-threshold','selection-buffer']) {
            const slider=popup.locator(`#image-advanced-inpainter-${id}`);await slider.scrollIntoViewIfNeeded();const box=await slider.boundingBox();
            assert.ok(box.width>100&&box.x>=0&&box.x+box.width<=391,`Clipped ${id} slider`);
        }
        await page.screenshot({path:'/tmp/legacy-advanced-inpaint-mobile.png'});
        await page.click('[data-editor-mode="generate"]');
        assert.equal(await page.evaluate(()=>imageEditor.inputActive||imageInpainter.inputActive||advancedInpainter.inputActive),false);
        assert.deepEqual(errors,[]);
        console.log('PASS: legacy color-weighted inpaint, independent ordinary mask and RGBA guide, original unchanged, view-only Main preview tabs, responsive layout, 1px defaults, magic sliders, undo/redo; no generation');
    } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
