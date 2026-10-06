// Real compiled pages and scripts; every HTTP request is intercepted. No GPU work or server mutation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const renderer = process.env.CPP_UI_RENDERER || path.join(root,'source/UI.cpp/build/bin/easy-diffusion-ui-render');
const ids = ['accessibility-improvements','animate','daily-folders','disable-source-image-zoom',
    'gpu-mode-quick-toggle','make-image-always-visible','processing-order-quick-toggle','prompt-diff',
    'prompt-translator','queue-counter','rabbit-hole','random-seed-quick-toggle','seed-randomizer',
    'spell-tokenizer','stig-image-to-img2img','stig-image-utilities','stig-lora-shuttle',
    'stig-text-to-prompt','storyteller','template-manager','toggle-spellcheck'];
const png = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=';
const models = [{model:'sd15',tags:['sd_v1']},{model:'second-model',tags:['sd_v1']},
    {model:'test-vae',tags:['vae']},{model:'test-lora',tags:['lora']}];
const requests=[],translations=[],errors=[],unexpected=[];
let streamDelay=20,failNext=false,translationFails=false;

(async()=>{
    const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
    try {
        const context=await browser.newContext({viewport:{width:1280,height:960}});
        context.on('page',page=>{
            page.on('pageerror',error=>errors.push(error.message));
            page.on('console',msg=>{if(msg.type()==='error')console.error('BROWSER:',msg.text());});
        });
        await context.addInitScript(ids=>{
            if(location.protocol==='data:' || location.href==='about:blank')return;
            if(localStorage.getItem('test-initialized'))return;
            localStorage.setItem('test-initialized','true');
            localStorage.setItem('easy-diffusion-enabled-local-plugins-v1',JSON.stringify(ids));
            localStorage.setItem('easy-diffusion-local-plugin-defaults-version','3');
        },ids);
        await context.route('**/*',async route=>{
            const req=route.request(),url=new URL(req.url()),p=url.pathname;
            if(url.hostname==='translate.googleapis.com'){
                translations.push(url.searchParams.get('q'));
                return route.fulfill(translationFails?{status:503,json:{}}:{json:[[['translated ',''],['prompt','']]]});
            }
            if(p==='/render'){
                const request=req.postDataJSON();requests.push(request);
                if(failNext){failNext=false;return route.fulfill({status:409,json:{detail:'Test render failure'}});}
                return route.fulfill({json:{task:requests.length,stream:`/image/stream/${requests.length}`}});
            }
            assert.equal(req.method(),'GET',`Unexpected mutation: ${req.method()} ${p}`);
            if(p.startsWith('/image/stream/')){
                await new Promise(resolve=>setTimeout(resolve,streamDelay));
                const request=requests[Number(p.split('/').at(-1))-1];
                return route.fulfill({json:{status:'succeeded',output:Array.from({length:request.num_outputs||1},(_,i)=>({data:png,seed:request.seed+i})),task_data:{output_format:'png'}}});
            }
            if(p==='/image/stop')return route.fulfill({json:{status:'OK'}});
            if(p==='/kiosk')return route.fulfill({json:{enabled:false,allowed_models:[]}});
            if(p==='/get/modifiers')return route.fulfill({json:[]});
            if(p.endsWith('/merged_2024-12-22_pt2-ia-dd-ed.csv'))return route.fulfill({contentType:'text/csv',body:'blue_eyes,0,10000,azure_eyes\nblue_hair,0,9000,\ncat,0,8000,\n'});
            if(p.startsWith('/get/')){
                const found=p==='/get/model'?models.filter(x=>x.tags.includes('sd_v1')):p==='/get/lora'?models.filter(x=>x.tags.includes('lora')):p==='/get/vae'?models.filter(x=>x.tags.includes('vae')):p==='/get/models'?models:[];
                return route.fulfill({json:{models:found}});
            }
            if(p==='/cpp-ui' || p==='/cpp-ui/settings/plugins')return route.fulfill({contentType:'text/html',body:execFileSync(renderer,[p==='/cpp-ui'?'/':'/settings/plugins'])});
            let file;
            if(p.startsWith('/cpp-ui/scripts/'))file=path.join(root,'source/UI.cpp/Pages/src/Plugin/plugin_scripts',p.slice('/cpp-ui/scripts/'.length));
            else if(p.startsWith('/cpp-ui/assets/'))file=path.join(root,'source/UI.cpp/Pages/assets',p.slice('/cpp-ui/assets/'.length));
            else if(p.startsWith('/media/'))file=path.join(root,'ui',p);
            else if(p.startsWith('/plugins/core/'))file=path.join(root,'ui/plugins/ui',p.slice('/plugins/core/'.length));
            if(file && fs.existsSync(file)){
                const contentType={'.js':'text/javascript','.json':'application/json','.css':'text/css','.html':'text/html','.woff2':'font/woff2','.woff':'font/woff'}[path.extname(file)]||'application/octet-stream';
                return route.fulfill({contentType,body:fs.readFileSync(file)});
            }
            if(req.resourceType()==='font'||p==='/favicon.ico')return route.fulfill({status:204});
            unexpected.push(p);return route.fulfill({status:404,body:'Not in fixture'});
        });
        const page=await context.newPage();await page.goto('http://cpp-plugins.test/cpp-ui');
        await page.waitForFunction(()=>window.CppCreativePlugins && window.CppOptionalPlugins && document.querySelector('#vae_model-model-list'));
        assert.equal(await page.title(),'Main · Easy Diffusion C++');
        await page.fill('#stable_diffusion_model','sd15');await page.locator('#stable_diffusion_model-model-list [data-path="sd15"]').click();
        await page.fill('#prompt','red cat');await page.fill('#seed','42');
        await page.selectOption('#vram_usage_level','low');
        await page.selectOption('#seed_randomizer_behavior','high');
        await page.click('#makeImage');await page.waitForFunction(()=>window.CppGeneration.state.images===0 && document.querySelector('.cpp-generated-image'));
        assert.equal(requests[0].vram_usage_level,'low');
        assert.equal(requests[0].seed,await page.evaluate(()=>CppOptionalPluginMath.crc32('red cat_42')));
        assert.match(requests[0].session_id,/^\d{4}-\d{2}-\d{2}$/);
        await page.selectOption('#seed_randomizer_behavior','normal');
        await page.fill('#prompt','blue cat <script>bad</script>');await page.click('#makeImage');
        await page.waitForFunction(()=>document.querySelectorAll('.cpp-generated-image').length===2);
        assert.match(await page.locator('.cpp-prompt-diff').first().innerText(),/blue cat/);
        assert.equal(await page.locator('.cpp-prompt-diff script').count(),0);
        await page.selectOption('#contextual_menu_invocation','left_click');
        const card=page.locator('.cpp-generated-image').first();
        assert.equal(await card.locator('.cpp-image-actions').first().evaluate(el=>getComputedStyle(el).visibility),'hidden');
        await card.locator('img').first().click();assert.equal(await card.getAttribute('data-actions-open'),'');
        await page.selectOption('#contextual_menu_invocation','hover');
        await page.check('#cpp-random-seed');assert.equal(await page.inputValue('#seed'),'-1');
        await page.uncheck('#cpp-random-seed');assert.equal(await page.inputValue('#seed'),'42');
        assert.equal(await page.locator('#cpp-floating-generate').isVisible(),true);

        // Live queue order and counter, rather than merely checking a toggle label.
        streamDelay=250;const first=requests.length;
        await page.selectOption('#cpp-processing-order','newest');
        await page.evaluate(()=>{window.queueChecks=[1,2,3].map(seed=>CppGeneration.enqueue({...CppGeneration.buildRequest(),seed,prompt:`order ${seed}`}));});
        await page.waitForFunction(()=>CppGeneration.state.pending===2);
        assert.match(await page.title(),/^\(3\/3\)/);
        await page.evaluate(()=>Promise.all(window.queueChecks));
        assert.deepEqual(requests.slice(first).map(request=>request.seed),[1,3,2]);
        assert.equal(await page.title(),'Main · Easy Diffusion C++');streamDelay=20;
        await page.selectOption('#cpp-processing-order','oldest');
        // Failure is visible and the queue remains usable.
        failNext=true;await page.click('#makeImage');await page.waitForFunction(()=>document.querySelector('#generation-queue-status').textContent.includes('Test render failure'));

        await page.waitForFunction(()=>document.querySelector('#prompt_language option[value="fr"]'));
        await page.selectOption('#prompt_language','fr');await page.fill('#prompt','bonjour & chat?');await page.fill('#negative_prompt','flou');
        const beforeConsent=requests.length;await page.click('#makeImage');
        await page.waitForFunction(()=>document.querySelector('#generation-queue-status').textContent.includes('Allow sending'));
        assert.equal(requests.length,beforeConsent);assert.equal(translations.length,0);
        await page.check('#cpp-translation-consent');await page.click('#makeImage');
        await page.waitForFunction(()=>CppGeneration.state.images===0 && document.querySelector('#generation-queue-status').textContent==='Generation complete.');
        assert.deepEqual(translations,['bonjour & chat?','flou']);assert.equal(requests.at(-1).prompt,'translated prompt');
        translationFails=true;await page.fill('#prompt','nouveau');await page.click('#makeImage');
        await page.waitForFunction(()=>document.querySelector('#generation-queue-status').textContent.includes('Google Translate failed'));
        translationFails=false;await page.selectOption('#prompt_language','en');await page.fill('#negative_prompt','');await page.fill('#prompt','cat');

        await page.locator('#cpp-rabbit-hole summary').click();
        await page.fill('#cpp-rabbit-max','4');await page.fill('#cpp-rabbit-seeds','2');await page.fill('#cpp-rabbit-stepsCount','2');
        const beforeRabbit=requests.length;await page.click('#cpp-rabbit-start');
        await page.waitForFunction(()=>document.querySelector('#cpp-creative-status').textContent==='Completed 4 Rabbit Hole images.');
        assert.equal(requests.length-beforeRabbit,4);assert.ok(requests.slice(beforeRabbit).every(request=>request.num_outputs===1));
        await page.fill('#cpp-rabbit-chain','2');const beforeChain=requests.length;
        await page.locator('.cpp-generated-image').first().hover();
        await page.locator('.cpp-generated-image').first().getByRole('button',{name:'Img2img chain',exact:true}).click();
        await page.waitForFunction(n=>document.querySelectorAll('.cpp-source-preview').length>=n,2);
        await page.waitForFunction(()=>CppGeneration.state.images===0);
        assert.equal(requests.length-beforeChain,2);assert.ok(requests.at(-1).init_image.startsWith('data:image/'));
        await page.uncheck('#disable_source_image_zoom');assert.equal(await page.getAttribute('html','data-cpp-no-source-zoom'),'');

        await page.locator('#cpp-animate summary').click();
        const imageBuffer=Buffer.from(png.split(',')[1],'base64');
        await page.setInputFiles('#cpp-animate-files',[{name:'one.png',mimeType:'image/png',buffer:imageBuffer},{name:'two.png',mimeType:'image/png',buffer:imageBuffer}]);
        await page.fill('#width','64').catch(()=>page.locator('#width').evaluate(el=>{el.value='64';el.dispatchEvent(new Event('input'));}));
        await page.locator('#height').evaluate(el=>{el.value='64';el.dispatchEvent(new Event('input'));});
        const beforeAnimate=requests.length;await page.selectOption('#cpp-animate-format','gif');await page.click('#cpp-animate-start');
        await page.waitForFunction(()=>document.querySelector('#cpp-animate-status').textContent==='Animation complete.',{},{timeout:30000}).catch(async error=>{console.error('ANIMATE:',await page.locator('#cpp-animate-status').innerText());throw error;});
        assert.equal(requests.length-beforeAnimate,2);
        const gifData=await page.locator('.cpp-creative-output a').evaluate(async link=>Array.from(new Uint8Array(await (await fetch(link.href)).arrayBuffer())));
        assert.equal(Buffer.from(gifData).subarray(0,6).toString(),'GIF89a');
        // Decode the exported GIF and rerender its frames, exercising the input path too.
        await page.setInputFiles('#cpp-animate-files',{name:'roundtrip.gif',mimeType:'image/gif',buffer:Buffer.from(gifData)});
        await page.selectOption('#cpp-animate-format','render');await page.click('#cpp-animate-start');
        await page.waitForFunction(()=>document.querySelector('#cpp-animate-status').textContent==='Animation complete.',{},{timeout:30000});
        await page.setInputFiles('#cpp-animate-files',[{name:'one.png',mimeType:'image/png',buffer:imageBuffer},{name:'two.png',mimeType:'image/png',buffer:imageBuffer}]);
        await page.selectOption('#cpp-animate-format','video');await page.click('#cpp-animate-start');
        await page.waitForFunction(()=>document.querySelector('#cpp-animate-status').textContent==='Animation complete.',{},{timeout:30000});
        assert.equal(await page.locator('.cpp-creative-output a').getAttribute('download'),'animation.webm');
        const videoData=await page.locator('.cpp-creative-output a').evaluate(async link=>Array.from(new Uint8Array(await (await fetch(link.href)).arrayBuffer())));
        assert.equal(Buffer.from(videoData).readUInt32BE(0),0x1a45dfa3);
        await page.setInputFiles('#cpp-animate-files',{name:'roundtrip.webm',mimeType:'video/webm',buffer:Buffer.from(videoData)});
        await page.selectOption('#cpp-animate-format','render');await page.fill('#cpp-animate-to','.2');await page.click('#cpp-animate-start');
        await page.waitForFunction(()=>document.querySelector('#cpp-animate-status').textContent==='Animation complete.',{},{timeout:30000});

        await require('./cpp_workbench_ui_checks.cjs')(context,page,requests);
        // Cancellation while a request is in flight must not start queued successors.
        streamDelay=400;const beforeCancel=requests.length;
        await page.evaluate(()=>{window.cancelChecks=[1,2,3].map(seed=>CppGeneration.enqueue({...CppGeneration.buildRequest(),seed}).then(()=>true,()=>false));});
        await page.waitForFunction(()=>CppGeneration.state.pending===2);
        await page.click('#stopImage');await page.evaluate(()=>Promise.all(window.cancelChecks));
        assert.ok(requests.length-beforeCancel<=1);streamDelay=20;
        const settings=await context.newPage();await settings.goto('http://cpp-plugins.test/cpp-ui/settings/plugins');
        await settings.waitForSelector('[data-optional-plugin-id="animate"]');
        for(const id of ids)assert.equal((await settings.locator(`[data-optional-plugin-status="${id}"]`).innerText()).includes('legacy UI'),false,id);
        await settings.locator('[data-optional-plugin-id="animate"]').uncheck();
        await page.waitForSelector('#cpp-animate',{state:'hidden'});
        await page.evaluate(()=>LocalPluginPreferences.setEnabled('animate',true));
        await settings.waitForFunction(()=>document.querySelector('[data-optional-plugin-id="animate"]').checked);
        await page.reload();await page.waitForFunction(()=>window.CppCreativePlugins);assert.equal(await page.locator('#cpp-animate').isVisible(),true);
        await page.evaluate(()=>LocalPluginPreferences.setEnabled('prompt-assist',true));
        await page.reload();await page.waitForSelector('#spell-tokenizer-plugin-token-counter',{state:'attached'});
        await page.waitForFunction(()=>window.CppWildcard);
        assert.equal(await page.getAttribute('#prompt','spellcheck'),'false','Standalone spellcheck preference must survive prompt-assist initialization');
        await page.evaluate(()=>LocalPluginPreferences.setEnabled('toggle-spellcheck',false));
        assert.equal(await page.getAttribute('#prompt','spellcheck'),'true','Prompt assistance resumes its own setting when the standalone toggle is disabled');
        await page.evaluate(()=>LocalPluginPreferences.setEnabled('toggle-spellcheck',true));
        assert.equal(await page.getAttribute('#prompt','spellcheck'),'false');
        await page.evaluate(()=>LocalPluginPreferences.setEnabled('prompt-assist',false));
        await page.reload();await page.waitForFunction(()=>window.CppCreativePlugins&&window.CppWildcard);
        const beforePolicy=requests.length;
        const denied=await page.evaluate(async()=>{
            const request=CppGeneration.buildRequest();document.documentElement.dataset.kiosk='unavailable';
            try{await CppGeneration.enqueue(request);return '';}catch(error){return error.message;}
            finally{document.documentElement.dataset.kiosk='off';}
        });
        assert.match(denied,/Cannot verify kiosk policy/);assert.equal(requests.length,beforePolicy);
        await page.screenshot({path:'/tmp/modern-plugin-ports-desktop.png'});
        await page.setViewportSize({width:390,height:844});
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile horizontal overflow');
        await page.locator('#cpp-rabbit-hole summary').click();await page.locator('#cpp-rabbit-hole').scrollIntoViewIfNeeded();
        await page.screenshot({path:'/tmp/modern-plugin-ports-mobile.png'});
        await page.evaluate(()=>LocalPluginPreferences.saveEnabled(new Set()));
        assert.equal(await page.locator('#cpp-optional-plugins').isVisible(),false);
        assert.equal(await page.locator('#cpp-animate').isVisible(),false);
        assert.equal(await page.locator('#cpp-rabbit-hole').isVisible(),false);
        assert.equal(await page.locator('#cpp-floating-generate').isVisible(),false);
        const untouched=await page.evaluate(async()=>{
            const request={prompt:'original',negative_prompt:'negative',seed:42,session_id:'original-session',vram_usage_level:'balanced'};
            await CppOptionalPlugins.prepare(request);return request;
        });
        assert.deepEqual(untouched,{prompt:'original',negative_prompt:'negative',seed:42,session_id:'original-session',vram_usage_level:'balanced'});
        assert.deepEqual(unexpected,[]);assert.deepEqual(errors,[]);
        console.log('PASS: all 21 modern plugin ports; FIFO/LIFO and cancellation; translation consent/error; XSS-safe diff; Rabbit Hole combinations/chain; GIF and WebM input/output roundtrips; cross-tab preferences both directions; desktop/mobile; zero page exceptions; no real backend requests.');
    }finally{await browser.close();}
})().catch(error=>{console.error(error);console.error({errors,unexpected,requests:requests.length});process.exitCode=1;});
