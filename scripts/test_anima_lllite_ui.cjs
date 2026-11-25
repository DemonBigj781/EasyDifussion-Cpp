const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
    const browser = await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH});
    try {
        for (const path of ["/"]) {
            const context = await browser.newContext({viewport:{width:1280,height:960}});
            const page = await context.newPage();
            const errors=[];
            page.on("pageerror", error => errors.push(error.message));
            await page.route("**/*", route => route.request().method() === "GET" ? route.continue() : route.abort());
            await page.goto((process.env.LLLITE_TEST_URL || "http://127.0.0.1:10000") + path);
            await page.waitForSelector("#lllite-model", {state:"attached",timeout:30000});
            if (!await page.locator("#stable_diffusion_model").isVisible()) await page.locator("#editor-settings > h4").click();
            await page.fill("#stable_diffusion_model", "anima-base-v1.0");
            await page.locator('#stable_diffusion_model-model-list [data-path="Anima/anima-base-v1.0"]').click();
            if (!await page.locator("#controlnet_mode").isVisible()) await page.locator("#sdkit3-controlnet-panel > h4").click();
            await page.selectOption("#controlnet_mode", "lllite");
            await page.fill("#lllite-model", "anima-lllite");
            const visible = page.locator('#lllite-model-model-list .model-file:visible');
            const names = await visible.evaluateAll(elements => elements.map(el=>el.dataset.path).filter(Boolean));
            console.log("LLLite visible",path,names);
            assert.equal(names.length,5,JSON.stringify(names));
            assert.ok(names.every(name=>name.startsWith("Anima/")));
            assert.ok(names.every(name=>!name.includes("inpainting")));
            await page.locator('#lllite-model-model-list [data-path="Anima/kohya-ss/base-v1/anima-lllite-any-test-like-v2"]').click();
            await page.screenshot({path:"/tmp/anima-lllite-legacy.png"});
            console.log("PASS",path,"5 actual installed compatible weights",JSON.stringify(errors));
            await context.close();
        }
    } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
