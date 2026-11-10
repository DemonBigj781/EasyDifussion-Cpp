// Real C++ markup, dropdowns, preferences, queue, and request builder; no model inference.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const {execFileSync} = require("node:child_process");
const root = path.resolve(__dirname, "..");
const read = name => fs.readFileSync(path.join(root, name), "utf8");
const renderer = process.env.CACHE_TEST_RENDERER || path.join(root, "source/UI.cpp/build/bin/easy-diffusion-ui-render");
const models = [
    {model:"renamed-sd15",tags:["stable-diffusion","sd_v1_base"]},
    {model:"renamed-xl",tags:["stable-diffusion","sd_xl_base"]},
    {model:"sd2",tags:["stable-diffusion","sd_v2_base"]},
    {model:"mochi-image",tags:["stable-diffusion","sd_xl_base"]},
    {model:"native-mochi",tags:["stable-diffusion","mochi_v1_preview"]},
    {model:"motion",tags:["motion-module","animatediff"]},
];
const scripts = ["ui/media/js/utils.js", "ui/media/js/searchable-models.js",
    ...["input_preferences", "generation_queue", "generate"].map(name => `source/UI.cpp/Pages/src/Plugin/plugin_scripts/${name}.js`)];
const assets = new Map(scripts.map((file,index) => [`/test-${index}.js`,file]));
assets.set("/routing-test.js", "source/UI.cpp/Pages/src/Plugin/plugin_scripts/native-device-routing.js");
assets.set("/cpp-ui/assets/ui.css", "source/UI.cpp/Pages/assets/ui.css");
assets.set("/media/css/themes.css", "ui/media/css/themes.css");
assets.set("/media/css/fonts.css", "ui/media/css/fonts.css");
assets.set("/media/css/searchable-models.css", "ui/media/css/searchable-models.css");
for (const font of fs.readdirSync(path.join(root,"ui/media/fonts"))) {
    if (font.endsWith(".woff2") || font.endsWith(".woff")) assets.set(`/media/fonts/${font}`,`ui/media/fonts/${font}`);
}
const html = execFileSync(renderer,["/"],{encoding:"utf8"}).replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "")
    .replace("</head>", '<link rel="stylesheet" href="/media/css/searchable-models.css"></head>')
    .replace("</body>", '<script>document.documentElement.dataset.kiosk="off";window.CppKiosk={ready:Promise.resolve({enabled:false})};</script>'
        + scripts.map((_,index) => `<script src="/test-${index}.js"></script>`).join("") + "</body>");
const server = http.createServer((req,res) => {
    if (req.method !== "GET") {res.writeHead(409,{"Content-Type":"application/json"});return res.end(JSON.stringify({detail:"Test intercepted: no inference"}));}
    if (req.url === "/") {res.setHeader("Content-Type","text/html");return res.end(html);}
    const catalog = {"/get/model":models.filter(m => m.tags[0] === "stable-diffusion"),
        "/get/vae":[],"/get/other":models.filter(m => m.tags[0] === "motion-module")};
    if (catalog[req.url]) {res.setHeader("Content-Type","application/json");return res.end(JSON.stringify({models:catalog[req.url]}));}
    const file = assets.get(req.url);
    if (!file) {res.writeHead(404);return res.end();}
    res.setHeader("Content-Type",file.endsWith(".css") ? "text/css" : file.endsWith(".js") ? "text/javascript" : "font/woff2");
    res.end(fs.readFileSync(path.join(root,file)));
});

async function run() {
    await new Promise(resolve => server.listen(process.env.CACHE_TEST_PORT || 0,"127.0.0.1",resolve));
    const url = `http://127.0.0.1:${server.address().port}`;
    if (process.argv.includes("--serve")) {console.log(url);return;}
    const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
    let browser;
    try {
        browser = await chromium.launch({headless:true, ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
        const page = await browser.newPage({viewport:{width:1100,height:850}}), errors = [], requests = [];
        page.on("pageerror",error => errors.push(error.message));
        page.on("request",request => {if (request.method() === "POST") requests.push(request.postDataJSON());});
        await page.goto(url);
        await page.waitForSelector("#motion_module_model-model-list",{state:"attached"});
        async function choose(id,model) {
            await page.locator(`#${id}`).click();
            await page.locator(`#${id}-model-list .model-file[data-path=${JSON.stringify(model)}]`).click();
        }
        await choose("stable_diffusion_model","renamed-sd15");
        await choose("motion_module_model","motion");
        assert.equal(await page.locator("#video_mode").isChecked(),true);
        assert.equal(await page.locator("#steps").isDisabled(),false);
        await page.fill("#steps","4");
        await page.fill("#video_steps","12");
        const stepRequest = await page.evaluate(() => CppGeneration.buildRequest());
        assert.equal(stepRequest.num_inference_steps,4);
        assert.equal(stepRequest.video_steps,12);
        assert.equal(await page.locator("#tome_ratio").inputValue(),"0");
        assert.equal(await page.locator("#negative_guidance_min_sigma").inputValue(),"0");
        await page.fill("#tome_ratio","0.25");
        await page.fill("#negative_guidance_min_sigma","1");
        const optimized = await page.evaluate(() => CppGeneration.buildRequest());
        assert.equal(optimized.tome_ratio,.25);
        assert.equal(optimized.negative_guidance_min_sigma,1);
        await page.selectOption("#video_cache_mode","easycache");
        assert.match(await page.evaluate(() => {try {CppGeneration.buildRequest();return "accepted";} catch(error) {return error.message;}}),/requires video caching to be disabled/);
        await page.fill("#negative_guidance_min_sigma","0");
        assert.deepEqual(await page.locator("#video_cache_mode option").evaluateAll(elements => elements.map(e => e.value)),["disabled","easycache","teacache"]);
        for (const model of ["renamed-sd15","renamed-xl","mochi-image"]) {
            await choose("stable_diffusion_model",model);
            for (const mode of ["easycache","teacache","disabled","easycache"]) {
                await page.selectOption("#video_cache_mode",mode);
                const request = await page.evaluate(() => CppGeneration.buildRequest());
                assert.equal(request.cache_mode,mode);
                assert.equal(request.motion_module_model,"motion");
                assert.equal(request.video_mode,true);
                assert.equal(await page.locator("#video_cache_threshold").isDisabled(),mode === "disabled");
            }
        }
        await choose("stable_diffusion_model","native-mochi");
        assert.equal(await page.locator("#motion_module_model").isDisabled(),true);
        await page.fill("#tome_ratio","0.25");
        await page.fill("#negative_guidance_min_sigma","1");
        await page.selectOption("#video_cache_mode","disabled");
        const nativeVideoRequest = await page.evaluate(() => CppGeneration.buildRequest());
        assert.equal(nativeVideoRequest.motion_module_model,undefined);
        assert.equal(nativeVideoRequest.tome_ratio,0);
        assert.equal(nativeVideoRequest.negative_guidance_min_sigma,0);
        await choose("stable_diffusion_model","renamed-sd15");
        assert.equal(await page.locator("#motion_module_model").isDisabled(),false);
        assert.equal(await page.locator("#motion_module_model").getAttribute("data-path"),"motion");
        await page.fill("#negative_guidance_min_sigma","0");
        await page.selectOption("#video_cache_mode","easycache");
        await page.fill("#video_cache_threshold","0.08");
        await page.fill("#video_cache_start","1");
        await page.fill("#video_cache_end","90");
        const request = await page.evaluate(() => CppGeneration.buildRequest());
        assert.equal(request.cache_threshold,.08);assert.equal(request.cache_start_percent,1);assert.equal(request.cache_end_percent,90);
        await page.fill("#prompt","cache plumbing fixture");
        await page.click("#makeImage");
        await page.waitForFunction(() => document.getElementById("generation-queue-status").textContent.includes("Test intercepted"));
        assert.equal(requests.length,1);assert.equal(requests[0].cache_mode,"easycache");
        await page.fill("#video_cache_start","95");
        const validation = await page.evaluate(() => {try {CppGeneration.buildRequest();return "accepted";} catch(e) {return e.message;}});
        assert.match(validation,/Cache range/);
        await page.fill("#video_cache_start","1");
        await page.reload();
        await page.waitForFunction(() => document.getElementById("video_cache_threshold").value === "0.08");
        assert.equal((await page.evaluate(() => CppGeneration.buildRequest())).cache_mode,"easycache");
        assert.equal((await page.evaluate(() => CppGeneration.buildRequest())).tome_ratio,.25);
        assert.equal((await page.evaluate(() => CppGeneration.buildRequest())).negative_guidance_min_sigma,0);
        await page.uncheck("#video_mode");
        assert.equal((await page.evaluate(() => CppGeneration.buildRequest())).cache_mode,undefined);
        assert.equal(await page.locator("#video_cache_mode").isDisabled(),true);
        await choose("stable_diffusion_model","sd2");
        assert.equal(await page.locator("#video_mode").isDisabled(),true);
        await choose("stable_diffusion_model","renamed-xl");
        await page.selectOption("#video_cache_mode","teacache");
        await page.locator("#video-cache-settings").scrollIntoViewIfNeeded();
        await page.screenshot({path:"/tmp/cpp-video-cache-desktop.png"});
        await page.setViewportSize({width:390,height:844});
        await page.locator("#video-cache-settings").scrollIntoViewIfNeeded();
        const box = await page.locator("#video-cache-settings").boundingBox();
        assert.ok(box.x >= 0 && box.x + box.width <= 390);
        await page.screenshot({path:"/tmp/cpp-video-cache-mobile.png"});
        await page.locator("#tome_ratio").scrollIntoViewIfNeeded();
        const optimizationBox = await page.locator("#negative_guidance_min_sigma").boundingBox();
        assert.ok(optimizationBox.x >= 0 && optimizationBox.x + optimizationBox.width <= 390);
        await page.screenshot({path:"/tmp/cpp-sampling-optimizations-mobile.png"});
        assert.deepEqual(errors,[]);
        console.log("PASS: C++ cache selection, SD1.5/SDXL, exclusive modes, threshold/range, queue payload, reload, disable, SD2 gating, desktop/mobile");
    } finally {if (browser) await browser.close();server.close();}
}
run().catch(error => {console.error(error);server.close();process.exitCode=1;});
