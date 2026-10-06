const assert = require("node:assert/strict")
const fs = require("node:fs")
const http = require("node:http")
const path = require("node:path")
const root = path.resolve(__dirname, "..")
const source = name => fs.readFileSync(path.join(root, name), "utf8")
const models = [
    {model: "image", tags: ["stable-diffusion", "sdxl"]},
    {model: "video/mochi", tags: ["stable-diffusion", "mochi_v1_preview"]},
    {model: "video/ltx-distilled", tags: ["stable-diffusion", "ltx_video_v0_9_7"]},
    {model: "external/wan", tags: ["video", "wan_t2v_14b"]},
    {model: "video/ltx-2.3-dev", tags: ["stable-diffusion", "ltx2_3"]},
    ...["ltx-2.3-22b-dev_video_vae", "ltx-2.3-22b-dev_audio_vae"].map(model => ({model,tags:["vae"]})),
    ...["gemma-3-12b-it-Q4_K_M", "ltx-2.3-22b-dev_embeddings_connectors"].map(model => ({model,tags:["text-encoder"]})),
    ...["image-vae", "mochi/mochi_vae", "mochi/mochi_vae_fp8_e4m3fn", "wan/wan_2.1_vae"].map(model => ({model, tags: ["vae"]})),
    ...["image-encoder", "t5xxl_fp8_e4m3fn", "umt5_xxl_fp8_e4m3fn_scaled"].map(model => ({model, tags: ["text-encoder"]})),
    {model: "t5xxl_fp16", tags: ["text-encoder"], installed: false},
]
const fixture = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Legacy video selection — component regression</title><link rel="stylesheet" href="/ui/media/css/searchable-models.css">
<style>
body {font:16px system-ui;background:#202329;color:#eceff5;margin:20px auto;padding:0 16px;max-width:850px}
.panel-box {background:#30343c;border-radius:8px;padding:18px;margin:16px 0}.displayNone{display:none!important}
input,select,button {font:inherit;color:inherit;background:#202329;border:1px solid #687080;border-radius:4px;padding:6px;box-sizing:border-box;max-width:100%}
.model-input{position:relative;min-width:0}.model-filter{width:100%}.model-list{background:#252930;z-index:5;max-height:240px;overflow:auto;max-width:100%;box-sizing:border-box}
.model-list li{cursor:pointer}.model-list .selected{background:#52638c}small{display:block;line-height:1.5;margin-top:12px}
.options-grid{display:grid;grid-template-columns:130px minmax(0,1fr);gap:12px}h4{margin:0 0 12px;cursor:pointer}
.model_name{position:relative}.model_entry{margin-bottom:8px}.add_model_entry{margin-top:4px}
@media(max-width:700px){.options-grid{grid-template-columns:1fr}body{margin:8px auto}}
</style></head><body><h2>Legacy UI video selection</h2><p>Isolated check using the real dropdowns and video plugin. No generation backend.</p>
<div id="editor-settings" class="panel-box"><h4>Options</h4><div class="options-grid">
<label for="stable_diffusion_model">Model:</label><div class="model-input"><input id="stable_diffusion_model" data-path="image"></div>
<label for="vae_model">VAE:</label><div class="model-input"><input id="vae_model" data-path="image-vae"></div>
<label for="text_encoder_model">Text Encoder:</label><div id="text_encoder_model" data-path='{"modelNames":["image-encoder"],"modelWeights":[0.5]}'></div>
</div></div><script src="/ui/media/js/utils.js"></script><script src="/ui/media/js/searchable-models.js"></script>
<script src="/ui/media/js/multi-model-selector.js"></script><script>
const fixtureModels = ${JSON.stringify(models)};
modelsOptions = convertToLegacyModelOptions(fixtureModels); modelsDB = buildModelsDB(fixtureModels);
const stableDiffusionModelField = new ModelDropdown(document.getElementById("stable_diffusion_model"), "stable-diffusion", "None");
const vaeModelField = new ModelDropdown(document.getElementById("vae_model"), "vae", "None");
const textEncoderModelField = new MultiModelSelector(document.getElementById("text_encoder_model"), "text-encoder", "Text Encoder", .5, .02, false);
const PLUGINS = {TASK_BUILD:[],TASK_CREATE:[]};
window.SD = {VideoTask:class VideoTask{constructor(body){this.body=body}}};
window.loadRequiredPluginHTML = () => ${JSON.stringify(source("ui/plugins/ui/video_plugin/native-video.plugin.html"))};
window.fixtureRequest = () => {
 const event={reqBody:{use_stable_diffusion_model:stableDiffusionModelField.value,use_vae_model:vaeModelField.value,use_text_encoder_model:textEncoderModelField.modelNames,sampler_name:"euler_a"}};
 PLUGINS.TASK_BUILD.forEach(h=>h(event));PLUGINS.TASK_CREATE.forEach(h=>h(event));
 return {video:event.instance instanceof SD.VideoTask,body:event.reqBody};
};
</script><script src="/ui/plugins/ui/video_plugin/native-video.plugin.js"></script></body></html>`

const assets = new Set(["ui/media/js/utils.js", "ui/media/js/searchable-models.js", "ui/media/js/multi-model-selector.js",
    "ui/media/css/searchable-models.css", "ui/plugins/ui/video_plugin/native-video.plugin.js"])
const server = http.createServer((req, res) => {
    const file = req.url.slice(1)
    if (req.method !== "GET") {res.writeHead(405); return res.end()}
    if (req.url === "/") {res.setHeader("Content-Type", "text/html"); return res.end(fixture)}
    if (!assets.has(file)) {res.writeHead(404); return res.end()}
    res.setHeader("Content-Type", file.endsWith(".css") ? "text/css" : "text/javascript")
    res.end(source(file))
})

async function run() {
    await new Promise(resolve => server.listen(process.env.VIDEO_TEST_PORT || 0, "127.0.0.1", resolve))
    const url = `http://127.0.0.1:${server.address().port}`
    if (process.argv.includes("--serve")) {console.log(url); return}
    const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright")
    let browser
    try {
        browser = await chromium.launch({headless:true, ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})})
        const page = await browser.newPage({viewport:{width:1200,height:900}}), errors = []
        page.on("pageerror", e => {errors.push(e.message); console.error("Browser exception:", e.message)})
        await page.goto(url)
        await page.waitForSelector("#stable_diffusion_model")
        assert.equal(await page.locator("#native-video-model").count(), 0)
        assert.equal(await page.locator("#native-video-enabled").count(), 0)
        assert.equal(await page.locator("#sdkit3-native-video-panel").isVisible(), false)
        async function choose(id, model) {
            await page.locator(`#${id}`).click()
            await page.locator(`#${id}-model-list .model-file[data-path=${JSON.stringify(model)}]`).click()
        }
        await choose("stable_diffusion_model", "video/mochi")
        assert.equal(await page.locator("#vae_model").getAttribute("data-path"), "mochi/mochi_vae_fp8_e4m3fn")
        assert.deepEqual(await page.evaluate(() => textEncoderModelField.modelNames), ["t5xxl_fp8_e4m3fn"])
        assert.equal(await page.evaluate(() => fixtureRequest().video), true)
        await page.locator("#sdkit3-native-video-panel > h4").click()
        await page.locator("#native-video-frames").fill("49")
        assert.equal(await page.evaluate(() => fixtureRequest().body.video_frames), 49)
        await choose("vae_model", "mochi/mochi_vae")
        await page.evaluate(() => document.dispatchEvent(new Event("refreshModels")))
        await page.waitForTimeout(50)
        assert.equal(await page.locator("#vae_model").getAttribute("data-path"), "mochi/mochi_vae")
        await choose("stable_diffusion_model", "video/ltx-2.3-dev")
        assert.equal(await page.locator("#native-video-ltx23-resources").isVisible(), true)
        assert.equal(await page.locator("#vae_model").getAttribute("data-path"), "ltx-2.3-22b-dev_video_vae")
        const ltx = await page.evaluate(() => fixtureRequest())
        assert.deepEqual(ltx.body.use_text_encoder_model, ["gemma-3-12b-it-Q4_K_M"])
        assert.equal(ltx.body.audio_vae_model, "ltx-2.3-22b-dev_audio_vae")
        assert.equal(ltx.body.embeddings_connectors_model, "ltx-2.3-22b-dev_embeddings_connectors")
        assert.equal(ltx.body.scheduler_name, "ltx2")
        await page.evaluate(() => document.dispatchEvent(new Event("refreshModels")))
        await page.waitForTimeout(50)
        assert.equal(await page.locator("#native-video-audio-vae").getAttribute("data-path"), "ltx-2.3-22b-dev_audio_vae")
        await choose("stable_diffusion_model", "external/wan")
        assert.equal(await page.locator("#vae_model").getAttribute("data-path"), "wan/wan_2.1_vae")
        await choose("stable_diffusion_model", "video/ltx-distilled")
        assert.equal(await page.locator("#vae_model").getAttribute("data-path"), "")
        assert.equal(await page.evaluate(() => fixtureRequest().body.num_inference_steps), 8)
        await choose("stable_diffusion_model", "image")
        assert.equal(await page.evaluate(() => fixtureRequest().video), false)
        assert.equal(await page.locator("#native-video-ltx23-resources").isVisible(), false)
        assert.equal(await page.locator("#vae_model").getAttribute("data-path"), "image-vae")
        assert.deepEqual(await page.evaluate(() => textEncoderModelField.modelNames), ["image-encoder"])
        await choose("stable_diffusion_model", "video/mochi")
        assert.equal(await page.locator("#vae_model").getAttribute("data-path"), "mochi/mochi_vae")
        await choose("stable_diffusion_model", "video/ltx-2.3-dev")
        await page.locator("#native-video-frames").scrollIntoViewIfNeeded()
        await page.screenshot({path:"/tmp/legacy-video-desktop.png"})
        await page.setViewportSize({width:390,height:844})
        await page.locator("#native-video-frames").scrollIntoViewIfNeeded()
        const box = await page.locator("#native-video-frames").boundingBox()
        assert.ok(box.x >= 0 && box.x + box.width <= 390)
        await page.screenshot({path:"/tmp/legacy-video-mobile.png"})
        assert.deepEqual(errors, [])
        console.log("PASS: real dropdowns, LTX-2.3 four-resource autoload and payload, catalog merge, overrides, refresh, image/video round trip, routing, frame count, desktop/mobile; no backend")
    } finally {if (browser) await browser.close(); server.close()}
}
run().catch(error => {console.error(error); server.close(); process.exitCode = 1})
