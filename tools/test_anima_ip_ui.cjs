const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const root = process.env.IP_UI_SOURCE || path.resolve(__dirname, "..");
const origin = process.env.IP_UI_URL || "http://127.0.0.1:10000";
const output = process.env.IP_UI_OUTPUT || "/tmp/anima-ip-ui";
const read = file => fs.readFileSync(path.join(root, file), "utf8");
const scripts = "source/UI.cpp/Pages/src/Plugin/plugin_scripts/";
const models = [
    {model: "Anima/base", tags: ["stable-diffusion", "anima"]},
    {model: "Anima/character", tags: ["ip-adapter", "ip_adapter_anima", "ip_adapter_embedding_768"]},
    {model: "siglip2", tags: ["clip-vision", "siglip2_base_patch16_512", "clip_hidden_768", "clip_projection_0"]},
    {model: "wrong-clip", tags: ["clip-vision", "clip_hidden_768", "clip_projection_768"]},
];
const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZQmcAAAAASUVORK5CYII=", "base64");
(async () => {
    const browser = await chromium.launch({headless: true, chromiumSandbox: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    fs.mkdirSync(output, {recursive: true});
    try {
        const context = await browser.newContext({viewport: {width: 1280, height: 960}});
        await context.addInitScript(() => localStorage.setItem("easy-diffusion-cpp-checkpoint-v1", "Anima/base"));
        let request;
        const errors = [];
        await context.route("**/*", async route => {
            const req = route.request(), url = new URL(req.url());
            const json = value => route.fulfill({contentType: "application/json", body: JSON.stringify(value)});
            if (req.method() === "POST" && url.pathname === "/render") {
                request = req.postDataJSON();
                return json({task: 1, stream: "/image/stream/test-ip"});
            }
            assert.equal(req.method(), "GET", "Test must not mutate the live backend");
            if (url.pathname === "/image/stream/test-ip") return json({status: "succeeded", output: []});
            if (url.pathname === "/kiosk") return json({enabled: false, allowed_models: []});
            if (url.pathname === "/get/models") return json({models});
            if (url.pathname === "/get/model") return json({models: [models[0]]});
            if (url.pathname === "/get/lora") return json({models: []});
            if (url.pathname.endsWith("/generate.js")) return route.fulfill({contentType: "text/javascript", body: read(scripts + "generate.js")});
            if (url.pathname.endsWith("/ip-adapter.plugin.html")) return route.fulfill({contentType: "text/html", body: read("ui/plugins/ui/controlnet_plugin/ip-adapter.plugin.html")});
            return route.continue();
        });
        const page = await context.newPage();
        page.on("pageerror", error => errors.push(error.message));
        await page.goto(origin + "/cpp-ui");
        await page.waitForFunction(() => window.CppGeneratePage === true);
        await page.addScriptTag({content: read("ui/media/js/ip-adapter-compatibility.js")});
        await page.addScriptTag({content: read(scripts + "ip_adapter.js")});
        await page.waitForSelector("#cpp-ip-adapter-panel #ip-adapter-model");
        await page.waitForFunction(() => document.getElementById("ip-adapter-clip").dataset.path === "siglip2");
        await page.locator("#ip-adapter-image-input").setInputFiles({name: "reference.png", mimeType: "image/png", buffer: png});
        await page.waitForFunction(() => document.getElementById("ip-adapter-enabled").checked);
        await page.fill("#ip-adapter-strength", "0.35");
        await page.fill("#prompt", "a red ceramic teapot on a wooden table");
        await page.click("#makeImage");
        await page.waitForFunction(() => document.getElementById("generation-queue-status").textContent === "Generation complete.");
        assert.equal(request.ip_adapter_model, "Anima/character");
        assert.equal(request.ip_adapter_clip_vision, "siglip2");
        assert.equal(request.ip_adapter_strength, 0.35);
        assert.ok(request.ip_adapter_image.startsWith("data:image/png;base64,"));
        await page.fill("#ip-adapter-strength", "0");
        assert.equal(await page.evaluate(() => CppIPAdapter.requestOptions().ip_adapter_strength), 0);
        await page.fill("#ip-adapter-start", "80");
        await page.fill("#ip-adapter-end", "20");
        assert.match(await page.evaluate(() => {try {CppIPAdapter.requestOptions();} catch (e) {return e.message;}}), /increasing step range/);
        await page.fill("#ip-adapter-start", "0");
        await page.fill("#ip-adapter-end", "100");
        await page.locator("#cpp-ip-adapter-panel").scrollIntoViewIfNeeded();
        await page.locator("#cpp-ip-adapter-panel").screenshot({path: path.join(output, "anima-ip-desktop.png")});
        await page.setViewportSize({width: 390, height: 844});
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        await page.locator("#cpp-ip-adapter-panel").screenshot({path: path.join(output, "anima-ip-mobile.png")});
        await page.click("#ip-adapter-image-clear");
        assert.deepEqual(await page.evaluate(() => CppIPAdapter.requestOptions()), {});
        assert.equal(await page.locator("#ip-adapter-image-wrapper").isVisible(), false);

        const legacy = await context.newPage();
        legacy.on("pageerror", error => errors.push(error.message));
        await legacy.goto(origin + "/cpp-ui");
        await legacy.waitForFunction(() => window.CppGeneratePage === true);
        const html = read("ui/plugins/ui/controlnet_plugin/ip-adapter.plugin.html");
        await legacy.evaluate(({models, html}) => {
            document.body.innerHTML = '<div class="app-shell"><input id="stable_diffusion_model" data-path="Anima/base"><div id="editor-settings"></div></div>';
            window.PLUGINS = {TASK_CREATE: [], IMAGE_INFO_BUTTONS: []};
            modelsDB = Object.fromEntries(["stable-diffusion", "ip-adapter", "clip-vision"].map(type =>
                [type, Object.fromEntries(models.filter(m => m.tags.includes(type)).map(m => [m.model, m]))]));
            modelsOptions = Object.fromEntries(["stable-diffusion", "ip-adapter", "clip-vision"].map(type =>
                [type, buildTree(models.filter(m => m.tags.includes(type)))]));
            window.loadRequiredPluginHTML = () => html;
            window.testIPWarnings = [];
            window.showToast = message => window.testIPWarnings.push(message);
            const saved = JSON.parse(localStorage.getItem("easy-diffusion-native-ip-adapter-v1") || "{}");
            localStorage.setItem("easy-diffusion-native-ip-adapter-v1", JSON.stringify({...saved, enabled: true}));
        }, {models, html});
        await legacy.addScriptTag({content: read("ui/media/js/ip-adapter-compatibility.js")});
        await legacy.addScriptTag({content: read("ui/plugins/ui/controlnet_plugin/ip-adapter.plugin.js")});
        await legacy.waitForFunction(() => document.getElementById("ip-adapter-clip").dataset.path === "siglip2");
        assert.equal(await legacy.locator("#ip-adapter-enabled").isChecked(), false, "Reload must not enable an adapter without its reference image");
        await legacy.locator("#ip-adapter-enabled").check();
        const incomplete = await legacy.evaluate(() => {const event = {reqBody: {}}; PLUGINS.TASK_CREATE[0](event); return event.reqBody;});
        assert.equal(incomplete.ip_adapter_model, "Anima/character", "An incomplete requested adapter must reach the backend validation, not silently disappear");
        assert.equal(incomplete.ip_adapter_image, null);
        await legacy.locator("#ip-adapter-image-input").setInputFiles({name: "reference.png", mimeType: "image/png", buffer: png});
        await legacy.waitForFunction(() => document.getElementById("ip-adapter-enabled").checked);
        const payload = await legacy.evaluate(() => {const event = {reqBody: {}}; PLUGINS.TASK_CREATE[0](event); return event.reqBody;});
        assert.equal(payload.ip_adapter_model, "Anima/character");
        assert.equal(payload.ip_adapter_clip_vision, "siglip2");
        await legacy.click("#ip-adapter-image-clear");
        assert.deepEqual(await legacy.evaluate(() => {const event = {reqBody: {}}; PLUGINS.TASK_CREATE[0](event); return event.reqBody;}), {});
        assert.deepEqual(errors, []);

        const kioskContext = await browser.newContext();
        await kioskContext.route("**/*", async route => {
            assert.equal(route.request().method(), "GET");
            if (new URL(route.request().url()).pathname === "/kiosk") return route.fulfill({contentType: "application/json", body: JSON.stringify({enabled: true, allowed_models: []})});
            return route.continue();
        });
        const kiosk = await kioskContext.newPage();
        await kiosk.goto(origin + "/cpp-ui");
        await kiosk.addScriptTag({content: read(scripts + "ip_adapter.js")});
        await kiosk.evaluate(() => CppKiosk.ready);
        assert.equal(await kiosk.locator("#cpp-ip-adapter-panel").count(), 0);
        assert.deepEqual(await kiosk.evaluate(() => CppIPAdapter.requestOptions()), {});
        console.log("PASS: C++ request payload, legacy payload, filtering, strength zero, validation, clear/disable, desktop/mobile and kiosk isolation; no live backend mutations.");
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
