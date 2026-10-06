// All generation submissions are intercepted; no image generation runs.
const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
    const browser = await chromium.launch({headless: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    try {
        const page = await browser.newPage({viewport: {width: 1280, height: 960}});
        const errors = [], requests = [];
        page.on("pageerror", error => errors.push(error.message));
        await page.route("**/*", async route => {
            const request = route.request(), url = new URL(request.url());
            if (request.resourceType() === "image" || /gallery|thumbnail/i.test(url.pathname)) return route.abort();
            if (request.method() === "POST") {
                assert.equal(url.pathname, "/render");
                requests.push(request.postDataJSON());
                return route.fulfill({status: 409, json: {detail: "Test intercepted: no generation started"}});
            }
            const models = {
                "/get/model": {models: [{model: "checkpoints/sd15", name: "SD 1.5"}]},
                "/get/lora": {models: [{model: "characters/robot-cat", name: "Robot cat"}]},
                "/get/models": {models: []},
                "/get/vae": {models: [{model: "Anima/qwen_image_vae", tags: ["vae"]}]},
                "/get/other": {models: [{model: "qwen_3_06b_base", tags: ["text-encoder"]}]},
            };
            if (models[url.pathname]) return route.fulfill({json: models[url.pathname]});
            return route.continue();
        });
        await page.goto(process.env.GENERATION_TEST_URL || "http://100.106.255.71:10000/cpp-ui");
        await page.waitForSelector("#stable_diffusion_model-model-list", {state: "attached"});
        await page.waitForSelector("#vae_model-model-list", {state: "attached", timeout: 5000});
        await page.fill("#vae_model", "qwen_image");
        await page.locator('#vae_model-model-list [data-path="Anima/qwen_image_vae"]').click();
        await page.fill("#text_encoder_model", "qwen_3");
        await page.locator('#text_encoder_model-model-list [data-path="qwen_3_06b_base"]').click();
        assert.equal(await page.title(), "Main · Easy Diffusion C++");
        for (const id of ["width", "height"]) {
            assert.equal(await page.getAttribute(`#${id}`, "type"), "range");
            assert.equal(await page.getAttribute(`#${id}`, "min"), "64");
            assert.equal(await page.getAttribute(`#${id}`, "step"), "64");
        }
        assert.equal(await page.locator("#makeImage").innerText(), "Generate");
        await page.evaluate(() => scrollTo(0, 0));
        assert.ok(await page.locator("#makeImage").evaluate(el => el.getBoundingClientRect().bottom <= innerHeight),
            "Generate must be visible in the initial desktop viewport");
        await page.fill("#stable_diffusion_model", "sd15");
        await page.locator('#stable_diffusion_model-model-list [data-path="checkpoints/sd15"]').click();
        await page.fill("#cpp-lora-0", "robot-cat");
        await page.locator('#cpp-lora-0-model-list [data-path="characters/robot-cat"]').click();
        await page.fill(".cpp-lora-entry .model_weight", "0.75");
        await page.fill("#prompt", "anthropomorphic robotic cat, white armor, blue eyes");
        await page.click("#makeImage");
        await page.waitForFunction(() => document.getElementById("generation-queue-status").textContent.includes("Test intercepted"));
        assert.equal(requests.length, 1);
        assert.equal(requests[0].use_stable_diffusion_model, "checkpoints/sd15");
        assert.equal(requests[0].use_lora_model, "characters/robot-cat");
        assert.equal(requests[0].lora_alpha, .75);
        assert.equal(requests[0].use_vae_model, "Anima/qwen_image_vae");
        assert.equal(requests[0].use_text_encoder_model, "qwen_3_06b_base");
        await page.reload();
        await page.waitForFunction(() => document.querySelector("#vae_model")?.dataset.path === "Anima/qwen_image_vae"
            && document.querySelector("#text_encoder_model")?.dataset.path === "qwen_3_06b_base");
        assert.deepEqual(errors, []);
        await page.evaluate(() => scrollTo(0, 0));
        await page.screenshot({path: process.env.GENERATION_TEST_SCREENSHOT || "/tmp/cpp-generation-desktop.png"});
        await page.setViewportSize({width: 390, height: 844});
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Mobile horizontal overflow");
        await page.screenshot({path: "/tmp/cpp-generation-mobile.png"});
        console.log("PASS: visible Generate, shared searchable checkpoint/LoRA controls, request payload, responsive layout, zero browser exceptions; no generation");
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
