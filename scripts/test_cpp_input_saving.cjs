const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
    const browser = await chromium.launch({headless: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    try {
        const page = await browser.newPage();
        await page.route("**/*", route => {
            assert.equal(route.request().method(), "GET");
            const fixtures = {
                "/get/model": {models: [{model: "test-checkpoint"}]},
                "/get/lora": {models: [{model: "test-lora"}]},
                "/get/vae": {models: [{model: "test-vae", tags: ["vae"]}]},
                "/get/other": {models: [{model: "test-encoder", tags: ["text-encoder"]}]},
                "/get/models": {models: []},
            };
            const fixture = fixtures[new URL(route.request().url()).pathname];
            return fixture ? route.fulfill({json: fixture}) : route.continue();
        });
        const origin = process.env.INPUT_SAVING_TEST_URL || "http://100.106.255.71:10000";
        await page.goto(origin + "/cpp-ui/settings");
        await page.evaluate(() => localStorage.setItem("user_settings_v2", JSON.stringify([
            {key: "auto_save_settings", value: true, ignore: false},
            {key: "prompt", value: "remembered prompt", ignore: false},
            {key: "width", value: "768", ignore: false},
            {key: "num_inference_steps", value: "15", ignore: false},
            {key: "stable_diffusion_model", value: "test-checkpoint", ignore: false},
            {key: "vae_model", value: "test-vae", ignore: false},
            {key: "text_encoder_model", value: "test-encoder", ignore: false},
            {key: "theme", value: "unrelated-legacy-setting", ignore: false},
            {key: "lora_model", value: JSON.stringify({modelNames: ["test-lora"], modelWeights: [.7]}), ignore: false},
        ])));
        await page.goto(origin + "/cpp-ui");
        await page.waitForFunction(() => document.querySelector("#prompt").value === "remembered prompt", null, {timeout: 5000});
        assert.equal(await page.inputValue("#width"), "768");
        assert.equal(await page.inputValue("#steps"), "15");
        assert.equal(await page.getAttribute("#vae_model", "data-path"), "test-vae");
        await page.waitForFunction(() => document.querySelector("#cpp-lora-0")?.dataset.path === "test-lora");
        await page.fill("#prompt", "updated prompt");
        await page.reload();
        await page.waitForFunction(() => document.querySelector("#prompt").value === "updated prompt");
        await page.goto(origin + "/cpp-ui/settings");
        await page.click("#cpp-configure-input-saving");
        await page.screenshot({path: "/tmp/cpp-input-saving-desktop.png"});
        await page.setViewportSize({width: 390, height: 844});
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        await page.screenshot({path: "/tmp/cpp-input-saving-mobile.png"});
        await page.locator('[data-input-setting="width"]').uncheck();
        await page.click("#cpp-close-input-saving");
        await page.goto(origin + "/cpp-ui");
        await page.waitForFunction(() => document.querySelector("#prompt").value === "updated prompt");
        assert.equal(await page.inputValue("#width"), "512", "Ignored width must not restore");
        await page.goto(origin + "/cpp-ui/settings");
        await page.locator("#auto_save_settings").uncheck();
        await page.goto(origin + "/cpp-ui");
        await page.waitForSelector("#text_encoder_model-model-list", {state: "attached"});
        assert.equal(await page.inputValue("#prompt"), "");
        assert.equal(await page.getAttribute("#stable_diffusion_model", "data-path"), "");
        assert.equal(await page.getAttribute("#vae_model", "data-path"), "");
        assert.equal(await page.getAttribute("#cpp-lora-0", "data-path"), "");
        await page.fill("#prompt", "do not persist");
        const saved = await page.evaluate(() => JSON.parse(localStorage.getItem("user_settings_v2")));
        assert.equal(saved.find(entry => entry.key === "prompt").value, "updated prompt");
        assert.equal(saved.find(entry => entry.key === "theme").value, "unrelated-legacy-setting");
        await page.evaluate(() => {
            const entries = JSON.parse(localStorage.getItem("user_settings_v2"));
            entries.find(entry => entry.key === "auto_save_settings").value = true;
            entries.push({key: "controlnet_mode", value: "standard", ignore: false});
            localStorage.setItem("user_settings_v2", JSON.stringify(entries));
        });
        await page.goto(origin + "/legacy");
        await page.waitForFunction(() => document.querySelector("#prompt")?.value === "updated prompt");
        await page.fill("#prompt", "saved from legacy");
        await page.goto(origin + "/cpp-ui");
        await page.waitForFunction(() => document.querySelector("#prompt")?.value === "saved from legacy");
        assert.equal(await page.inputValue("#controlnet_mode"), "standard", "Legacy saving must preserve modern-only settings");
        console.log("PASS: legacy input import, ordinary/model/LoRA restoration, autosave, field opt-out, global disable, preservation of unrelated preferences");
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
