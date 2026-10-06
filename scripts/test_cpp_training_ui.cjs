// All training API calls are intercepted; this test never launches training.
const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
    const browser = await chromium.launch({headless: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    try {
        const page = await browser.newPage();
        const errors = [], submitted = [];
        page.on("pageerror", error => errors.push(error.message));
        await page.route(/gallery|thumbnail/i, route => route.abort());
        await page.route("**/*", route => route.request().resourceType() === "image" ? route.abort() : route.fallback());
        await page.route("**/training/**", async route => {
            const path = new URL(route.request().url()).pathname;
            if (route.request().method() === "POST") {
                assert.equal(path, "/training/jobs");
                submitted.push(route.request().postDataJSON());
                return route.fulfill({status: 409, json: {detail: "Test intercepted: no training started"}});
            }
            const fixtures = {
                "/training/datasets": {datasets: ["cat-sd15"]},
                "/training/models": {models: [{name: "SD1.5", path: "/models/sd15.safetensors"}]},
                "/training/assets": {text_encoders: [], vaes: []},
                "/training/readiness": {detail: "Test ready", native_cpp: {ready: true}, python_runtime: {ready: true}},
                "/training/spritegpt": {running: false, detail: "Stopped"},
                "/training/jobs": {jobs: [{id: "safe-history", kind: "lora", status: "completed"}]},
                "/training/jobs/safe-history": {id: "safe-history", kind: "lora", status: "completed",
                    spec: {native_cpp: true}, progress: {step: 6, total: 6},
                    log: Array.from({length: 180}, (_, i) => ({message: `Step ${i} ${"x".repeat(600)}`}))},
            };
            assert.ok(fixtures[path], `Unexpected API call: ${path}`);
            return route.fulfill({json: fixtures[path]});
        });
        await page.goto(process.env.TRAINING_TEST_URL || "http://127.0.0.1:10000/cpp-ui/training");
        await page.waitForFunction(() => !document.getElementById("training-start").disabled);
        assert.equal(await page.title(), "Training · Easy Diffusion C++");
        const layout = await page.evaluate(() => {
            const log = document.getElementById("training-log");
            return {viewport: innerWidth, width: document.documentElement.scrollWidth,
                logHeight: log.clientHeight, contentHeight: log.scrollHeight};
        });
        assert.ok(layout.width <= layout.viewport, `Training page overflows horizontally: ${JSON.stringify(layout)}`);
        assert.ok(layout.logHeight <= 460 && layout.contentHeight > layout.logHeight,
            `Training log must scroll inside a bounded panel: ${JSON.stringify(layout)}`);
        assert.equal(await page.isVisible("#training-anima-settings"), false);
        await page.selectOption("#training-architecture", "anima");
        assert.equal(await page.isVisible("#training-anima-settings"), true);
        await page.selectOption("#training-architecture", "sd15");
        assert.equal(await page.isVisible("#training-anima-settings"), false);
        assert.equal(await page.inputValue("#training-rate"), "0.00005");
        assert.equal(await page.inputValue("#training-text-encoder-rate"), "0.000005");
        assert.equal(await page.inputValue("#training-steps"), "1200");
        assert.equal(submitted.length, 0);
        assert.equal(await page.inputValue("#training-alpha"), "");
        assert.equal(await page.isDisabled("#training-warmup"), true);
        await page.fill("#training-rank", "32");
        await page.fill("#training-alpha", "16");
        await page.selectOption("#training-scheduler", "cosine_with_restarts");
        assert.equal(await page.isDisabled("#training-warmup"), false);
        await page.fill("#training-warmup", "20");
        await page.fill("#training-cycles", "3");
        await page.click("#training-start");
        await page.waitForFunction(() => document.getElementById("training-error").textContent.includes("Test intercepted"));
        assert.equal(submitted[0].learning_rate, 5e-5);
        assert.equal(submitted[0].text_encoder_learning_rate, 5e-6);
        assert.equal(submitted[0].steps, 1200);
        assert.equal(submitted[0].network_alpha, 16);
        assert.equal(submitted[0].rank, 32);
        assert.equal(submitted[0].lr_scheduler, "cosine_with_restarts");
        assert.equal(submitted[0].lr_warmup_steps, 20);
        assert.equal(submitted[0].lr_scheduler_num_cycles, 3);
        await page.fill("#training-text-encoder-rate", "0");
        await page.click("#training-start");
        await page.waitForTimeout(100);
        assert.equal(submitted[1].text_encoder_learning_rate, 0);
        await page.fill("#training-text-encoder-rate", "0.000005");
        await page.selectOption("#training-architecture", "sdxl");
        assert.equal(await page.isDisabled("#training-text-encoder-rate"), true);
        await page.click("#training-start");
        await page.waitForTimeout(100);
        assert.equal(submitted[2].text_encoder_learning_rate, 0);
        await page.selectOption("#training-architecture", "sd15");
        assert.equal(await page.isDisabled("#training-text-encoder-rate"), false);
        assert.equal(await page.inputValue("#training-text-encoder-rate"), "0.000005");
        await page.fill("#training-text-encoder-rate", "-1");
        await page.click("#training-start");
        await page.waitForTimeout(100);
        assert.equal(submitted.length, 3);
        assert.match(await page.locator("#training-error").innerText(), /Learning rates/);
        await page.fill("#training-text-encoder-rate", "0.000005");
        await page.locator("#training-text-encoder-rate").dispatchEvent("change");
        await page.reload();
        await page.waitForFunction(() => !document.getElementById("training-start").disabled);
        assert.equal(await page.inputValue("#training-text-encoder-rate"), "0.000005");
        assert.equal(await page.inputValue("#training-steps"), "1200");
        assert.equal(submitted.length, 3, "reload must not submit a job");
        assert.equal(await page.inputValue("#training-alpha"), "16");
        assert.equal(await page.inputValue("#training-scheduler"), "cosine_with_restarts");
        assert.equal(await page.inputValue("#training-cycles"), "3");
        await page.fill("#training-alpha", "0");
        await page.click("#training-start");
        assert.match(await page.locator("#training-error").innerText(), /alpha/);
        assert.equal(submitted.length, 3);
        await page.fill("#training-alpha", "");
        await page.fill("#training-warmup", "1200");
        await page.click("#training-start");
        assert.match(await page.locator("#training-error").innerText(), /Scheduler/);
        assert.equal(submitted.length, 3);
        await page.selectOption("#training-scheduler", "constant");
        await page.click("#training-start");
        await page.waitForFunction(() => document.getElementById("training-error").textContent.includes("Test intercepted"));
        assert.equal(submitted[3].network_alpha, null);
        assert.equal(submitted[3].lr_warmup_steps, 0);
        assert.equal(submitted[3].lr_scheduler_num_cycles, 1);
        await page.selectOption("#training-backend", "python");
        assert.equal(await page.isDisabled("#training-batch"), false);
        await page.click("#training-start");
        await page.waitForTimeout(100);
        assert.equal(submitted.at(-1).backend, "python");
        await page.selectOption("#training-backend", "native");
        await page.fill("#training-epochs", "10");
        await page.locator("#training-epochs").dispatchEvent("change");
        await page.fill("#training-repeats", "150");
        await page.click("#training-start");
        await page.waitForTimeout(100);
        assert.equal(submitted.at(-1).backend, "native");
        assert.equal(submitted.at(-1).epochs, 10);
        assert.equal(submitted.at(-1).dataset_repeats, 150);
        assert.equal(await page.isDisabled("#training-steps"), true);
        await page.fill("#training-model", "SD1.5");
        await page.fill("#training-model", "sd15");
        await page.locator('#training-model-model-list [data-path="/models/sd15.safetensors"]').click();
        assert.equal(await page.locator("#training-model").getAttribute("data-path"), "/models/sd15.safetensors");
        assert.deepEqual(errors, []);
        await page.setViewportSize({width: 390, height: 844});
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
            "Training menu must fit a narrow viewport");
        await page.setViewportSize({width: 1280, height: 960});
        await page.locator("#training-rate").evaluate(element => element.scrollIntoView({block: "center"}));
        await page.screenshot({path: process.env.TRAINING_TEST_SCREENSHOT || "/tmp/sd15-text-encoder-ui-test.png"});
        console.log("PASS: defaults, separate rates, freeze, family switching, validation, persistence, no auto-start, no browser exceptions");
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
