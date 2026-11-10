const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const modelPath = "/mnt/models/gguf/text/prompt-model.gguf";
const settingsPayload = {
    model_path: modelPath,
    device: "cuda",
    gpu_layers: 999,
    context_length: 4096,
    max_tokens: 512,
    temperature: 0.7,
    top_p: 0.95,
    top_k: 40,
    repeat_penalty: 1.05,
    valid: true,
    error: null,
    cuda_available: true,
    models_root: "/mnt/models",
    models: [{path: modelPath, relative_path: "gguf/text/prompt-model.gguf", name: "prompt-model.gguf", size: 1024 ** 3}],
    backend: {ready: true, detail: "ready"},
    loaded: false,
};

async function assertVisibleLocalLlmControlsFit(page, rootSelector, label) {
    const overflowingElements = await page.evaluate(selector => {
        const root = document.querySelector(selector);
        if (!root) return [{id: `missing ${selector}`, left: null, right: null}];
        return [root, ...root.querySelectorAll("input, select, textarea, button")]
            .filter(element => {
                const bounds = element.getBoundingClientRect();
                return bounds.width > 0 && (bounds.left < 0 || bounds.right > innerWidth);
            })
            .map(element => ({
                id: element.id || element.tagName.toLowerCase(),
                left: element.getBoundingClientRect().left,
                right: element.getBoundingClientRect().right,
            }));
    }, rootSelector);
    assert.deepEqual(overflowingElements, [], `${label} overflowed the mobile viewport`);
}

(async () => {
    const browser = await chromium.launch({
        headless: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
    });
    try {
        const page = await browser.newPage({viewport: {width: 1280, height: 960}});
        const errors = [], processRequests = [], settingsRequests = [];
        page.on("pageerror", error => errors.push(error.message));
        await page.route("**/local-llm/**", async route => {
            const request = route.request();
            const path = new URL(request.url()).pathname;
            if (path === "/local-llm/health") {
                return route.fulfill({json: {ready: true, detail: "ready", model_path: modelPath, device: "cuda", gpu_layers: 999}});
            }
            if (path === "/local-llm/process") {
                processRequests.push(request.postDataJSON());
                return route.fulfill({json: {processed_prompt: "cinematic fox portrait, rim lighting", model_path: modelPath, device: "cuda"}});
            }
            if (path === "/local-llm/settings" && request.method() === "POST") {
                settingsRequests.push(request.postDataJSON());
                return route.fulfill({json: {...settingsPayload, ...request.postDataJSON(), valid: true}});
            }
            if (path === "/local-llm/settings") return route.fulfill({json: settingsPayload});
            return route.abort();
        });

        const base = process.env.LOCAL_LLM_TEST_URL || "http://127.0.0.1:10000";
        await page.goto(`${base}/cpp-ui`);
        await page.waitForSelector("#local-llm-prompt-panel");
        await page.fill("#prompt", "fox portrait");
        await page.click("#local-llm-prompt-panel > summary");
        await page.click("#local-llm-load-prompt");
        assert.equal(await page.inputValue("#local-llm-source-prompt"), "fox portrait");
        await page.fill("#local-llm-instruction", "Add cinematic lighting.");
        await page.click("#local-llm-process");
        await page.waitForFunction(() => document.getElementById("local-llm-output").value.includes("cinematic"));
        assert.deepEqual(processRequests, [{prompt: "fox portrait", instruction: "Add cinematic lighting."}]);
        await page.click("#local-llm-apply");
        assert.equal(await page.inputValue("#prompt"), "cinematic fox portrait, rim lighting");

        await page.goto(`${base}/cpp-ui/settings`);
        await page.waitForFunction(() => document.getElementById("local-llm-model-path")?.value === modelPath);
        await page.selectOption("#local-llm-device", "cpu");
        assert.equal(await page.inputValue("#local-llm-gpu-layers"), "0");
        assert.equal(await page.isDisabled("#local-llm-gpu-layers"), true);
        await page.click("#local-llm-save-settings");
        await page.waitForFunction(() => document.getElementById("local-llm-settings-status").textContent.includes("saved"));
        assert.equal(settingsRequests.length, 1);
        assert.equal(settingsRequests[0].device, "cpu");
        assert.equal(settingsRequests[0].gpu_layers, 0);
        assert.deepEqual(errors, []);

        await page.setViewportSize({width: 390, height: 844});
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Local LLM controls cause mobile overflow");
        await page.screenshot({path: process.env.LOCAL_LLM_TEST_SCREENSHOT || "/tmp/local-llm-settings-mobile.png"});

        await page.setViewportSize({width: 1280, height: 960});
        await page.goto(`${base}/legacy`);
        await page.waitForSelector("#local-llm-prompt-panel");
        await page.click("#local-llm-prompt-panel > .collapsible");
        await page.fill("#prompt", "fox portrait");
        await page.click("#local-llm-load-prompt");
        assert.equal(await page.inputValue("#local-llm-source-prompt"), "fox portrait");
        await page.fill("#local-llm-instruction", "Add cinematic lighting.");
        await page.click("#local-llm-process");
        await page.waitForFunction(() => document.getElementById("local-llm-output").value.includes("cinematic"));
        assert.equal(processRequests.length, 2);
        await page.click("#local-llm-apply");
        assert.equal(await page.inputValue("#prompt"), "cinematic fox portrait, rim lighting");

        await page.setViewportSize({width: 390, height: 844});
        await assertVisibleLocalLlmControlsFit(page, "#local-llm-prompt-panel", "Legacy Local LLM prompt controls");

        await page.setViewportSize({width: 1280, height: 960});
        await page.click("#tab-settings");
        await page.waitForFunction(() => document.getElementById("local-llm-model-path")?.value === modelPath);
        await page.selectOption("#local-llm-device", "cpu");
        assert.equal(await page.inputValue("#local-llm-gpu-layers"), "0");
        assert.equal(await page.isDisabled("#local-llm-gpu-layers"), true);
        await page.click("#local-llm-save-settings");
        await page.waitForFunction(() => document.getElementById("local-llm-settings-status").textContent.includes("saved"));
        assert.equal(settingsRequests.length, 2);

        await page.setViewportSize({width: 390, height: 844});
        await assertVisibleLocalLlmControlsFit(page, "#local-llm-settings", "Legacy Local LLM settings controls");
        await page.screenshot({path: process.env.LOCAL_LLM_LEGACY_TEST_SCREENSHOT || "/tmp/local-llm-legacy-settings-mobile.png"});
        assert.deepEqual(errors, []);
        console.log("PASS: C++ and legacy prompt processing, prompt replacement, persisted CPU/GPU settings, and responsive layout");
    } finally {
        await browser.close();
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
