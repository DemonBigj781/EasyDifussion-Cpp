// Policy/data fixtures are intercepted. No real settings, jobs or images are modified.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {execFileSync} = require("node:child_process");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
    const browser = await chromium.launch({headless: true, chromiumSandbox: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    const output = process.env.KIOSK_TEST_OUTPUT || "/tmp/cpp-kiosk-tests";
    fs.mkdirSync(output, {recursive: true});
    try {
        const context = await browser.newContext({viewport: {width: 1280, height: 960}});
        const blank = await context.newPage();
        const jpeg = await blank.evaluate(() => {
            const canvas = document.createElement("canvas"); canvas.width = canvas.height = 32;
            canvas.getContext("2d").fillRect(0, 0, 32, 32); return canvas.toDataURL("image/jpeg");
        });
        await blank.close();
        const bases = ["1.5/sd-v1-5", "sdxl/sd_xl_base_1.0", "Anima/anima-base-v1.0"];
        let enabled = false, galleryError = false, wrongSource = false, wrongRating = false, policyError = false;
        const errors = [], renders = [], galleryCalls = [], ratingCalls = [];
        const policy = () => ({enabled, allowed_models: bases.map(model => ({model, label: model})),
            gallery_source: enabled ? "destockd" : "local", perchance_content_filter: enabled ? "g" : "none"});
        await context.route("**/*", async route => {
            const request = route.request(), url = new URL(request.url());
            if (process.env.KIOSK_TEST_RENDERER && request.method() === "GET" && url.pathname.startsWith("/cpp-ui")) {
                const stage = process.env.KIOSK_TEST_STAGE;
                if (url.pathname.startsWith("/cpp-ui/scripts/")) {
                    const file = path.join(stage, "source/UI.cpp/Pages/src/Plugin/plugin_scripts", path.basename(url.pathname));
                    if (fs.existsSync(file)) return route.fulfill({contentType: "application/javascript", body: fs.readFileSync(file)});
                } else if (url.pathname === "/cpp-ui/assets/ui.css") {
                    return route.fulfill({contentType: "text/css", body: fs.readFileSync(path.join(stage, "source/UI.cpp/Pages/assets/ui.css"))});
                } else if (!url.pathname.includes("/assets/") && !url.pathname.includes("/api/")) {
                    const page = url.pathname.slice("/cpp-ui".length) || "/";
                    return route.fulfill({contentType: "text/html", body: execFileSync(process.env.KIOSK_TEST_RENDERER, [page])});
                }
            }
            if (url.pathname === "/kiosk") {
                if (request.method() === "POST") {
                    enabled = request.postDataJSON().enabled;
                }
                return policyError ? route.fulfill({status: 503, json: {detail: "Test policy failure"}}) : route.fulfill({json: policy()});
            }
            if (url.pathname === "/get/app_config") return route.fulfill({json: {backend_config: {platform: "cuda"}}});
            if (url.pathname === "/get/model" || url.pathname === "/get/models") {
                const names = enabled ? bases : [...bases, "custom/fine-tune"];
                return route.fulfill({json: {models: names.map(model => ({model, name: model, tags: ["stable-diffusion"]}))}});
            }
            if (url.pathname === "/get/lora") {
                assert.equal(enabled, false, "Kiosk must not even request the LoRA catalog");
                return route.fulfill({json: {models: [{model: "private/test-lora", name: "Test LoRA"}]}});
            }
            if (url.pathname === "/gallery/settings") {
                assert.equal(enabled, false, "Kiosk must not load local-directory settings");
                return route.fulfill({json: {gallery_directory: "/test-only", exists: true}});
            }
            if (url.pathname === "/gallery/images") {
                galleryCalls.push(enabled);
                if (galleryError) return route.fulfill({status: 502, json: {detail: "Destockd unavailable; no local fallback"}});
                const source = enabled && !wrongSource ? "destockd" : "local";
                const image = source === "destockd" ? "https://destockd.com/keyframes/Test/shot_001.jpg" : "/gallery/file/test.jpg";
                return route.fulfill({json: {source, directory: source === "destockd" ? "https://destockd.com" : "/test-only",
                    exists: true, page: 1, total_pages: 1, total: 1, start_index: 1, end_index: 1,
                    images: [{id: "test", filename: "Test image", url: image, thumbnail_url: image, source_url: "https://destockd.com/#/shot/Test/shot_001"}]}});
            }
            if (url.pathname === "/perchance/gallery/list") {
                const data = request.postDataJSON(); ratingCalls.push(data);
                assert.equal(data.content_filter, enabled ? "g" : "none");
                return route.fulfill({json: {content_filter: wrongRating ? "none" : data.content_filter,
                    entries: [{preview_data_url: jpeg, prompt: "A forest"}], suppressed_count: 1}});
            }
            if (url.pathname === "/render") {
                const data = request.postDataJSON(); renders.push(data);
                if (enabled) { assert.ok(bases.includes(data.use_stable_diffusion_model)); assert.ok(!data.use_lora_model); }
                return route.fulfill({status: 409, json: {detail: "Kiosk test intercepted: no generation"}});
            }
            if (url.pathname === "/gallery/file/test.jpg" || url.origin === "https://destockd.com") {
                if (url.origin !== "https://destockd.com") assert.equal(enabled, false, "Local image requested in kiosk mode");
                return route.fulfill({contentType: "image/jpeg", body: Buffer.from(jpeg.split(",")[1], "base64")});
            }
            if (request.method() === "POST") throw new Error(`Unexpected POST: ${url.pathname}`);
            return route.continue();
        });
        const main = await context.newPage(), settings = await context.newPage();
        for (const page of [main, settings]) {
            page.setDefaultTimeout(10000);
            page.on("pageerror", error => { errors.push(error.message); console.error("Browser exception:", error.message); });
        }
        const base = process.env.KIOSK_TEST_URL || "http://100.106.255.71:10000";
        await main.goto(base + "/cpp-ui");
        await main.evaluate(() => {
            localStorage.setItem("easy-diffusion-cpp-lora-v1", JSON.stringify([{name: "private/test-lora", weight: .75}]));
            localStorage.setItem("easy-diffusion-cpp-checkpoint-v1", "custom/fine-tune");
        });
        await main.reload();
        await main.waitForSelector(".cpp-lora-entry");
        await settings.goto(base + "/cpp-ui/settings");
        await settings.locator("#kiosk-mode-enabled").check();
        await settings.click("#kiosk-mode-save");
        await main.waitForFunction(() => document.documentElement.dataset.kiosk === "on");
        await main.waitForFunction(() => document.querySelectorAll(".cpp-lora-entry").length === 0);
        assert.equal(await main.locator('nav a[href="/cpp-ui/training"]').isVisible(), false);
        await main.fill("#prompt", "a forest");
        await main.click("#makeImage");
        await main.waitForFunction(() => document.getElementById("generation-queue-status").textContent.includes("Kiosk test intercepted"));
        assert.equal(renders.length, 1);
        await main.screenshot({path: path.join(output, "kiosk-main.png")});
        await main.goto(base + "/cpp-ui/gallery/images");
        await main.waitForSelector(".cpp-gallery-card img");
        assert.equal(await main.locator("#gallery-directory-input").isVisible(), false);
        assert.equal(await main.locator("#gallery-delete-selected").isVisible(), false);
        assert.ok((await main.locator(".cpp-gallery-card img").getAttribute("src")).startsWith("https://destockd.com/"));
        await main.screenshot({path: path.join(output, "kiosk-gallery.png")});
        galleryError = true; await main.click("#gallery-refresh");
        await main.waitForFunction(() => document.getElementById("gallery-source").textContent.includes("Destockd unavailable"));
        assert.equal(await main.locator(".cpp-gallery-card").count(), 0);
        galleryError = false; wrongSource = true; await main.click("#gallery-refresh");
        await main.waitForFunction(() => document.getElementById("gallery-source").textContent.includes("source mismatch"));
        assert.equal(await main.locator(".cpp-gallery-card").count(), 0); wrongSource = false;
        await main.goto(base + "/cpp-ui/perchance/gallery");
        await main.click("#perchance-generator-gallery-list-button");
        await main.waitForSelector(".cpp-perchance-card img");
        assert.match(await main.locator("[data-perchance-gallery-status]").innerText(), /G-filtered/);
        wrongRating = true; await main.click("#perchance-generator-gallery-list-button");
        await main.waitForFunction(() => document.querySelector("[data-perchance-gallery-status]").textContent.includes("Unverified"));
        assert.equal(await main.locator(".cpp-perchance-card img").count(), 0); wrongRating = false;
        await settings.setViewportSize({width: 390, height: 844});
        await settings.locator("#kiosk-settings").scrollIntoViewIfNeeded();
        assert.ok(await settings.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        await settings.screenshot({path: path.join(output, "kiosk-settings-mobile.png")});
        await settings.locator("#kiosk-mode-enabled").uncheck(); await settings.click("#kiosk-mode-save");
        await settings.waitForFunction(() => document.documentElement.dataset.kiosk === "off");
        await main.waitForFunction(() => document.documentElement.dataset.kiosk === "off");
        await main.goto(base + "/cpp-ui"); await main.waitForSelector(".cpp-lora-entry");
        assert.ok((await main.evaluate(() => localStorage.getItem("easy-diffusion-cpp-lora-v1"))).includes("private/test-lora"));
        await main.fill("#prompt", "a forest"); await main.click("#makeImage");
        await main.waitForFunction(() => !document.getElementById("makeImage").disabled);
        assert.equal(renders.at(-1).use_stable_diffusion_model, "custom/fine-tune");
        assert.equal(renders.at(-1).use_lora_model, "private/test-lora");
        policyError = true; const before = galleryCalls.length;
        await main.goto(base + "/cpp-ui/gallery/images");
        await main.waitForFunction(() => document.documentElement.dataset.kiosk === "unavailable");
        assert.equal(galleryCalls.length, before, "Unknown policy must not fall back to local gallery");
        assert.deepEqual(errors, []);
        console.log("PASS: kiosk on/off, cross-tab refresh, preserved LoRA/model preferences, restricted render payload, Destockd-only/failed-source gallery, G-only Perchance response, unavailable-policy fail-closed, mobile layout; no real settings or generation changed.");
    } catch (error) {
        for (const context of browser.contexts()) for (const [index, page] of context.pages().entries()) {
            console.error("Page state:", page.url(), await page.evaluate(() => ({
                policy: document.documentElement.dataset.kiosk,
                banner: document.getElementById("kiosk-status-banner")?.textContent,
                lora: document.getElementById("lora_model")?.textContent,
                rows: document.querySelectorAll(".cpp-lora-entry").length,
            })).catch(() => ({})));
            await page.screenshot({path: path.join(output, `failure-${index}.png`)}).catch(() => {});
        }
        throw error;
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
