// Browser regression: every render POST is intercepted; no generation runs.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
    const browser = await chromium.launch({headless: true, chromiumSandbox: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    const output = process.env.MODIFIER_TEST_OUTPUT || "/tmp/cpp-modifier-tests";
    fs.mkdirSync(output, {recursive: true});
    try {
        const page = await browser.newPage({viewport: {width: 1280, height: 960}});
        const errors = [], requests = [];
        let catalogMode = "live";
        page.on("pageerror", error => errors.push(error.message));
        await page.route("**/*", async route => {
            const request = route.request(), url = new URL(request.url());
            if (request.method() === "POST") {
                assert.equal(url.pathname, "/render");
                requests.push(request.postDataJSON());
                return route.fulfill({status: 409, json: {detail: "Modifier test intercepted: no generation started"}});
            }
            if (url.pathname === "/get/modifiers") {
                if (catalogMode === "fail") return route.fulfill({status: 503, json: {detail: "Test failure"}});
                if (catalogMode === "empty") return route.fulfill({json: []});
                if (catalogMode === "invalid") return route.fulfill({json: {wrong: "shape"}});
                if (catalogMode === "unsafe") return route.fulfill({json: [{category: "Custom", modifiers: [
                    {modifier: "<b>Literal modifier</b>", previews: [{name: "landscape", path: "missing-test-preview.jpg"}]},
                ]}]});
            }
            if (url.pathname.endsWith("/missing-test-preview.jpg")) return route.fulfill({status: 404, body: "missing"});
            const models = {
                "/get/model": {models: [{model: "checkpoints/modifier-test", name: "Modifier test"}]},
                "/get/lora": {models: []}, "/get/models": {models: []},
            };
            if (models[url.pathname]) return route.fulfill({json: models[url.pathname]});
            return route.continue();
        });
        const url = process.env.MODIFIER_TEST_URL || "http://100.106.255.71:10000/cpp-ui";
        await page.goto(url);
        assert.equal(await page.title(), "Main · Easy Diffusion C++");
        const catalogResponse = await page.request.get(new URL("/get/modifiers", url).href);
        assert.ok(catalogResponse.ok());
        const catalog = await catalogResponse.json();
        const total = catalog.reduce((sum, category) => sum + category.modifiers.length, 0);
        assert.ok(total > 0, "Live modifier catalog must be populated");
        const cards = page.locator(".cpp-modifier-card");
        await page.waitForFunction(() => document.querySelectorAll(".cpp-modifier-card").length > 0, null, {timeout: 10000});
        assert.equal(await cards.count(), total, "C++ UI must display the existing catalog");
        const first = catalog[0].modifiers[0].modifier, second = catalog[0].modifiers[1].modifier;
        const card = name => cards.filter({has: page.locator(".cpp-modifier-name", {hasText: name})});
        const selected = page.locator("#editor-inputs-tags-list button");
        await page.locator("#cpp-image-modifiers").scrollIntoViewIfNeeded();
        await page.waitForFunction(() => {
            const image = document.querySelector(".cpp-modifier-card img");
            return image?.complete && image.naturalWidth > 0;
        });
        await page.fill("#prompt", "cityscape");
        await page.fill("#cpp-modifier-search", first);
        assert.equal(await cards.count(), 1);
        await card(first).click();
        assert.equal(await card(first).getAttribute("aria-pressed"), "true");
        assert.equal(await selected.count(), 1);
        assert.equal(await page.inputValue("#prompt"), "cityscape", "Selection must not mutate typed prompt");
        await card(first).click();
        assert.equal(await selected.count(), 0, "Clicking again must deselect");
        await card(first).click();
        await page.fill("#cpp-modifier-search", "zzzz-no-existing-modifier");
        assert.equal(await cards.count(), 0);
        assert.match(await page.locator("#cpp-modifier-status").innerText(), /No modifiers match/);
        await page.getByRole("button", {name: `Remove ${first}`, exact: true}).click();
        assert.equal(await selected.count(), 0, "Selected chips remain removable while filtering");
        await page.fill("#cpp-modifier-search", "");
        await page.selectOption("#cpp-modifier-category", catalog[1].category);
        assert.equal(await cards.count(), catalog[1].modifiers.length);
        await page.selectOption("#cpp-modifier-category", "");
        await card(first).click(); await card(second).click();
        await page.fill("#stable_diffusion_model", "modifier-test");
        await page.locator('#stable_diffusion_model-model-list [data-path="checkpoints/modifier-test"]').click();
        await page.click("#makeImage");
        await page.waitForFunction(() => document.getElementById("generation-queue-status").textContent.includes("Modifier test intercepted"));
        assert.equal(requests.length, 1);
        assert.equal(requests[0].prompt, `cityscape, ${first}, ${second}`);
        assert.equal(requests[0].original_prompt, "cityscape");
        assert.deepEqual(requests[0].active_tags, [first, second]);
        assert.deepEqual(requests[0].inactive_tags, []);
        await page.click("#makeImage");
        await page.waitForFunction(() => !document.getElementById("makeImage").disabled);
        assert.equal(requests.length, 2);
        assert.equal(requests[1].prompt, requests[0].prompt, "Repeated requests must not accumulate modifiers");
        await page.reload();
        await page.waitForFunction(() => document.querySelectorAll("#editor-inputs-tags-list button").length === 2);
        await page.locator("#cpp-image-modifiers").scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(output, "desktop.png")});
        await page.setViewportSize({width: 390, height: 844});
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Mobile horizontal overflow");
        await page.locator("#cpp-image-modifiers").scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(output, "mobile.png")});
        await page.click("#cpp-modifier-clear");
        assert.equal(await selected.count(), 0);
        assert.ok(await page.locator("#cpp-modifier-clear").isDisabled());
        await page.fill("#prompt", "cityscape");
        await page.click("#makeImage");
        await page.waitForFunction(() => !document.getElementById("makeImage").disabled);
        assert.deepEqual(requests.at(-1).active_tags, []);
        assert.equal(requests.at(-1).prompt, "cityscape");
        catalogMode = "fail";
        await page.reload();
        await page.waitForFunction(() => document.getElementById("cpp-modifier-status")?.textContent.includes("HTTP 503"));
        assert.ok(await page.locator("#cpp-modifier-retry").isVisible());
        catalogMode = "live";
        await page.click("#cpp-modifier-retry");
        await page.waitForFunction(() => document.querySelectorAll(".cpp-modifier-card").length > 0);
        assert.equal(await cards.count(), total);
        for (const mode of ["empty", "invalid"]) {
            catalogMode = mode;
            await page.reload();
            const expected = mode === "empty" ? "No image modifiers available" : "Invalid modifier catalog";
            await page.waitForFunction(text => document.getElementById("cpp-modifier-status")?.textContent.includes(text), expected);
        }
        catalogMode = "unsafe";
        await page.reload();
        await page.waitForSelector(".cpp-modifier-card");
        assert.equal(await cards.first().innerText(), "<b>Literal modifier</b>");
        assert.equal(await cards.locator("b").count(), 0, "Modifier labels must be rendered as text, not HTML");
        await cards.first().scrollIntoViewIfNeeded();
        await page.waitForFunction(() => document.querySelector(".cpp-modifier-card img")?.hidden === true);
        await cards.first().click();
        assert.equal(await selected.count(), 1, "Missing thumbnail must not disable the modifier");
        assert.deepEqual(errors, [], "No uncaught browser exceptions");
        console.log(`PASS: ${total} live modifiers; thumbnails, search/category, select/deselect/remove/clear, persistence, prompt payloads, retry/empty/invalid responses, safe text, missing thumbnail, desktop/mobile; no generation.`);
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
