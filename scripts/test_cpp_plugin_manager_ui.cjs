// The legacy manager is exercised in an isolated DOM, preserving live kiosk policy.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
    const browser = await chromium.launch({headless: true, chromiumSandbox: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    const output = process.env.PLUGIN_TEST_OUTPUT || "/tmp/cpp-plugin-manager-test";
    fs.mkdirSync(output, {recursive: true});
    try {
        const context = await browser.newContext({viewport: {width: 1280, height: 960}});
        const errors = [];
        context.on("page", page => page.on("pageerror", error => errors.push(error.message)));
        await context.route("**/*", route => {
            assert.equal(route.request().method(), "GET", "Plugin-manager tests must not mutate backend settings or start generation");
            return route.continue();
        });
        const origin = process.env.PLUGIN_TEST_URL || "http://100.106.255.71:10000";
        const cpp = await context.newPage();
        await cpp.goto(origin + "/cpp-ui/settings/plugins");
        assert.equal(await cpp.locator("#backend_platform").count(), 0, "Plugin Config must not contain GPU/backend selection");
        assert.equal(await cpp.locator("#save-system-settings-btn").count(), 0);
        await cpp.waitForSelector("#cpp-plugin-manager-controls [data-optional-plugin-id]");
        assert.equal(await cpp.locator("[data-optional-plugin-id]").count(), 26);
        const input = (page, id) => page.locator(`[data-optional-plugin-id="${id}"]`);
        const legacy = await context.newPage();
        await legacy.goto(origin + "/cpp-ui/settings/plugins");
        await legacy.waitForSelector("[data-optional-plugin-id]");
        await legacy.evaluate(() => {
            document.body.innerHTML = '<div class="app-shell"><div id="tab-container"></div><div id="tab-content-wrapper"></div>' +
                '<div id="editor-settings" hidden><select id="sampler_name"><option value="euler">Euler</option></select>' +
                '<select id="scheduler_name"><option value="normal">Normal</option></select></div>' +
                '<div id="editor-inputs-tags-container">Image Modifiers <span id="editor-inputs-tags-list">Retained selection</span></div>' +
                '<div id="editor-modifiers" style="display:block">Modifier catalog</div>' +
                '<dialog id="modifier-settings-config">Custom modifiers</dialog></div>';
            window.linkTabContents = () => {};
        });
        await legacy.addScriptTag({url: origin + "/plugins/core/image_plugin/image-settings.plugin.js"});
        await legacy.addScriptTag({url: origin + "/media/js/utils.js"});
        await legacy.addScriptTag({url: origin + "/media/js/plugins.js"});
        await legacy.evaluate(() => {
            window.testPluginLoads = [];
            loadScript = async url => { window.testPluginLoads.push(url); return url; };
            createLocalPluginManagerTab();
        });
        assert.equal(await legacy.locator("[data-optional-plugin-id]").count(), 26);
        const generate = await context.newPage();
        await generate.goto(origin + "/cpp-ui");
        await generate.waitForSelector("#cpp-image-modifiers");
        assert.equal(await input(cpp, "image-modifiers").isChecked(), true);
        await legacy.evaluate(() => document.querySelector("#modifier-settings-config").showModal());
        await input(cpp, "image-modifiers").uncheck();
        await legacy.waitForFunction(() => !document.querySelector('[data-optional-plugin-id="image-modifiers"]').checked);
        await generate.waitForSelector("#cpp-image-modifiers", {state: "hidden"});
        assert.equal(await legacy.locator("#editor-inputs-tags-container").isVisible(), false);
        assert.equal(await legacy.locator("#editor-modifiers").isVisible(), false);
        assert.equal(await legacy.locator("#modifier-settings-config").evaluate(dialog => dialog.open), false);
        assert.equal(await legacy.locator("#editor-inputs-tags-list").textContent(), "Retained selection");
        await generate.reload();
        await generate.waitForFunction(() => window.LocalPluginPreferences);
        assert.equal(await generate.locator("#cpp-image-modifiers").isVisible(), false);
        await input(legacy, "image-modifiers").check();
        await cpp.waitForFunction(() => document.querySelector('[data-optional-plugin-id="image-modifiers"]').checked);
        await generate.waitForSelector("#cpp-image-modifiers");
        assert.equal(await legacy.locator("#editor-inputs-tags-container").isVisible(), true);
        assert.equal(await legacy.locator("#editor-modifiers").isVisible(), true);
        assert.deepEqual(await legacy.evaluate(() => testPluginLoads), [], "Visibility controls must not load a plugin script");
        await input(legacy, "image-modifiers").uncheck();
        await cpp.waitForFunction(() => !document.querySelector('[data-optional-plugin-id="image-modifiers"]').checked);
        await generate.waitForSelector("#cpp-image-modifiers", {state: "hidden"});
        await cpp.reload();
        await cpp.waitForSelector("[data-optional-plugin-id]");
        assert.equal(await input(cpp, "image-modifiers").isChecked(), false);
        await input(cpp, "image-modifiers").check();
        await legacy.waitForFunction(() => document.querySelector('[data-optional-plugin-id="image-modifiers"]').checked);
        await generate.waitForSelector("#cpp-image-modifiers");
        await input(cpp, "toggle-spellcheck").check();
        await legacy.waitForFunction(() => document.querySelector('[data-optional-plugin-id="toggle-spellcheck"]').checked);
        await input(legacy, "toggle-spellcheck").uncheck();
        await cpp.waitForFunction(() => !document.querySelector('[data-optional-plugin-id="toggle-spellcheck"]').checked);
        await input(legacy, "toggle-spellcheck").check();
        await cpp.waitForFunction(() => document.querySelector('[data-optional-plugin-id="toggle-spellcheck"]').checked);
        assert.equal(await legacy.evaluate(() => testPluginLoads.filter(url => url.includes("toggle-spellcheck")).length), 1);
        await input(cpp, "toggle-spellcheck").uncheck();
        await legacy.waitForFunction(() => document.querySelector('[data-optional-plugin-status="toggle-spellcheck"]').textContent.includes("disabled after reload"));
        await cpp.reload();
        await cpp.waitForSelector("[data-optional-plugin-id]");
        assert.equal(await input(cpp, "toggle-spellcheck").isChecked(), false);
        await cpp.fill("#cpp-plugin-filter", "toggle-spellcheck");
        assert.equal(await cpp.locator(".optional-ui-plugin-row:visible").count(), 1);
        await cpp.fill("#cpp-plugin-filter", "");
        await input(cpp, "perchance-gallery").uncheck();
        const native = await context.newPage();
        await native.goto(origin + "/cpp-ui/perchance/gallery");
        await native.waitForSelector("#cpp-plugin-disabled-notice");
        assert.equal(await native.locator("#perchance-generator-gallery-list-button").isVisible(), false);
        await input(cpp, "perchance-gallery").check();
        await native.waitForFunction(() => !document.documentElement.hasAttribute("data-native-plugin-disabled"));
        assert.equal(await native.locator("#perchance-generator-gallery-list-button").isVisible(), true);
        // A plugin preference must never undo the separate kiosk restriction.
        if (await cpp.evaluate(() => window.CppKiosk.enabled))
            assert.equal(await cpp.locator('nav a[href="/cpp-ui/perchance/image"]').isVisible(), false);
        await cpp.screenshot({path: path.join(output, "plugins-desktop.png")});
        await cpp.setViewportSize({width: 390, height: 844});
        assert.ok(await cpp.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        await cpp.screenshot({path: path.join(output, "plugins-mobile.png")});
        await native.goto(origin + "/cpp-ui/settings/gpu");
        await native.waitForSelector("#backend_platform");
        assert.equal(await native.locator("#backend_platform").count(), 1);
        assert.deepEqual(errors, []);
        console.log("PASS: 26 shared controls; modifier hide/show in both directions, modal closure, retained selections and reload persistence; no visibility script loading; legacy loader callback; native Perchance gating; kiosk priority; desktop/mobile; no backend writes.");
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
