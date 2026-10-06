const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
    const browser = await chromium.launch({headless: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    try {
        const context = await browser.newContext();
        await context.route("**/*", route => route.request().method() === "GET" ? route.continue() : route.abort());
        const page = await context.newPage();
        const origin = process.env.PROMPT_TEST_URL || "http://100.106.255.71:10000";
        await page.goto(origin + "/cpp-ui/settings/plugins");
        const toggle = page.locator('[data-optional-plugin-id="prompt-assist"]');
        await toggle.waitFor({timeout: 5000});
        assert.equal(await toggle.isChecked(), false);
        await toggle.check();
        for (const route of ["/cpp-ui", "/legacy"]) {
            await page.goto(origin + route);
            await page.waitForSelector("#spell-tokenizer-plugin-token-counter", {state: "attached"});
            assert.equal(await page.locator("#spell-tokenizer-plugin-token-counter").count(), 1);
            await page.fill("#prompt", "a blue cup on a table");
            assert.ok(Number(await page.locator("#spell-tokenizer-plugin-token-counter").innerText()) > 0);
        }
        await page.goto(origin + "/cpp-ui/settings/plugins");
        await toggle.uncheck();
        for (const route of ["/cpp-ui", "/legacy"]) {
            await page.goto(origin + route);
            if (route === "/legacy") await page.waitForSelector("#tab-plugin");
            else await page.waitForFunction(() => window.CppGeneratePage);
            assert.equal(await page.locator("#spell-tokenizer-plugin-token-counter").count(), 0);
        }
        console.log("PASS: shared prompt-assistance toggle is opt-in, activates once in each UI, and stays off after reload");
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
