const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
    const browser = await chromium.launch({headless: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    try {
        const page = await browser.newPage({viewport: {width: 1280, height: 960}});
        const errors = [];
        page.on("pageerror", error => errors.push(error.message));
        await page.route("**/*", route => {
            assert.equal(route.request().method(), "GET", "Diagnostics must be read-only");
            return route.continue();
        });
        const origin = process.env.DIAGNOSTICS_TEST_URL || "http://100.106.255.71:10000";
        await page.goto(origin + "/cpp-ui/console");
        const command = async text => {
            await page.fill("#ui-console-command", text);
            await page.click("#ui-console-run");
            await page.waitForFunction(() => !document.querySelector("#ui-console-run").disabled);
            return page.locator("#ui-console-output").innerText();
        };
        assert.match(await command("help"), /stat <path>/);
        assert.match(await command("status"), /Online/);
        assert.match(await command("ls"), /models/);
        assert.match(await command("cd models"), /Directory: models/);
        assert.match(await command("stat vae"), /"type": "directory"/);
        await command("cd");
        assert.match(await command("cd .."), /relative to the application root/);
        assert.match(await command("logs 2"), /info:/);
        assert.match(await command("logs 1001"), /Usage: logs/);
        assert.match(await command("rm ignored"), /Shell execution is not supported/);
        await page.locator("#ui-console-command").press("ArrowUp");
        assert.equal(await page.inputValue("#ui-console-command"), "rm ignored");
        await command("clear");
        assert.equal(await page.locator("#ui-console-output").innerText(), "");
        await command("ls");
        await page.screenshot({path: "/tmp/cpp-console-desktop.png"});
        await page.setViewportSize({width: 390, height: 844});
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        await page.screenshot({path: "/tmp/cpp-console-mobile.png"});
        await page.goto(origin + "/cpp-ui/logs");
        await page.waitForFunction(() => document.querySelector("#log-viewer-status").textContent.startsWith("Showing"));
        assert.ok((await page.locator("#log-viewer-output").innerText()).length > 0);
        await page.route("**/cpp-ui/api/logs", route => route.fulfill({status: 503, json: {detail: "Test unavailable"}}));
        await page.click("#log-viewer-refresh");
        await page.waitForFunction(() => document.querySelector("#log-viewer-status").textContent.includes("Test unavailable"));
        assert.deepEqual(errors, []);
        console.log("PASS: live logs, errors, diagnostic commands, file metadata/navigation, root boundary, history, no shell/writes, desktop/mobile");
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
