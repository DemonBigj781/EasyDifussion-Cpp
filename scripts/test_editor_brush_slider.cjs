const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
    const browser = await chromium.launch({headless: true,
        ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
    try {
        const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
        await context.route("**/*", route => {
            const request = route.request();
            assert.equal(request.method(), "GET", "Brush tests must not mutate the backend");
            assert.notEqual(new URL(request.url()).pathname, "/image/stop");
            return route.continue();
        });
        const page = await context.newPage(), errors = [];
        page.on("pageerror", error => errors.push(error.message));
        await page.goto((process.env.EDITOR_TEST_URL || "http://100.106.255.71:10000") + "/legacy");
        await page.waitForSelector("#tab-image-editor-page");
        await page.click("#tab-image-editor-page");
        for (const [mode, popup, size] of [["draw", "image-editor", 37], ["inpaint", "image-inpainter", 19]]) {
            await page.click(`[data-editor-mode="${mode}"]`);
            const section = page.locator(`#${popup} .image_editor_brush_size`);
            const slider = section.locator('input[type="range"]');
            assert.equal(await slider.count(), 1, `${mode} must expose a brush-size slider instead of presets`);
            assert.equal(await section.locator(".editor-options-container").count(), 0, "Old preset buttons are removed");
            assert.equal(await slider.getAttribute("min"), "6");
            assert.equal(await slider.getAttribute("max"), "64");
            assert.equal(await slider.getAttribute("step"), "1");
            assert.equal(await slider.inputValue(), "48");
            await slider.focus();
            await page.keyboard.press("Home");
            assert.equal(await slider.inputValue(), "6");
            await page.keyboard.press("End");
            assert.equal(await slider.inputValue(), "64");
            await slider.evaluate((input, size) => {input.value = String(size); input.dispatchEvent(new Event("input", {bubbles: true}));}, size);
            assert.equal(await section.locator("output").innerText(), `${size} px`);
            assert.equal(await slider.getAttribute("aria-valuetext"), `${size} pixels`);
            const brush = await page.evaluate(mode => {
                const editor = mode === "draw" ? imageEditor : imageInpainter;
                return {value: editor.options.brush_size, lineWidth: editor.ctx_current.lineWidth * editor.containerScale};
            }, mode);
            assert.equal(brush.value, size);
            assert.ok(Math.abs(brush.lineWidth - size) < .001, "Slider updates the canvas brush immediately");
            const canvas = page.locator(`#${popup} .editor-canvas-overlay`);
            await canvas.scrollIntoViewIfNeeded();
            const box = await canvas.boundingBox();
            await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
            await page.mouse.down();
            await page.mouse.move(box.x + box.width / 2 + 20, box.y + box.height / 2 + 10, {steps: 4});
            await page.mouse.up();
            const drawnSize = await page.evaluate(mode => {
                const editor = mode === "draw" ? imageEditor : imageInpainter;
                return editor.history.events.at(-1)?.options?.brush_size;
            }, mode);
            assert.equal(drawnSize, size, "Actual strokes capture the selected brush size");
            await section.scrollIntoViewIfNeeded();
            await page.screenshot({path: `/tmp/${mode}-brush-slider-desktop.png`});
        }
        await page.click('[data-editor-mode="draw"]');
        assert.equal(await page.locator('#image-editor .image_editor_brush_size input').inputValue(), "37", "Draw size survives mode changes");
        assert.equal(await page.locator('#image-inpainter .image_editor_brush_size input').inputValue(), "19", "Inpaint size remains independent");
        await page.setViewportSize({width: 390, height: 844});
        for (const [mode, popup] of [["draw", "image-editor"], ["inpaint", "image-inpainter"]]) {
            await page.click(`[data-editor-mode="${mode}"]`);
            const slider = page.locator(`#${popup} .image_editor_brush_size input`);
            await slider.evaluate(input => input.scrollIntoView({block: "center"}));
            await slider.focus();
            await page.keyboard.press("Home");
            const bounds = await slider.boundingBox();
            assert.ok(bounds.width > 100 && bounds.x >= 0 && bounds.x + bounds.width <= 390, "Usable slider without mobile clipping");
            await slider.click({position: {x: bounds.width * .4, y: bounds.height / 2}});
            const state = await page.evaluate(mode => {
                const editor = mode === "draw" ? imageEditor : imageInpainter;
                return {size: editor.options.brush_size, displayed: editor.popup.querySelector('.image_editor_brush_size output').textContent};
            }, mode);
            assert.ok(state.size > 6 && state.size < 64, "Clicking the mobile track changes the brush size");
            assert.equal(state.displayed, `${state.size} px`);
            await slider.evaluate(input => input.scrollIntoView({block: "center"}));
            await page.screenshot({path: `/tmp/${mode}-brush-slider-mobile.png`});
        }
        assert.deepEqual(errors, []);
        console.log("PASS: Draw and Inpaint sliders, 6–64 range/default 48, live brush updates and values, keyboard endpoints, actual strokes, independent mode state, desktop/mobile layout, zero browser exceptions; no backend writes.");
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
