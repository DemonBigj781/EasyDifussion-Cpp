// Browser test for the Training plugin with a controlled API and tab host.
const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright")
const root = path.resolve(__dirname, "..")

;(async () => {
    const browser = await chromium.launch({ headless: true,
        ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) })
    try {
        const page = await browser.newPage()
        const errors = []
        const requests = []
        page.on("pageerror", (error) => errors.push(error.message))
        await page.route("http://trainer.test/**", async (route) => {
            const url = new URL(route.request().url())
            if (!url.pathname.startsWith("/training/")) {
                await route.fulfill({ contentType: "text/html", body: '<div id="tabs"></div><div id="contents"></div>' })
                return
            }
            const body = route.request().postDataJSON()
            requests.push({ path: url.pathname, body })
            const data = {
                "/training/readiness": { ready: true, detail: "Ready", gpu: "Test GPU", dataset_root: "/datasets" },
                "/training/models": { models: [{ path: "/models/base.safetensors", name: "base" }] },
                "/training/jobs": { jobs: [] },
                "/training/dataset/scan": { count: 1, items: [{ image: "one.png", caption: "blue sky" }] },
                "/training/dataset/caption": { caption: "edited caption" },
            }[url.pathname] || {}
            if (url.pathname === "/training/jobs" && body) {
                await route.fulfill({ status: 409, json: { detail: "Generation is busy" } })
            } else await route.fulfill({ json: data })
        })
        await page.goto("http://trainer.test/")
        const html = fs.readFileSync(path.join(root, "ui/plugins/ui/training_plugin/training.tab.plugin.html"), "utf8")
        await page.evaluate((html) => {
            window.loadRequiredPluginHTML = () => html
            window.createTab = (request) => {
                const tab = document.createElement("button")
                tab.id = `tab-${request.id}`
                tab.textContent = request.label
                document.getElementById("tabs").append(tab)
                const content = document.createElement("div")
                content.id = request.id
                content.innerHTML = request.content
                document.getElementById("contents").append(content)
                tab.onclick = request.onOpen
            }
        }, html)
        await page.addScriptTag({ path: path.join(root, "ui/plugins/ui/training_plugin/training.tab.plugin.js") })
        await page.click("#tab-training")
        await page.waitForFunction(() => document.querySelector("#training-readiness").textContent.includes("Test GPU"))
        await page.fill("#training-dataset", "example")
        await page.click("#training-scan")
        await page.waitForFunction(() => document.querySelector("#training-caption").value === "blue sky")
        await page.fill("#training-caption", "edited caption")
        await page.click("#training-save-caption")
        await page.waitForFunction(() => document.querySelector("#training-dataset-status").textContent.includes("Saved"))
        await page.selectOption("#training-kind", "embedding")
        assert.equal(await page.isDisabled("#training-rank"), true)
        assert.equal(await page.isDisabled("#training-vectors"), false)
        assert.equal(await page.inputValue("#training-rate"), "0.0005")
        await page.selectOption("#training-architecture", "sdxl")
        assert.equal(await page.inputValue("#training-resolution"), "1024")
        await page.selectOption("#training-model", "/models/base.safetensors")
        await page.fill("#training-trigger", "newtoken")
        await page.click("#training-start")
        await page.waitForFunction(() => document.querySelector("#training-error").textContent === "Generation is busy")
        const submitted = requests.find((item) => item.path === "/training/jobs" && item.body)
        assert.equal(submitted.body.kind, "embedding")
        assert.equal(submitted.body.resolution, 1024)
        assert.equal(submitted.body.trigger, "newtoken")
        await page.addScriptTag({ path: path.join(root, "ui/plugins/ui/training_plugin/training.tab.plugin.js") })
        assert.equal(await page.locator("#tab-training").count(), 1)
        assert.deepEqual(errors, [])
        console.log("PASS: browser caption editing, family/type controls, job submission, errors, duplicate loading")
    } finally { await browser.close() }
})().catch((error) => { console.error(error); process.exitCode = 1 })
