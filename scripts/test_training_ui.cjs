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
                "/training/readiness": { ready: true, native_cpp: {ready: true}, python_runtime: {ready: true}, detail: "Ready", gpu: "Test GPU", dataset_root: "/datasets" },
                "/training/models": { models: [{ path: "/models/base.safetensors", name: "base" }] },
                "/training/datasets": {datasets: ["example"]},
                "/training/assets": {text_encoders: [], vaes: []},
                "/training/scrape/sources": {sources: []},
                "/training/spritegpt": {running: false, detail: "Stopped"},
                "/training/jobs": { jobs: [] },
            }[url.pathname] || {}
            if (url.pathname === "/training/jobs" && body) {
                await route.fulfill({ status: 409, json: { detail: "Generation is busy" } })
            } else await route.fulfill({ json: data })
        })
        await page.goto("http://trainer.test/")
        await page.addScriptTag({ path: path.join(root, "ui/media/js/utils.js") })
        await page.addScriptTag({ path: path.join(root, "ui/media/js/searchable-models.js") })
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
        await page.selectOption("#training-dataset", "example")
        await page.fill("#training-model", "base")
        await page.locator('#training-model-model-list [data-path="/models/base.safetensors"]').click()
        await page.evaluate(() => {
            modelsOptions = {"stable-diffusion": ["not-a-training-checkpoint"]}
            document.dispatchEvent(new Event("refreshModels"))
        })
        assert.equal(await page.locator("#training-model").getAttribute("data-path"), "/models/base.safetensors")
        assert.equal(await page.inputValue("#training-rate"), "0.00005")
        assert.equal(await page.inputValue("#training-text-encoder-rate"), "0.000005")
        assert.equal(await page.inputValue("#training-steps"), "1200")
        assert.equal(await page.inputValue("#training-alpha"), "")
        assert.equal(await page.isDisabled("#training-warmup"), true)
        await page.fill("#training-rank", "32")
        await page.fill("#training-alpha", "16")
        await page.selectOption("#training-scheduler", "cosine_with_restarts")
        await page.fill("#training-warmup", "20")
        await page.fill("#training-cycles", "3")
        await page.click("#training-start")
        await page.waitForFunction(() => document.querySelector("#training-error").textContent === "Generation is busy")
        const native = requests.find(item => item.path === "/training/jobs" && item.body)
        assert.equal(native.body.text_encoder_learning_rate, 5e-6)
        assert.equal(native.body.learning_rate, 5e-5)
        assert.equal(native.body.steps, 1200)
        assert.equal(native.body.network_alpha, 16)
        assert.equal(native.body.lr_scheduler, "cosine_with_restarts")
        assert.equal(native.body.lr_warmup_steps, 20)
        assert.equal(native.body.lr_scheduler_num_cycles, 3)
        await page.selectOption("#training-architecture", "sdxl")
        assert.equal(await page.isDisabled("#training-text-encoder-rate"), true)
        await page.selectOption("#training-kind", "embedding")
        assert.equal(await page.isDisabled("#training-rank"), true)
        assert.equal(await page.isDisabled("#training-alpha"), true)
        await page.selectOption("#training-scheduler", "constant")
        assert.equal(await page.isDisabled("#training-vectors"), false)
        assert.equal(await page.inputValue("#training-rate"), "0.0005")
        await page.selectOption("#training-architecture", "sdxl")
        assert.equal(await page.inputValue("#training-resolution"), "1024")
        await page.fill("#training-model", "base")
        await page.locator('#training-model-model-list [data-path="/models/base.safetensors"]').click()
        await page.fill("#training-trigger", "newtoken")
        await page.click("#training-start")
        await page.waitForFunction(() => document.querySelector("#training-error").textContent === "Generation is busy")
        const submitted = requests.findLast((item) => item.path === "/training/jobs" && item.body)
        assert.equal(submitted.body.kind, "embedding")
        assert.equal(submitted.body.network_alpha, null)
        assert.equal(submitted.body.lr_warmup_steps, 0)
        assert.equal(submitted.body.lr_scheduler_num_cycles, 1)
        assert.equal(submitted.body.text_encoder_learning_rate, 0)
        assert.equal(submitted.body.resolution, 1024)
        assert.equal(submitted.body.trigger, "newtoken")
        await page.selectOption("#training-kind", "lora")
        assert.equal(await page.inputValue("#training-rate"), "0.00005")
        await page.selectOption("#training-architecture", "sd15")
        assert.equal(await page.isDisabled("#training-text-encoder-rate"), false)
        assert.equal(await page.inputValue("#training-text-encoder-rate"), "0.000005")
        await page.selectOption("#training-backend", "python")
        assert.equal(await page.isDisabled("#training-batch"), false)
        await page.click("#training-start")
        await page.waitForTimeout(100)
        assert.equal(requests.findLast(item => item.body && item.path === "/training/jobs").body.backend, "python")
        await page.selectOption("#training-backend", "native")
        await page.fill("#training-epochs", "10")
        await page.fill("#training-repeats", "150")
        await page.click("#training-start")
        await page.waitForTimeout(100)
        const epochJob = requests.findLast(item => item.body && item.path === "/training/jobs").body
        assert.equal(epochJob.backend, "native")
        assert.equal(epochJob.epochs, 10)
        assert.equal(epochJob.dataset_repeats, 150)
        await page.addScriptTag({ path: path.join(root, "ui/plugins/ui/training_plugin/training.tab.plugin.js") })
        assert.equal(await page.locator("#tab-training").count(), 1)
        assert.deepEqual(errors, [])
        console.log("PASS: prepared datasets, separate rates, family/type controls in both directions, job submission, errors, duplicate loading")
    } finally { await browser.close() }
})().catch((error) => { console.error(error); process.exitCode = 1 })
