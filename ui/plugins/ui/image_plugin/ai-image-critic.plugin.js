/*
 * AI Image Critic for Easy Diffusion C++.
 * Adapted from AvidGameFan's ai-image-critic.plugin.js v1.1.0:
 * https://github.com/AvidGameFan/ed-plugins/blob/master/ai-image-critic.plugin.js
 *
 * This port uses the bundled llama.cpp server and local GGUF models. It does
 * not use Perchance and never sends the image to an external API.
 */

;(function () {
    "use strict"
    if (window.__easyDiffusionAiImageCriticLoaded) return
    window.__easyDiffusionAiImageCriticLoaded = true

    const ID = "ai-image-critic"
    const STORAGE_KEY = "easy-diffusion-ai-image-critic-v1"
    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode || !Array.isArray(PLUGINS?.IMAGE_INFO_BUTTONS)) {
        console.error("AI Image Critic: required Easy Diffusion UI APIs were not found")
        return
    }

    const escapeHtml = (value) => String(value || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")

    const notify = (message, isError = false) => {
        if (typeof showToast === "function") showToast(message, 6500, isError)
        else if (isError) console.error(message)
        else console.log(message)
    }

    const readSettings = () => {
        try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") }
        catch (_) { return {} }
    }

    const panel = document.createElement("div")
    panel.id = `${ID}-settings`
    panel.className = "settings-box panel-box"
    panel.innerHTML = `
        <h4 class="collapsible"><i class="fa-solid fa-magnifying-glass"></i> AI Image Critic</h4>
        <div class="collapsible-content settings-panel-entries">
            <p>Analyze a finished image locally with llama.cpp. A compatible multimodal model needs both a language GGUF and an <code>mmproj</code> vision GGUF.</p>
            <label for="${ID}-text-model">Text GGUF</label>
            <select id="${ID}-text-model"></select>
            <label for="${ID}-vision-model">Vision GGUF / bundle</label>
            <select id="${ID}-vision-model"></select>
            <label for="${ID}-gpu-layers">GPU layers</label>
            <input id="${ID}-gpu-layers" type="number" min="0" max="999" step="1" value="99">
            <label for="${ID}-context">Context size</label>
            <input id="${ID}-context" type="number" min="2048" max="32768" step="1024" value="4096">
            <button id="${ID}-refresh" type="button" class="tertiaryButton">Refresh models</button>
            <small id="${ID}-status">Checking local GGUF models…</small>
        </div>`
    editor.after(panel)
    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const textModel = document.getElementById(`${ID}-text-model`)
    const visionModel = document.getElementById(`${ID}-vision-model`)
    const gpuLayers = document.getElementById(`${ID}-gpu-layers`)
    const contextSize = document.getElementById(`${ID}-context`)
    const status = document.getElementById(`${ID}-status`)
    const saved = readSettings()
    gpuLayers.value = saved.gpuLayers ?? "99"
    contextSize.value = saved.contextSize ?? "4096"

    function saveSettings() {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            textModel: textModel.value,
            visionModel: visionModel.value,
            gpuLayers: gpuLayers.value,
            contextSize: contextSize.value,
        }))
    }
    ;[textModel, visionModel, gpuLayers, contextSize].forEach((field) => {
        field.addEventListener("change", saveSettings)
    })

    async function requestJson(url, options) {
        const response = await fetch(url, options)
        const raw = await response.text()
        let data
        try { data = raw ? JSON.parse(raw) : {} }
        catch (_) { data = { detail: raw } }
        if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`)
        return data
    }

    function fillSelect(select, values, selected, label) {
        select.replaceChildren()
        if (!values.length) {
            const option = document.createElement("option")
            option.value = ""
            option.textContent = `No ${label} found`
            select.appendChild(option)
            return
        }
        values.forEach((entry) => {
            const value = typeof entry === "string" ? entry : entry.model
            const option = document.createElement("option")
            option.value = value
            option.textContent = typeof entry === "string"
                ? entry
                : `${value}${entry.usable ? "" : " — incompatible"}`
            option.title = typeof entry === "string" ? value : entry.detail
            option.dataset.usable = typeof entry === "string" || entry.usable ? "true" : "false"
            select.appendChild(option)
        })
        if (values.some((entry) => (typeof entry === "string" ? entry : entry.model) === selected)) {
            select.value = selected
        }
    }

    async function refreshModels() {
        status.textContent = "Checking local GGUF models…"
        try {
            const inventory = await requestJson("/ai-critic/models")
            fillSelect(textModel, inventory.text_models || [], readSettings().textModel, "text GGUFs")
            fillSelect(visionModel, inventory.vision_models || [], readSettings().visionModel, "vision GGUFs")
            const usable = (inventory.vision_models || []).filter((entry) => entry.usable).length
            if (!inventory.ready) {
                status.textContent = "llama-server is not built. Run ./install.sh --llama-build."
            } else if (!usable) {
                status.textContent = `No usable vision pair found in ${inventory.vision_directory}. Add a matching mmproj GGUF.`
            } else {
                status.textContent = `${usable} compatible vision choice${usable === 1 ? "" : "s"} ready. Images stay local.`
            }
            saveSettings()
        } catch (error) {
            status.textContent = `Could not list critic models: ${error.message}`
        }
    }
    document.getElementById(`${ID}-refresh`).addEventListener("click", refreshModels)

    function imageAsJpeg(image) {
        const width = image.naturalWidth || image.width
        const height = image.naturalHeight || image.height
        if (!width || !height) throw new Error("The selected image is not ready.")
        const maximum = 1536
        const scale = Math.min(1, maximum / Math.max(width, height))
        const canvas = document.createElement("canvas")
        canvas.width = Math.max(1, Math.round(width * scale))
        canvas.height = Math.max(1, Math.round(height * scale))
        const context = canvas.getContext("2d")
        if (!context) throw new Error("The browser could not create an image canvas.")
        context.drawImage(image, 0, 0, canvas.width, canvas.height)
        return canvas.toDataURL("image/jpeg", 0.88)
    }

    const severityColors = {
        none: "#28a745",
        minor: "#d39e00",
        moderate: "#fd7e14",
        severe: "#dc3545",
    }

    function appendPrompt(targetSelector, addition, button) {
        const target = document.querySelector(targetSelector)
        if (!target || !addition.trim()) return
        const current = target.value.trim()
        target.value = current ? `${current}, ${addition.trim()}` : addition.trim()
        target.dispatchEvent(new Event("input", { bubbles: true }))
        target.dispatchEvent(new Event("change", { bubbles: true }))
        const oldText = button.textContent
        button.textContent = "Added"
        setTimeout(() => { button.textContent = oldText }, 1600)
    }

    function showReport(report) {
        document.getElementById(`${ID}-modal`)?.remove()
        const modal = document.createElement("div")
        modal.id = `${ID}-modal`
        modal.style.cssText = "position:fixed;inset:0;z-index:20000;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.65);padding:16px"
        const severity = severityColors[report.severity] ? report.severity : "moderate"
        const issues = Array.isArray(report.issues) ? report.issues : []
        const issueRows = issues.map((issue) => `
            <tr>
                <td style="padding:6px 9px;font-weight:bold;color:${severityColors[issue.severity] || severityColors.minor}">${escapeHtml(issue.area)}</td>
                <td style="padding:6px 9px">${escapeHtml(issue.description)}</td>
                <td style="padding:6px 9px">${escapeHtml(issue.severity)}</td>
            </tr>`).join("")
        const positive = escapeHtml(report.positive_prompt_additions)
        const negative = escapeHtml(report.negative_prompt_additions)
        modal.innerHTML = `
            <div style="background:var(--background-color2,#1e1e2e);color:var(--text-color,#eee);border-radius:10px;padding:22px;max-width:760px;width:100%;max-height:88vh;overflow:auto;box-shadow:0 8px 32px rgba(0,0,0,.7)">
                <div style="display:flex;justify-content:space-between;gap:12px;align-items:center">
                    <h3 style="margin:0">AI Image Critic <span style="font-size:12px;padding:3px 9px;border-radius:12px;background:${severityColors[severity]};color:white">${severity}</span></h3>
                    <button type="button" data-critic-close class="tertiaryButton" aria-label="Close">×</button>
                </div>
                <p>${escapeHtml(report.summary)}</p>
                <h4>Issues</h4>
                <table style="width:100%;border-collapse:collapse"><tbody>${issueRows || '<tr><td style="padding:8px">No issues detected.</td></tr>'}</tbody></table>
                ${positive ? `<h4>Positive prompt additions</h4><div style="display:flex;gap:8px;align-items:start"><textarea id="${ID}-positive" readonly style="flex:1;min-height:58px">${positive}</textarea><button type="button" data-critic-append="#prompt" data-source="${ID}-positive" class="tertiaryButton">Add</button></div>` : ""}
                ${negative ? `<h4>Negative prompt additions</h4><div style="display:flex;gap:8px;align-items:start"><textarea id="${ID}-negative" readonly style="flex:1;min-height:58px">${negative}</textarea><button type="button" data-critic-append="#negative_prompt" data-source="${ID}-negative" class="tertiaryButton">Add</button></div>` : ""}
                ${report.parameter_suggestions ? `<h4>Parameter suggestions</h4><p>${escapeHtml(report.parameter_suggestions)}</p>` : ""}
                <div style="text-align:right;margin-top:16px"><button type="button" data-critic-close class="secondaryButton">Close</button></div>
            </div>`
        document.body.appendChild(modal)
        modal.querySelectorAll("[data-critic-close]").forEach((button) => button.addEventListener("click", () => modal.remove()))
        modal.addEventListener("click", (event) => { if (event.target === modal) modal.remove() })
        modal.querySelectorAll("[data-critic-append]").forEach((button) => {
            button.addEventListener("click", () => {
                const source = document.getElementById(button.dataset.source)
                appendPrompt(button.dataset.criticAppend, source?.value || "", button)
            })
        })
    }

    async function analyzeImage(_request, image, _event, tools) {
        if (!visionModel.value) {
            notify("Add and select a compatible vision GGUF and mmproj first.", true)
            return
        }
        if (visionModel.selectedOptions[0]?.dataset.usable === "false") {
            notify(visionModel.selectedOptions[0].title || "The selected vision GGUF has no projector.", true)
            return
        }
        const button = this
        button.disabled = true
        if (tools?.spinner && tools?.spinnerStatus) {
            tools.spinnerStatus.textContent = "Loading local vision model…"
            tools.spinner.classList.remove("displayNone")
        }
        status.textContent = "Analyzing image locally…"
        try {
            const report = await requestJson("/ai-critic/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: imageAsJpeg(image),
                    text_model: textModel.value,
                    vision_model: visionModel.value,
                    gpu_layers: Number.parseInt(gpuLayers.value, 10) || 0,
                    context_length: Number.parseInt(contextSize.value, 10) || 4096,
                }),
            })
            showReport(report)
            status.textContent = "Analysis complete."
        } catch (error) {
            status.textContent = `Analysis failed: ${error.message}`
            notify(status.textContent, true)
        } finally {
            button.disabled = false
            tools?.spinner?.classList.add("displayNone")
        }
    }

    PLUGINS.IMAGE_INFO_BUTTONS.push([
        { type: "label", text: "AI Critic" },
        { html: '<i class="fa-solid fa-magnifying-glass"></i> Analyze', on_click: analyzeImage, filter: () => true },
    ])

    refreshModels()
})()
