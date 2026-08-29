// Native sdkit3 ControlNet-LLLite image-conditioning panel.

;(function () {
    "use strict"
    if (window.__controlNetLLLitePluginLoaded) return
    window.__controlNetLLLitePluginLoaded = true

    const STATE_KEY = "easy-diffusion-controlnet-lllite-v2"
    const LEGACY_STATE_KEY = "easy-diffusion-controlled-image-generation-v1"
    const DEFAULT_MODEL = "controlnet-lllite-depth-test"
    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode || typeof ModelDropdown !== "function" || typeof PLUGINS !== "object") {
        console.error("ControlNet-LLLite plugin: required Easy Diffusion UI APIs were not found")
        return
    }

    const panel = document.createElement("div")
    panel.id = "sdkit3-lllite-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel gated-feature"
    panel.dataset.featureKeys = "backend_sdkit3"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/controlnet-lllite.plugin.html")

    const preprocessorPanel = document.getElementById("sdkit3-controlnet-preprocessor-panel")
    const controlPanel = document.getElementById("sdkit3-controlnet-panel")
    if (preprocessorPanel) preprocessorPanel.after(panel)
    else if (controlPanel) controlPanel.after(panel)
    else editor.after(panel)
    window.orderSdkitSettingsPanels?.()
    setTimeout(() => window.orderSdkitSettingsPanels?.(), 250)

    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const byId = (id) => document.getElementById(id)
    const enabled = byId("lllite-enabled")
    const imageInput = byId("lllite-image-input")
    const preview = byId("lllite-image-preview")
    const wrapper = byId("lllite-image-wrapper")
    const model = new ModelDropdown(byId("lllite-model"), "controlnet-lllite", "None")

    function readJSON(key) {
        try { return JSON.parse(localStorage.getItem(key) || "{}") }
        catch (_) { return {} }
    }

    function clamp(value, minimum, maximum, fallback) {
        const number = Number(value)
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback
    }

    function notify(message, isError) {
        if (typeof showToast === "function") showToast(message, 5000, Boolean(isError))
        else if (isError) console.error(message)
        else console.log(message)
    }

    function hasImage() {
        return /^data:image\//.test(preview.getAttribute("src") || "")
    }

    function updateImageUI() {
        const present = hasImage()
        wrapper.classList.toggle("displayNone", !present)
    }

    function saveState() {
        localStorage.setItem(STATE_KEY, JSON.stringify({
            enabled: enabled.checked,
            model: model.value,
            strength: byId("lllite-strength").value,
            start: byId("lllite-start").value,
            end: byId("lllite-end").value,
        }))
    }

    function setImage(source) {
        if (!source || !/^data:image\//.test(source)) return false
        preview.src = source
        enabled.checked = true
        updateImageUI()
        saveState()
        return true
    }

    imageInput.addEventListener("change", () => {
        const file = imageInput.files?.[0]
        if (!file?.type.startsWith("image/")) return
        const reader = new FileReader()
        reader.addEventListener("load", () => setImage(reader.result))
        reader.readAsDataURL(file)
    })
    byId("lllite-use-init").addEventListener("click", () => {
        const init = document.getElementById("init_image_preview")
        if (!setImage(init?.getAttribute("src") || "")) notify("No Initial Image is loaded.", true)
    })
    byId("lllite-image-clear").addEventListener("click", () => {
        imageInput.value = ""
        preview.removeAttribute("src")
        enabled.checked = false
        updateImageUI()
        saveState()
    })
    preview.addEventListener("load", updateImageUI)

    const state = readJSON(STATE_KEY)
    const legacy = readJSON(LEGACY_STATE_KEY)
    enabled.checked = Boolean(state.enabled ?? legacy.llliteEnabled)
    byId("lllite-strength").value = state.strength ?? legacy.llliteStrength ?? "1"
    byId("lllite-start").value = state.start ?? legacy.llliteStart ?? "0"
    byId("lllite-end").value = state.end ?? legacy.llliteEnd ?? "100"
    model.value = state.model || legacy.llliteModel || DEFAULT_MODEL
    panel.querySelectorAll("input").forEach((input) => input.addEventListener("change", saveState))
    updateImageUI()

    PLUGINS.TASK_CREATE.push(function (event) {
        if (!enabled.checked) return
        if (!model.value || !hasImage()) {
            notify("ControlNet-LLLite is enabled, but its model or condition image is missing.", true)
            return
        }
        const rawStart = clamp(byId("lllite-start").value, 0, 100, 0)
        const rawEnd = clamp(byId("lllite-end").value, 0, 100, 100)
        event.reqBody.control_net_lllite_model = model.value
        event.reqBody.control_net_lllite_image = preview.src
        event.reqBody.control_net_lllite_strength = clamp(byId("lllite-strength").value, -10, 10, 1)
        event.reqBody.control_net_lllite_start_percent = Math.min(rawStart, rawEnd)
        event.reqBody.control_net_lllite_end_percent = Math.max(rawStart, rawEnd)
        saveState()
    })
})()
