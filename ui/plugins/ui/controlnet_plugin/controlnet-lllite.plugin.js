// Native ControlNet-LLLite mode for the unified ControlNet panel.

;(function () {
    "use strict"
    if (window.__controlNetLLLitePluginLoaded) return
    window.__controlNetLLLitePluginLoaded = true

    const controller = window.controlNetModeController
    const slot = document.getElementById("controlnet-extension-modes")
    if (!controller || !slot || typeof ModelDropdown !== "function" || typeof PLUGINS !== "object") {
        console.error("ControlNet-LLLite mode: required Easy Diffusion UI APIs were not found")
        return
    }

    const STATE_KEY = "easy-diffusion-controlnet-lllite-v2"
    const LEGACY_STATE_KEY = "easy-diffusion-controlled-image-generation-v1"
    const DEFAULT_MODEL = "controlnet-lllite-depth-test"
    const section = document.createElement("div")
    section.id = "controlnet-mode-lllite"
    section.innerHTML = window.loadRequiredPluginHTML("/plugins/core/controlnet_plugin/controlnet-lllite.plugin.html")
    slot.appendChild(section)
    if (typeof prettifyInputs === "function") prettifyInputs(section)

    const byId = (id) => document.getElementById(id)
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

    function saveState() {
        localStorage.setItem(STATE_KEY, JSON.stringify({
            model: model.value,
            strength: byId("lllite-strength").value,
            start: byId("lllite-start").value,
            end: byId("lllite-end").value,
        }))
    }

    const state = readJSON(STATE_KEY)
    const legacy = readJSON(LEGACY_STATE_KEY)
    byId("lllite-strength").value = state.strength ?? legacy.llliteStrength ?? "1"
    byId("lllite-start").value = state.start ?? legacy.llliteStart ?? "0"
    byId("lllite-end").value = state.end ?? legacy.llliteEnd ?? "100"
    model.value = state.model || legacy.llliteModel || DEFAULT_MODEL
    section.querySelectorAll("input").forEach((input) => input.addEventListener("change", saveState))
    controller.register("lllite", section)
    controller.registerValidator("lllite", () => {
        if (!model.value) return "ControlNet-LLLite: choose a model."
        if (!/^data:image\//.test(controller.sharedImage())) return "ControlNet-LLLite: load the shared ControlNet image."
        return ""
    })

    PLUGINS.TASK_BUILD.push(function (event) {
        if (controller.mode !== "lllite") return
        const image = controller.sharedImage()
        if (!model.value || !/^data:image\//.test(image)) {
            notify("ControlNet-LLLite needs a model and the shared ControlNet image.", true)
            return
        }
        const rawStart = clamp(byId("lllite-start").value, 0, 100, 0)
        const rawEnd = clamp(byId("lllite-end").value, 0, 100, 100)
        delete event.reqBody.use_controlnet_model
        event.reqBody.control_net_lllite_model = model.value
        event.reqBody.control_net_lllite_image = image
        event.reqBody.control_image = image
        event.reqBody.control_net_lllite_strength = clamp(byId("lllite-strength").value, -10, 10, 1)
        event.reqBody.control_net_lllite_start_percent = Math.min(rawStart, rawEnd)
        event.reqBody.control_net_lllite_end_percent = Math.max(rawStart, rawEnd)
        const filter = controller.sharedFilter()
        if (filter) event.reqBody.control_filter_to_apply = filter
        else delete event.reqBody.control_filter_to_apply
        saveState()
    })
})()
