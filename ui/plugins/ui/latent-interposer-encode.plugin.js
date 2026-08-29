// Native sdkit3 VAE -> checkpoint latent conversion.
// Independent Easy Diffusion settings plugin.

;(function () {
    "use strict"
    if (window.__latentInterposerEncodePluginLoaded) return
    window.__latentInterposerEncodePluginLoaded = true

    const PANEL_ID = "sdkit3-encode-interpose-panel"
    const STATE_KEY = "easy-diffusion-latent-interposer-encode-v2"
    const LEGACY_STATE_KEY = "easy-diffusion-controlled-image-generation-v1"
    const conversions = new Set([
        "v1-to-xl", "v1-to-v3", "xl-to-v1", "xl-to-v3",
        "v3-to-v1", "v3-to-xl", "fx-to-v1", "fx-to-xl", "fx-to-v3",
    ])
    const labels = {
        v1: "Stable Diffusion v1.x",
        xl: "SDXL",
        v3: "Stable Diffusion 3",
        fx: "Flux.1",
        ca: "Stable Cascade",
    }

    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode || typeof ModelDropdown !== "function" || typeof PLUGINS !== "object") {
        console.error("Encode Interpose: required Easy Diffusion UI APIs were not found")
        return
    }
    if (document.getElementById(PANEL_ID)) return

    const panel = document.createElement("div")
    panel.id = PANEL_ID
    panel.className = "settings-box panel-box latent-interposer-panel gated-feature"
    panel.dataset.featureKeys = "backend_sdkit3"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/latent-interposer-encode.plugin.html")

    const decodePanel = document.getElementById("sdkit3-decode-interpose-panel")
    if (decodePanel) decodePanel.before(panel)
    else editor.after(panel)
    window.orderSdkitSettingsPanels?.()
    setTimeout(() => window.orderSdkitSettingsPanels?.(), 250)

    if (!document.getElementById("latent-interposer-plugin-style")) {
        const style = document.createElement("style")
        style.id = "latent-interposer-plugin-style"
        style.textContent = `
            .latent-interposer-panel h4 { cursor: pointer; }
            .latent-interposer-panel h4 small { float: right; }
            .latent-interposer-grid {
                display: grid;
                grid-template-columns: minmax(145px, auto) minmax(0, 1fr);
                gap: 6px 9px;
                align-items: center;
                margin: 8px 0;
            }
            @media (max-width: 700px) {
                .latent-interposer-grid { grid-template-columns: 1fr; }
            }`
        document.head.appendChild(style)
    }
    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const byId = (id) => document.getElementById(id)
    const enabled = byId("latent-interposer-encode-enabled")
    const source = byId("latent-interposer-encode-source")
    const destination = byId("latent-interposer-encode-destination")
    const direction = byId("latent-interposer-encode-direction")
    const status = byId("latent-interposer-encode-status")
    const model = new ModelDropdown(byId("latent-interposer-encode-model"), "latent-interposer", "None")

    function readJSON(key) {
        try { return JSON.parse(localStorage.getItem(key) || "{}") }
        catch (_) { return {} }
    }

    function modelTags(type, name) {
        if (typeof modelsDB === "undefined" || !modelsDB || !name) return []
        return modelsDB[type]?.[name]?.tags || []
    }

    function checkpointFamily() {
        const name = document.getElementById("stable_diffusion_model")?.dataset.path || ""
        const tags = modelTags("stable-diffusion", name)
        if (tags.some((tag) => String(tag).startsWith("stable_cascade"))) return "ca"
        if (tags.some((tag) => String(tag).startsWith("flux") || tag === "chroma")) return "fx"
        if (tags.some((tag) => String(tag).startsWith("sd_v3"))) return "v3"
        if (tags.some((tag) => String(tag).startsWith("sd_xl") || String(tag).startsWith("playground_v2_5"))) return "xl"
        if (tags.some((tag) => String(tag).startsWith("sd_v1") || String(tag).startsWith("sd_v2") || tag === "instruct_pix2pix")) return "v1"

        const lower = name.toLowerCase()
        if (lower.includes("cascade")) return "ca"
        if (lower.includes("flux")) return "fx"
        if (lower.includes("sd3") || lower.includes("stable-diffusion-3")) return "v3"
        if (lower.includes("sdxl") || lower.includes("/xl") || lower.includes("pony") || lower.includes("illustrious")) return "xl"
        return "v1"
    }

    function vaeFamily(modelFamily) {
        const name = document.getElementById("vae_model")?.dataset.path || ""
        if (!name || name.toLowerCase() === "none") return modelFamily
        const tags = modelTags("vae", name)
        for (const family of ["v1", "xl", "v3", "fx"]) {
            if (tags.includes(`vae_${family}`)) return family
        }

        const lower = name.toLowerCase()
        if (lower.includes("flux") || /(^|\/)ae(?:\.|$)/.test(lower)) return "fx"
        if (lower.includes("sd3") || lower.includes("stable-diffusion-3")) return "v3"
        if (lower.includes("sdxl") || lower.includes("sd_xl") || lower.includes("pony") || lower.includes("illustrious")) return "xl"
        return "v1"
    }

    function saveState() {
        localStorage.setItem(STATE_KEY, JSON.stringify({
            enabled: enabled.checked,
            model: model.value,
        }))
    }

    function refresh() {
        const target = checkpointFamily()
        const origin = vaeFamily(target)
        const route = `${origin}-to-${target}`
        const same = origin === target
        const available = !same && conversions.has(route)

        source.textContent = labels[origin] || origin
        destination.textContent = labels[target] || target
        direction.textContent = same ? "No conversion needed" : `${origin} → ${target}`
        model.disabled = !available
        enabled.disabled = !available
        if (available) {
            const converter = `${route}_interposer-v4.0`
            model.value = converter
            status.textContent = `Auto-selected ${converter}.`
        } else {
            model.value = ""
            enabled.checked = false
            status.textContent = same
                ? "Custom VAE already matches the checkpoint; encode conversion is not needed."
                : `city96 v4.0 does not provide ${route}.`
        }
        saveState()
    }

    const state = readJSON(STATE_KEY)
    const legacy = readJSON(LEGACY_STATE_KEY)
    enabled.checked = Boolean(state.enabled ?? legacy.encodeInterposerEnabled)
    if (state.model) model.value = state.model

    enabled.addEventListener("change", saveState)
    model.addEventListener("change", saveState)
    document.getElementById("stable_diffusion_model")?.addEventListener("change", refresh)
    document.getElementById("vae_model")?.addEventListener("change", refresh)
    document.addEventListener("refreshModels", refresh)

    PLUGINS.TASK_CREATE.push(function (event) {
        if (!enabled.checked || !model.value) return
        event.reqBody.latent_interposer_encode_enabled = true
        event.reqBody.latent_interposer_encode_model = model.value
        saveState()
    })

    refresh()
})()
