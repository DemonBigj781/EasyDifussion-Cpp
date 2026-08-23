;(function () {
    "use strict"

    const STATE_KEY = "easy-diffusion-latent-interposer-decode-v1"
    const LEGACY_STATE_KEY = "easy-diffusion-controlled-image-generation-v1"
    const outputSettings = document.querySelector("#output-settings")
    if (!outputSettings?.parentNode || typeof ModelDropdown !== "function" || typeof PLUGINS !== "object" || document.querySelector("#sdkit3-decode-interpose-panel")) return

    const panel = document.createElement("div")
    panel.id = "sdkit3-decode-interpose-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel"
    panel.innerHTML = `
        <h4 class="collapsible">Decode Interpose <small>model → VAE</small></h4>
        <div class="collapsible-content controlled-generation-content">
            <label><input id="latent-interposer-decode-enabled" type="checkbox"> Convert sampled checkpoint latents back into the VAE standard</label>
            <div class="controlled-generation-grid">
                <label>Detected model standard</label><span id="latent-interposer-decode-source">—</span>
                <label>Detected VAE standard</label><span id="latent-interposer-decode-destination">—</span>
                <label>Direction</label><strong id="latent-interposer-decode-direction">—</strong>
                <label>Conversion model</label><input id="latent-interposer-decode-model" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="">
            </div>
            <small id="latent-interposer-decode-status">Decode reverses the encode direction: Model → Custom VAE.</small>
        </div>`

    const anchor = document.querySelector("#sdkit3-encode-interpose-panel") || document.querySelector("#sdkit3-lllite-panel") || document.querySelector("#sdkit3-controlnet-preprocessor-panel") || document.querySelector("#sdkit3-controlnet-panel") || outputSettings
    anchor.parentNode.insertBefore(panel, anchor.nextSibling)

    const style = document.createElement("style")
    style.textContent = `
        #sdkit3-decode-interpose-panel { margin-top: 10px; }
        #sdkit3-decode-interpose-panel h4 { cursor: pointer; }
        #sdkit3-decode-interpose-panel h4 small { float: right; }
        #sdkit3-decode-interpose-panel .controlled-generation-content { padding-top: 8px; }
        #sdkit3-decode-interpose-panel .controlled-generation-grid { display: grid; grid-template-columns: minmax(150px, auto) minmax(0, 1fr); gap: 6px 9px; align-items: center; margin: 7px 0; }
    `
    document.head.appendChild(style)

    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const byId = (id) => document.getElementById(id)
    const modelDropdown = new ModelDropdown(byId("latent-interposer-decode-model"), "latent-interposer", "None")
    const conversions = new Set(["v1-to-xl", "v1-to-v3", "xl-to-v1", "xl-to-v3", "v3-to-v1", "v3-to-xl", "fx-to-v1", "fx-to-xl", "fx-to-v3"])
    const labels = { v1: "Stable Diffusion v1.x", xl: "SDXL", v3: "Stable Diffusion 3", fx: "Flux.1", ca: "Stable Cascade" }

    function modelTags(type, name) {
        return typeof modelsDB !== "undefined" && name ? modelsDB?.[type]?.[name]?.tags || [] : []
    }

    function checkpointFamily() {
        const path = document.querySelector("#stable_diffusion_model")?.dataset.path || ""
        const tags = modelTags("stable-diffusion", path)
        if (tags.some((tag) => tag.startsWith("stable_cascade"))) return "ca"
        if (tags.some((tag) => tag.startsWith("flux") || tag === "chroma")) return "fx"
        if (tags.some((tag) => tag.startsWith("sd_v3"))) return "v3"
        if (tags.some((tag) => tag.startsWith("sd_xl") || tag.startsWith("playground_v2_5"))) return "xl"
        const fallback = path.toLowerCase()
        if (fallback.includes("cascade")) return "ca"
        if (fallback.includes("flux")) return "fx"
        if (fallback.includes("sd3") || fallback.includes("stable-diffusion-3")) return "v3"
        if (fallback.includes("sdxl") || fallback.includes("/xl") || fallback.includes("pony") || fallback.includes("illustrious")) return "xl"
        return "v1"
    }

    function vaeFamily(modelFamily) {
        const path = document.querySelector("#vae_model")?.dataset.path || ""
        if (!path || path.toLowerCase() === "none") return modelFamily
        const tags = modelTags("vae", path)
        for (const family of ["v1", "xl", "v3", "fx"]) if (tags.includes(`vae_${family}`)) return family
        const fallback = path.toLowerCase()
        if (fallback.includes("flux") || /(^|\/)ae$/.test(fallback)) return "fx"
        if (fallback.includes("sd3") || fallback.includes("stable-diffusion-3")) return "v3"
        if (fallback.includes("sdxl") || fallback.includes("sd_xl") || fallback.includes("pony") || fallback.includes("illustrious")) return "xl"
        return "v1"
    }

    function writeState() {
        localStorage.setItem(STATE_KEY, JSON.stringify({ enabled: byId("latent-interposer-decode-enabled").checked }))
    }

    function refresh() {
        const source = checkpointFamily()
        const destination = vaeFamily(source)
        const direction = `${source}-to-${destination}`
        const sameFamily = source === destination
        const available = !sameFamily && conversions.has(direction)
        byId("latent-interposer-decode-source").textContent = labels[source] || source
        byId("latent-interposer-decode-destination").textContent = labels[destination] || destination
        byId("latent-interposer-decode-direction").textContent = sameFamily ? "No conversion needed" : `${source} → ${destination}`
        modelDropdown.value = available ? `${direction}_interposer-v4.0` : ""
        modelDropdown.disabled = !available
        byId("latent-interposer-decode-enabled").disabled = !available
        if (!available) byId("latent-interposer-decode-enabled").checked = false
        byId("latent-interposer-decode-status").textContent = sameFamily
            ? "The selected VAE already matches the checkpoint; decode interpose is not needed."
            : available ? `Auto-selected ${direction}_interposer-v4.0.` : `city96 v4.0 does not provide ${direction}.`
        writeState()
    }

    let saved = {}
    try { saved = JSON.parse(localStorage.getItem(STATE_KEY) || localStorage.getItem(LEGACY_STATE_KEY) || "{}") } catch (_) {}
    byId("latent-interposer-decode-enabled").checked = Boolean(saved.enabled ?? saved.decodeInterposerEnabled)
    byId("latent-interposer-decode-enabled").addEventListener("change", writeState)
    document.querySelector("#stable_diffusion_model")?.addEventListener("change", refresh)
    document.querySelector("#vae_model")?.addEventListener("change", refresh)
    document.addEventListener("refreshModels", refresh)
    refresh()

    PLUGINS.TASK_CREATE.push(function (event) {
        if (!byId("latent-interposer-decode-enabled").checked) return
        event.reqBody.latent_interposer_decode_enabled = true
        event.reqBody.latent_interposer_decode_model = modelDropdown.value
        writeState()
    })
})()
