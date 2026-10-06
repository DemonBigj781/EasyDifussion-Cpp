// Native stable-diffusion.cpp video generation with EasyCache and TeaCache.

;(function () {
    "use strict"
    if (window.__nativeVideoPluginLoaded) return
    window.__nativeVideoPluginLoaded = true

    const STATE_KEY = "easy-diffusion-native-video-v1"
    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode || typeof PLUGINS !== "object" || typeof window.SD?.VideoTask !== "function") {
        console.error("Native Video plugin: required Easy Diffusion APIs were not found")
        return
    }

    const panel = document.createElement("div")
    panel.id = "sdkit3-native-video-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel gated-feature"
    panel.dataset.featureKeys = "backend_sdkit3"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/video_plugin/native-video.plugin.html")
    const ipPanel = document.getElementById("sdkit3-ip-adapter-panel")
    if (ipPanel) ipPanel.after(panel)
    else editor.after(panel)
    window.orderSdkitSettingsPanels?.()
    setTimeout(() => window.orderSdkitSettingsPanels?.(), 250)

    if (!document.getElementById("native-video-style")) {
        const style = document.createElement("style")
        style.id = "native-video-style"
        style.textContent = `
            .native-video-grid {
                display: grid;
                grid-template-columns: minmax(130px, auto) minmax(0, 1fr);
                gap: 7px 10px;
                align-items: center;
            }
            .native-video-grid input[type="number"] { width: 82px; }
            .native-video-end-preview { max-width: 220px; max-height: 180px; margin: 10px auto; }
            @media (max-width: 700px) { .native-video-grid { grid-template-columns: 1fr; } }
        `
        document.head.appendChild(style)
    }
    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const byId = (id) => document.getElementById(id)
    const modelInput = byId("stable_diffusion_model")
    const cache = byId("native-video-cache")
    const threshold = byId("native-video-threshold")
    const endInput = byId("native-video-end-input")
    const endPreview = byId("native-video-end-preview")
    const audioVae = new ModelDropdown(byId("native-video-audio-vae"), "vae", "Auto-detect LTX-2.3 audio VAE")
    const connectors = new ModelDropdown(byId("native-video-connectors"), "text-encoder", "Auto-detect LTX-2.3 connectors")

    function clamp(value, minimum, maximum, fallback) {
        const number = Number(value)
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback
    }

    function readState() {
        try {
            const value = JSON.parse(localStorage.getItem(STATE_KEY) || "{}")
            return value && typeof value === "object" && !Array.isArray(value) ? value : {}
        }
        catch (_) { return {} }
    }

    const state = readState()
    const companions = Object.assign(Object.create(null), state.companions || {})
    let imageCompanions = state.imageCompanions || {vae: "", textEncoders: []}
    let activeModel = ""
    let activeFamily = ""
    let selectionInitialized = false
    let updatingCompanions = false

    function selectedModel() {
        return stableDiffusionModelField.value || ""
    }

    function videoFamily(model) {
        if (!model || typeof modelsDB === "undefined") return ""
        const metadata = modelsDB?.["stable-diffusion"]?.[model] || modelsDB?.video?.[model]
        const tags = metadata?.tags || []
        const family = tags.find(tag => /^(mochi_|wan_|ltx_video|ltx2|svd|stable.video.diffusion)/i.test(tag))
        // Prefer architecture metadata, including for renamed checkpoints.
        // Filename hints are only for models whose architecture is unknown.
        const hint = family || (tags.length < 2 ? model : "")
        if (/(?:^|[/\\_-])mochi/i.test(hint)) return "mochi"
        if (/(?:^|[/\\_-])wan(?:[0-9_.-]|$)/i.test(hint)) return "wan"
        if (/ltx[-_ ]?2[._-]3(?:[_. /-]|$)/i.test(hint)) return "ltx23"
        // Other LTX-2 versions must not inherit LTX-2.3's resource preset.
        if (/ltx[-_ ]?2(?:[_. /-]|$)/i.test(hint)) return "video"
        if (/ltxv|ltx[-_ ]?video/i.test(hint)) return "ltx"
        if (/svd|stable.video.diffusion/i.test(hint)) return "svd"
        return metadata?.tags?.[0] === "video" ? "video" : ""
    }

    function installedModels(type) {
        return Object.values(modelsDB?.[type] || {}).filter(model => model.installed !== false)
    }

    function preferredModel(type, pattern) {
        const preferences = ["q4_0", "q4", "fp8_e4m3fn", "fp8", "fp16"]
        const rank = model => {
            const index = preferences.findIndex(token => model.model.toLowerCase().includes(token))
            return index < 0 ? preferences.length : index
        }
        return installedModels(type).filter(model => pattern.test(model.model))
            .sort((a, b) => rank(a) - rank(b) || a.model.localeCompare(b.model))[0]?.model || ""
    }

    function automaticCompanions(family, model) {
        let vae = "", encoder = ""
        if (family === "ltx23") {
            return {
                vae: preferredModel("vae", /ltx[-_ ]?2[._-]3.*video[_-]vae/i),
                textEncoders: [preferredModel("text-encoder", /(?:^|[/\\])gemma[-_ ]?3[-_ ]12b[-_ ]it/i)].filter(Boolean),
                audioVae: preferredModel("vae", /ltx[-_ ]?2[._-]3.*audio[_-]vae/i),
                connectors: preferredModel("text-encoder", /ltx[-_ ]?2[._-]3.*embeddings[_-]connectors/i),
            }
        }
        if (family === "mochi") {
            vae = preferredModel("vae", /mochi/i)
            encoder = preferredModel("text-encoder", /(?:^|[/\\])t5(?:xxl|_xxl)/i)
        } else if (family === "wan") {
            const is5b = /(?:^|[^0-9])5b(?:[^a-z0-9]|$)/i.test(model)
            vae = preferredModel("vae", is5b ? /wan.*2[._]?2/i : /wan.*2[._]?1/i)
            encoder = preferredModel("text-encoder", /(?:^|[/\\])umt5[_-]?xxl/i)
        } else if (family === "ltx") {
            // Official original LTX-Video checkpoints include their VAE.
            encoder = preferredModel("text-encoder", /(?:^|[/\\])t5(?:xxl|_xxl)/i)
        }
        return {vae, textEncoders: encoder ? [encoder] : []}
    }

    function currentCompanions() {
        return {vae: vaeModelField.value || "", textEncoders: [...textEncoderModelField.modelNames],
            ...(activeFamily === "ltx23" ? {audioVae: audioVae.value || "", connectors: connectors.value || ""} : {})}
    }

    function setCompanions(value) {
        updatingCompanions = true
        try {
            vaeModelField.value = value.vae || ""
            textEncoderModelField.modelNames = value.textEncoders || []
            audioVae.value = value.audioVae || ""
            connectors.value = value.connectors || ""
        } finally { updatingCompanions = false }
    }

    function restoreVideoCompanions() {
        const defaults = automaticCompanions(activeFamily, activeModel)
        const saved = companions[activeModel] || {}
        const available = (type, name) => installedModels(type).some(model => model.model === name)
        // Accept the former panel's single text-encoder preference as well.
        const encoders = Array.isArray(saved.textEncoders) ? saved.textEncoders : [saved.textEncoder].filter(Boolean)
        setCompanions({
            vae: available("vae", saved.vae) ? saved.vae : defaults.vae,
            textEncoders: encoders.length && encoders.every(name => available("text-encoder", name))
                ? encoders : defaults.textEncoders,
            audioVae: available("vae", saved.audioVae) ? saved.audioVae : defaults.audioVae,
            connectors: available("text-encoder", saved.connectors) ? saved.connectors : defaults.connectors,
        })
    }

    function saveState() {
        if (updatingCompanions) return
        if (activeFamily) companions[activeModel] = currentCompanions()
        localStorage.setItem(STATE_KEY, JSON.stringify({
            companions,
            imageCompanions,
            frames: byId("native-video-frames").value,
            fps: byId("native-video-fps").value,
            cache: cache.value,
            threshold: threshold.value,
            start: byId("native-video-cache-start").value,
            end: byId("native-video-cache-end").value,
        }))
    }

    function updateCacheUI() {
        const active = cache.value !== "disabled"
        threshold.disabled = !active
        byId("native-video-cache-start").disabled = !active
        byId("native-video-cache-end").disabled = !active
        const defaults = cache.value === "teacache"
            ? "TeaCache defaults: 0.05 for LTX, 0.20 for Wan."
            : (cache.value === "easycache" ? "EasyCache default threshold: 0.20." : "Caching is disabled; every denoising step is exact.")
        const missing = []
        if (["mochi", "wan", "ltx23"].includes(activeFamily) && !vaeModelField.value) missing.push("VAE")
        if (["mochi", "wan", "ltx", "ltx23"].includes(activeFamily) && !textEncoderModelField.modelNames.length) missing.push("text encoder")
        if (activeFamily === "ltx23" && !audioVae.value) missing.push("audio VAE")
        if (activeFamily === "ltx23" && !connectors.value) missing.push("embedding connectors")
        const companionHint = activeFamily === "video"
            ? " No automatic companion profile for this video architecture; configure its resources under Options."
            : missing.length
            ? ` Missing installed ${missing.join(" and ")}; select the required resources under Options before generating.`
            : " Uses the Model, VAE, and Text Encoder selections under Options."
        const familyHint = activeFamily === "mochi" ? " Mochi is text-to-video only."
            : activeFamily === "ltx" ? " Original LTX-Video uses its embedded VAE; distilled checkpoints use 8 steps and CFG 1."
            : activeFamily === "ltx23" ? " LTX-2.3 uses Gemma 3 12B IT plus video/audio VAEs and embedding connectors. The current output is video frames only; generated audio is not exported." : ""
        byId("native-video-status").textContent = `${defaults}${companionHint}${familyHint} Frames are returned as a numbered strip while MP4 encoding is still being added.`
        saveState()
    }

    function updateModel() {
        if (typeof modelsDB === "undefined" || !modelsDB) return
        const model = selectedModel(), family = videoFamily(model)
        if (model !== activeModel || family !== activeFamily) {
            if (activeFamily) companions[activeModel] = currentCompanions()
            else if (selectionInitialized && family) imageCompanions = currentCompanions()
            const wasVideo = Boolean(activeFamily)
            activeModel = model
            activeFamily = family
            if (family) restoreVideoCompanions()
            else if (wasVideo) setCompanions(imageCompanions)
        } else if (family) {
            activeFamily = family
            restoreVideoCompanions()
        }
        selectionInitialized = true
        panel.classList.toggle("displayNone", !family)
        byId("native-video-ltx23-resources").classList.toggle("displayNone", family !== "ltx23")
        updateCacheUI()
    }

    endInput.addEventListener("change", () => {
        const file = endInput.files?.[0]
        if (!file?.type.startsWith("image/")) return
        const reader = new FileReader()
        reader.addEventListener("load", () => {
            endPreview.src = reader.result
            endPreview.classList.remove("displayNone")
        })
        reader.readAsDataURL(file)
    })
    byId("native-video-end-clear").addEventListener("click", () => {
        endInput.value = ""
        endPreview.removeAttribute("src")
        endPreview.classList.add("displayNone")
    })

    byId("native-video-frames").value = state.frames ?? "25"
    byId("native-video-fps").value = state.fps ?? "8"
    cache.value = state.cache ?? "disabled"
    threshold.value = state.threshold ?? ""
    byId("native-video-cache-start").value = state.start ?? "15"
    byId("native-video-cache-end").value = state.end ?? "95"
    panel.querySelectorAll("input, select")
        .forEach((input) => input.addEventListener("change", saveState))
    modelInput.addEventListener("change", updateModel)
    for (const id of ["vae_model", "text_encoder_model", "native-video-audio-vae", "native-video-connectors"]) byId(id).addEventListener("change", () => {
        if (!updatingCompanions) updateCacheUI()
    })
    // MultiModelSelector restores its entries on the next tick after a scan.
    document.addEventListener("refreshModels", () => setTimeout(updateModel, 0))
    cache.addEventListener("change", updateCacheUI)

    function prepareVideoRequest(event) {
        const model = event.reqBody.use_stable_diffusion_model
        const family = videoFamily(model)
        if (!family) return false
        if (family === "ltx23") {
            const resources = model === activeModel ? currentCompanions() : companions[model] || automaticCompanions(family, model)
            event.reqBody.audio_vae_model ??= resources.audioVae || null
            event.reqBody.embeddings_connectors_model ??= resources.connectors || null
            const required = {"video VAE": event.reqBody.use_vae_model, "Gemma 3 text encoder": event.reqBody.use_text_encoder_model,
                "audio VAE": event.reqBody.audio_vae_model, "embedding connectors": event.reqBody.embeddings_connectors_model}
            const missing = Object.entries(required).filter(([, value]) => !value || (Array.isArray(value) && !value.length)).map(([label]) => label)
            if (missing.length) throw new Error(`LTX-2.3 is missing ${missing.join(" and ")}; select its resources before generating.`)
            event.reqBody.sampler_name = "euler"
            event.reqBody.scheduler_name = "ltx2"
            if (model.toLowerCase().includes("distilled")) {
                event.reqBody.num_inference_steps = 8
                event.reqBody.guidance_scale = 1.0
            }
        }
        if (family === "mochi") {
            event.reqBody.sampler_name = "euler"
            event.reqBody.scheduler_name = "mochi"
        } else if (family === "ltx") {
            event.reqBody.sampler_name = "euler"
            event.reqBody.scheduler_name = "linear_quadratic"
            if (model.toLowerCase().includes("distilled")) {
                event.reqBody.num_inference_steps = 8
                event.reqBody.guidance_scale = 1.0
            }
        }
        event.reqBody.video_frames ??= Math.round(clamp(byId("native-video-frames").value, 1, 513, 25))
        event.reqBody.fps ??= Math.round(clamp(byId("native-video-fps").value, 1, 60, 8))
        event.reqBody.cache_mode ??= cache.value
        event.reqBody.cache_start_percent ??= clamp(byId("native-video-cache-start").value, 0, 100, 15)
        event.reqBody.cache_end_percent ??= clamp(byId("native-video-cache-end").value, 0, 100, 95)
        const thresholdValue = Number(threshold.value)
        if (!("cache_threshold" in event.reqBody)) event.reqBody.cache_threshold = threshold.value !== "" && Number.isFinite(thresholdValue)
            ? Math.max(0, thresholdValue) : null
        if (!("end_image" in event.reqBody)) {
            event.reqBody.end_image = /^data:image\//.test(endPreview.getAttribute("src") || "") ? endPreview.src : null
        }
        return true
    }

    PLUGINS.TASK_BUILD.push(function (event) {
        if (prepareVideoRequest(event)) event.reqBody.backend_assignment = window.NativeDeviceRouting?.assignmentFor("video") || ""
    })
    PLUGINS.TASK_CREATE.push(function (event) {
        if (!prepareVideoRequest(event)) return
        event.reqBody.backend_assignment ??= window.NativeDeviceRouting?.assignmentFor("video") || ""
        // The video task uses the normal queue, progress stream, selected
        // checkpoint, prompt, seed, dimensions, sampler, and Initial Image.
        event.instance = new SD.VideoTask(event.reqBody)
        saveState()
    })

    updateModel()
})()
