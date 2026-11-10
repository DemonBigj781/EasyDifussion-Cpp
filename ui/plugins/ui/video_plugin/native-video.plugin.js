// Native stable-diffusion.cpp video generation with EasyCache and TeaCache.

;(function () {
    "use strict"
    if (window.__nativeVideoPluginLoaded) return
    window.__nativeVideoPluginLoaded = true

    const STATE_KEY = "easy-diffusion-native-video-v1"
    const LTX_VERSIONS = {ltx2: "2.0", ltx23: "2.3"}
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
    const videoToggle = byId("native-video-enabled")
    const cache = byId("native-video-cache")
    const threshold = byId("native-video-threshold")
    const endInput = byId("native-video-end-input")
    const endPreview = byId("native-video-end-preview")
    const motionModule = new ModelDropdown(byId("native-video-motion-module"), "motion-module", "None")
    const audioVae = new ModelDropdown(byId("native-video-audio-vae"), "vae", "Auto-detect matching LTX-2 audio VAE")
    const connectors = new ModelDropdown(byId("native-video-connectors"), "text-encoder", "Auto-detect matching LTX-2 connectors")

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
    let manualVideoEnabled = state.videoEnabled ?? Boolean(state.motionModule)
    const companions = Object.assign(Object.create(null), state.companions || {})
    let imageCompanions = state.imageCompanions || {vae: "", textEncoders: []}
    let activeModel = ""
    let activeFamily = ""
    let selectionInitialized = false
    let updatingCompanions = false

    function selectedModel() {
        return stableDiffusionModelField.value || ""
    }

    function modelMetadata(model) {
        return modelsDB?.["stable-diffusion"]?.[model] || modelsDB?.video?.[model] || {}
    }

    function isAnimateDiffBase(model) {
        const metadata = modelMetadata(model)
        const tags = metadata.tags || []
        if (tags.length > 1) return tags.some(tag => /^(sd_v1_(base|inpainting)|instruct_pix2pix|sd_xl_(base|inpainting)|sdxl|sd15|sd1\.5)$/.test(tag))
        return /(?:^|[/\\_ .-])(?:sd15|sd1\.5|sdxl|sd[_-]?xl)(?:[/\\_ .-]|$)/i.test(model)
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
        if (family === "ltx2" || /ltx[-_ ]?2(?:[._-]0)?(?:[-_ ](?:19b|dev|distilled)|$)/i.test(hint)) return "ltx2"
        // Unknown LTX-2 versions must not inherit either resource preset.
        if (/ltx[-_ ]?2(?:[_. /-]|$)/i.test(hint)) return "video"
        if (/ltxv|ltx[-_ ]?video/i.test(hint)) return "ltx"
        if (/svd|stable.video.diffusion/i.test(hint)) return "svd"
        return metadata?.tags?.[0] === "video" ? "video" : ""
    }

    function installedModels(type) {
        return Object.values(modelsDB?.[type] || {}).filter(model => model.installed !== false)
    }

    function preferredModel(type, pattern, compatible = () => true) {
        const preferences = ["q4_0", "q4", "fp8_e4m3fn", "fp8", "fp16"]
        const rank = model => {
            const index = preferences.findIndex(token => model.model.toLowerCase().includes(token))
            return index < 0 ? preferences.length : index
        }
        return installedModels(type).filter(model => pattern.test(model.model) && compatible(model.model))
            .sort((a, b) => rank(a) - rank(b) || a.model.localeCompare(b.model))[0]?.model || ""
    }

    function automaticCompanions(family, model) {
        let vae = "", encoder = ""
        if (LTX_VERSIONS[family]) {
            const version = family === "ltx2" ? /(?:^|[/\\])ltx[-_ ]?2(?:[._-]0)?[_-]/i : /ltx[-_ ]?2[._-]3[_-]/i
            const variant = /distill/i.test(model) ? /distill/i : /(?:^|[_-])dev(?:[_-]|$)/i
            const matching = name => version.test(name)
            const matchingVariant = name => matching(name) && variant.test(name)
            return {
                vae: preferredModel("vae", /video[_-]vae/i, family === "ltx2" ? matching : matchingVariant),
                textEncoders: [preferredModel("text-encoder", /(?:^|[/\\])gemma[-_ ]?3[-_ ]12b[-_ ]it/i)].filter(Boolean),
                audioVae: preferredModel("vae", /audio[_-]vae/i, family === "ltx2" ? matching : matchingVariant),
                connectors: preferredModel("text-encoder", /embeddings[_-]connectors?/i, matchingVariant),
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
            ...(LTX_VERSIONS[activeFamily] ? {audioVae: audioVae.value || "", connectors: connectors.value || ""} : {})}
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
            motionModule: motionModule.value || "",
            steps: byId("native-video-steps").value,
            frames: byId("native-video-frames").value,
            fps: byId("native-video-fps").value,
            cache: cache.value,
            threshold: threshold.value,
            start: byId("native-video-cache-start").value,
            end: byId("native-video-cache-end").value,
            videoEnabled: manualVideoEnabled,
        }))
    }

    function updateVideoToggle() {
        const nativeVideo = Boolean(activeFamily)
        const allowedBase = isAnimateDiffBase(activeModel)
        byId("native-video-motion-module").disabled = nativeVideo
        videoToggle.disabled = nativeVideo || !allowedBase
        videoToggle.checked = nativeVideo || (allowedBase && manualVideoEnabled)
        videoToggle.title = nativeVideo
            ? "Enabled automatically for native video checkpoints."
            : allowedBase
            ? "Available for SD 1.5 and SDXL checkpoints; AnimateDiff also needs a motion module."
            : "Video mode is available only for SD 1.5 or SDXL checkpoints."
    }

    function updateCacheUI() {
        const animateDiff = !activeFamily && Boolean(motionModule.value)
        const frames = byId("native-video-frames")
        const frameLimit = animateDiff ? 32 : 513
        frames.max = String(frameLimit)
        if (Number(frames.value) > frameLimit) frames.value = String(frameLimit)
        const active = cache.value !== "disabled"
        threshold.disabled = !active
        byId("native-video-cache-start").disabled = !active
        byId("native-video-cache-end").disabled = !active
        const defaults = cache.value === "teacache"
            ? "TeaCache defaults: 0.20 for AnimateDiff/Wan, 0.05 for LTX. Approximate reuse may change quality."
            : (cache.value === "easycache" ? "EasyCache default threshold: 0.20." : "Caching is disabled; every denoising step is exact.")
        const missing = []
        if (["mochi", "wan", "ltx2", "ltx23"].includes(activeFamily) && !vaeModelField.value) missing.push("VAE")
        if (["mochi", "wan", "ltx", "ltx2", "ltx23"].includes(activeFamily) && !textEncoderModelField.modelNames.length) missing.push("text encoder")
        if (LTX_VERSIONS[activeFamily] && !audioVae.value) missing.push("audio VAE")
        if (LTX_VERSIONS[activeFamily] && !connectors.value) missing.push("embedding connectors")
        const companionHint = activeFamily === "video"
            ? " No automatic companion profile for this video architecture; configure its resources under Options."
            : missing.length
            ? ` Missing installed ${missing.join(" and ")}; select the required resources under Options before generating.`
            : " Uses the Model, VAE, and Text Encoder selections under Options."
        const familyHint = animateDiff ? " AnimateDiff / SDXL beta is limited to 32 frames by the motion module positional encoding; 16 frames is the trained regime."
            : activeFamily === "mochi" ? " Mochi is text-to-video only."
            : activeFamily === "ltx" ? " Original LTX-Video uses its embedded VAE; distilled checkpoints use 8 steps and CFG 1."
            : LTX_VERSIONS[activeFamily] ? ` LTX-${LTX_VERSIONS[activeFamily]} uses Gemma 3 12B IT plus matching video/audio VAEs and embedding connectors. The current output is video frames only; generated audio is not exported.` : ""
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
        // Keep the panel available for still-image checkpoints so users can
        // select an AnimateDiff motion module and switch that checkpoint to
        // video generation. Native video-family-specific fields remain
        // collapsed or disabled through their own state updates.
        panel.classList.remove("displayNone")
        byId("native-video-ltx23-resources").classList.toggle("displayNone", !LTX_VERSIONS[family])
        updateVideoToggle()
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

    byId("native-video-steps").value = state.steps ?? "20"
    byId("native-video-frames").value = state.frames ?? "25"
    byId("native-video-fps").value = state.fps ?? "8"
    cache.value = state.cache ?? "disabled"
    threshold.value = state.threshold ?? ""
    byId("native-video-cache-start").value = state.start ?? "15"
    byId("native-video-cache-end").value = state.end ?? "95"
    videoToggle.checked = state.videoEnabled === true
    motionModule.value = state.motionModule ?? ""
    panel.querySelectorAll("input, select")
        .forEach((input) => input.addEventListener("change", saveState))
    modelInput.addEventListener("change", updateModel)
    motionModule.addEventListener("change", () => {
        manualVideoEnabled = Boolean(motionModule.value)
        updateModel()
    })
    videoToggle.addEventListener("change", () => {
        manualVideoEnabled = videoToggle.checked
        updateVideoToggle()
        updateCacheUI()
    })
    for (const id of ["vae_model", "text_encoder_model", "native-video-audio-vae", "native-video-connectors"]) byId(id).addEventListener("change", () => {
        if (!updatingCompanions) updateCacheUI()
    })
    // MultiModelSelector restores its entries on the next tick after a scan.
    document.addEventListener("refreshModels", () => setTimeout(() => {
        if (!motionModule.value && state.motionModule) motionModule.value = state.motionModule
        updateModel()
    }, 0))
    cache.addEventListener("change", updateCacheUI)

    function prepareVideoRequest(event) {
        const model = event.reqBody.use_stable_diffusion_model
        const family = videoFamily(model)
        if (!("video_mode" in event.reqBody)) {
            event.reqBody.video_mode = Boolean(family || (model === selectedModel() && videoToggle.checked))
            if (event.reqBody.video_mode && !family)
                event.reqBody.motion_module_model ??= model === selectedModel() ? motionModule.value : ""
        }
        if (!event.reqBody.video_mode) return false
        const animateDiff = Boolean(event.reqBody.motion_module_model)
        if (!family && !isAnimateDiffBase(model)) {
            throw new Error("Video mode is available only for SD 1.5 or SDXL checkpoints.")
        }
        if (!family && !animateDiff) {
            throw new Error("Select an AnimateDiff motion module before enabling video generation.")
        }
        if (animateDiff && family) {
            throw new Error("AnimateDiff requires an SD 1.5 or SDXL checkpoint, not a native video checkpoint.")
        }
        if (family) {
            event.reqBody.tome_ratio = 0
            event.reqBody.negative_guidance_min_sigma = 0
        }
        if (animateDiff) {
            event.reqBody.video_frames ??= Math.round(clamp(byId("native-video-frames").value, 1, 32, 16))
        }
        event.reqBody.video_steps ??= Math.round(clamp(byId("native-video-steps").value, 1, 200, 20))
        if (LTX_VERSIONS[family]) {
            const resources = model === activeModel ? currentCompanions() : companions[model] || automaticCompanions(family, model)
            event.reqBody.audio_vae_model ??= resources.audioVae || null
            event.reqBody.embeddings_connectors_model ??= resources.connectors || null
            const required = {"video VAE": event.reqBody.use_vae_model, "Gemma 3 text encoder": event.reqBody.use_text_encoder_model,
                "audio VAE": event.reqBody.audio_vae_model, "embedding connectors": event.reqBody.embeddings_connectors_model}
            const missing = Object.entries(required).filter(([, value]) => !value || (Array.isArray(value) && !value.length)).map(([label]) => label)
            if (missing.length) throw new Error(`LTX-${LTX_VERSIONS[family]} is missing ${missing.join(" and ")}; select its resources before generating.`)
            event.reqBody.sampler_name = "euler"
            event.reqBody.scheduler_name = "ltx2"
            if (model.toLowerCase().includes("distilled")) {
                event.reqBody.video_steps = 8
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
                event.reqBody.video_steps = 8
                event.reqBody.guidance_scale = 1.0
            }
        }
        event.reqBody.video_frames ??= Math.round(clamp(byId("native-video-frames").value, 1, 513, 25))
        event.reqBody.fps ??= Math.round(clamp(byId("native-video-fps").value, 1, 60, 8))
        event.reqBody.cache_mode ??= cache.value
        if (event.reqBody.negative_guidance_min_sigma > 0 && event.reqBody.cache_mode !== "disabled")
            throw new Error("Negative guidance minimum sigma requires video caching to be disabled.")
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
