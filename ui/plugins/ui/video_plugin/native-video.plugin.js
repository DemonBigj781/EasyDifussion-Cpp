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
    const enabled = byId("native-video-enabled")
    const modelInput = byId("stable_diffusion_model")
    const vaeInput = byId("native-video-vae")
    const textEncoderInput = byId("native-video-text-encoder")
    const cache = byId("native-video-cache")
    const threshold = byId("native-video-threshold")
    const endInput = byId("native-video-end-input")
    const endPreview = byId("native-video-end-preview")

    function clamp(value, minimum, maximum, fallback) {
        const number = Number(value)
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback
    }

    function readState() {
        try { return JSON.parse(localStorage.getItem(STATE_KEY) || "{}") }
        catch (_) { return {} }
    }

    const state = readState()
    const companions = state.companions && typeof state.companions === "object" ? state.companions : {}

    function selectedModel() {
        return modelInput?.dataset.path || ""
    }

    function saveState() {
        const model = selectedModel()
        if (model) {
            companions[model] = {
                vae: videoVae.value,
                textEncoder: videoTextEncoder.value,
            }
        }
        localStorage.setItem(STATE_KEY, JSON.stringify({
            enabled: enabled.checked,
            companions,
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
        const companionHint = selectedModel().toLowerCase().includes("mochi")
            ? " Mochi auto-detects its sibling VAE and T5 XXL when these fields are blank; it is text-to-video only."
            : " Select the video checkpoint under Options; each checkpoint keeps its own VAE and text encoder selections."
        byId("native-video-status").textContent = `${defaults}${companionHint} Frames are returned as a numbered strip while MP4 encoding is still being added.`
        saveState()
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

    const savedCompanions = companions[selectedModel()] || {}
    vaeInput.dataset.path = typeof savedCompanions.vae === "string" ? savedCompanions.vae : ""
    textEncoderInput.dataset.path = typeof savedCompanions.textEncoder === "string"
        ? savedCompanions.textEncoder
        : ""
    const videoVae = new ModelDropdown(vaeInput, "vae", "Auto-detect / embedded")
    const videoTextEncoder = new ModelDropdown(textEncoderInput, "text-encoder", "Auto-detect / embedded")
    enabled.checked = Boolean(state.enabled)
    byId("native-video-frames").value = state.frames ?? "25"
    byId("native-video-fps").value = state.fps ?? "8"
    cache.value = state.cache ?? "disabled"
    threshold.value = state.threshold ?? ""
    byId("native-video-cache-start").value = state.start ?? "15"
    byId("native-video-cache-end").value = state.end ?? "95"
    panel.querySelectorAll("input, select").forEach((input) => input.addEventListener("change", saveState))
    let previousVideoModel = selectedModel()
    modelInput.addEventListener("change", () => {
        if (previousVideoModel) {
            companions[previousVideoModel] = {
                vae: videoVae.value,
                textEncoder: videoTextEncoder.value,
            }
        }
        previousVideoModel = selectedModel()
        const selectedCompanions = companions[previousVideoModel] || {}
        videoVae.value = selectedCompanions.vae || ""
        videoTextEncoder.value = selectedCompanions.textEncoder || ""
        updateCacheUI()
    })
    cache.addEventListener("change", updateCacheUI)

    PLUGINS.TASK_BUILD.push(function (event) {
        if (!enabled.checked) return
        // Video uses the checkpoint selected in the shared Options panel, but
        // keeps its companion VAE and text encoder independent from images.
        event.reqBody.use_vae_model = videoVae.value || null
        event.reqBody.use_text_encoder_model = videoTextEncoder.value || null
        if ((event.reqBody.use_stable_diffusion_model || selectedModel()).toLowerCase().includes("mochi")) {
            event.reqBody.sampler_name = "euler"
            event.reqBody.scheduler_name = "mochi"
        }
    })

    PLUGINS.TASK_CREATE.push(function (event) {
        if (!enabled.checked) return
        event.reqBody.video_frames = Math.round(clamp(byId("native-video-frames").value, 1, 513, 25))
        event.reqBody.fps = Math.round(clamp(byId("native-video-fps").value, 1, 60, 8))
        event.reqBody.cache_mode = cache.value
        event.reqBody.cache_start_percent = clamp(byId("native-video-cache-start").value, 0, 100, 15)
        event.reqBody.cache_end_percent = clamp(byId("native-video-cache-end").value, 0, 100, 95)
        const thresholdValue = Number(threshold.value)
        event.reqBody.cache_threshold = threshold.value !== "" && Number.isFinite(thresholdValue)
            ? Math.max(0, thresholdValue)
            : null
        if (/^data:image\//.test(endPreview.getAttribute("src") || "")) {
            event.reqBody.end_image = endPreview.src
        }
        // The video task uses the normal queue, progress stream, selected
        // checkpoint, prompt, seed, dimensions, sampler, and Initial Image.
        event.instance = new SD.VideoTask(event.reqBody)
        saveState()
    })

    updateCacheUI()
})()
