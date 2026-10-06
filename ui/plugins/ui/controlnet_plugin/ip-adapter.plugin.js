// RAM-aware native sdkit3 IP-Adapter image conditioning.

;(function () {
    "use strict"
    if (window.__nativeIPAdapterPluginLoaded) return
    window.__nativeIPAdapterPluginLoaded = true

    const STATE_KEY = "easy-diffusion-native-ip-adapter-v1"
    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode || typeof ModelDropdown !== "function" || typeof PLUGINS !== "object") {
        console.error("IP-Adapter plugin: required Easy Diffusion UI APIs were not found")
        return
    }

    const panel = document.createElement("div")
    panel.id = "sdkit3-ip-adapter-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel gated-feature"
    panel.dataset.featureKeys = "backend_sdkit3"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/controlnet_plugin/ip-adapter.plugin.html")

    const lllitePanel = document.getElementById("sdkit3-lllite-panel")
    const controlPanel = document.getElementById("sdkit3-controlnet-panel")
    if (lllitePanel) lllitePanel.after(panel)
    else if (controlPanel) controlPanel.after(panel)
    else editor.after(panel)
    window.orderSdkitSettingsPanels?.()
    setTimeout(() => window.orderSdkitSettingsPanels?.(), 250)

    if (!document.getElementById("native-ip-adapter-style")) {
        const style = document.createElement("style")
        style.id = "native-ip-adapter-style"
        style.textContent = `
            .ip-adapter-grid {
                display: grid;
                grid-template-columns: minmax(120px, auto) minmax(0, 1fr);
                gap: 7px 10px;
                align-items: center;
            }
            .ip-adapter-grid input[type="number"] { width: 78px; }
            .ip-adapter-preview { position: relative; width: min(100%, 260px); min-height: 3rem; margin: 10px auto; }
            .ip-adapter-preview img { display: block; max-width: 100%; max-height: 260px; margin: auto; }
            .ip-adapter-preview .image_clear_btn { position: absolute; right: 3px; top: 3px; }
            @media (max-width: 700px) { .ip-adapter-grid { grid-template-columns: 1fr; } }
        `
        document.head.appendChild(style)
    }
    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const byId = (id) => document.getElementById(id)
    const enabled = byId("ip-adapter-enabled")
    const model = new ModelDropdown(byId("ip-adapter-model"), "ip-adapter", "None")
    const clip = new ModelDropdown(byId("ip-adapter-clip"), "clip-vision", "None")
    const imageInput = byId("ip-adapter-image-input")
    const preview = byId("ip-adapter-image-preview")
    const wrapper = byId("ip-adapter-image-wrapper")
    const status = byId("ip-adapter-status")
    const compatibilityApi = window.NativeIPAdapterCompatibility
    let lastAutomaticModel = ""

    function readState() {
        try { return JSON.parse(localStorage.getItem(STATE_KEY) || "{}") }
        catch (_) { return {} }
    }

    function clamp(value, minimum, maximum, fallback) {
        const number = Number(value)
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback
    }

    function notify(message, isError) {
        if (typeof showToast === "function") showToast(message, 6000, Boolean(isError))
        else if (isError) console.error(message)
        else console.log(message)
    }

    function selectedCheckpointFamily() {
        const field = document.getElementById("stable_diffusion_model")
        const name = field?.dataset.path || field?.value || ""
        const tags = typeof modelsDB === "object" ? (modelsDB?.["stable-diffusion"]?.[name]?.tags || []) : []
        return compatibilityApi.family(name, tags)
    }

    function hasImage() {
        return /^data:image\//.test(preview.getAttribute("src") || "")
    }

    function setImage(source) {
        if (!source || !/^data:image\//.test(source)) return false
        preview.src = source
        enabled.checked = !enabled.disabled
        wrapper.classList.remove("displayNone")
        saveState()
        return true
    }

    function saveState() {
        localStorage.setItem(STATE_KEY, JSON.stringify({
            enabled: enabled.checked,
            model: model.value,
            clip: clip.value,
            strength: byId("ip-adapter-strength").value,
            start: byId("ip-adapter-start").value,
            end: byId("ip-adapter-end").value,
        }))
    }

    function metadataTags(type, name) {
        return Array.isArray(modelsDB?.[type]?.[name]?.tags) ? modelsDB[type][name].tags : []
    }

    function compatibilityFromTags(name) {
        return compatibilityApi.adapter(name, metadataTags("ip-adapter", name))
    }

    function isCompatibleClip(path, compatibility) {
        return compatibilityApi.compatible(path, metadataTags("clip-vision", path), compatibility)
    }

    function firstCompatibleClip(compatibility) {
        if (!compatibility) return ""
        return Array.from(clip.modelElements || [])
            .map((entry) => entry.dataset.path || "")
            .find((path) => path && isCompatibleClip(path, compatibility)) || ""
    }

    function modelPathExists(dropdown, path) {
        return Boolean(path) && Array.from(dropdown.modelElements || [])
            .some((entry) => entry.dataset.path === path)
    }

    function firstAdapterForFamily(family) {
        const paths = Array.from(model.modelElements || [])
            .map((entry) => entry.dataset.path || "")
            .filter((path) => path && compatibilityFromTags(path))
        if (family === "anima") return paths.find(path => compatibilityFromTags(path)?.kind === "anima") || ""
        const familyPattern = family === "sdxl"
            ? /(?:^|[/_-])(?:sdxl|sd_xl|xl)(?:$|[/_.-])/i
            : /(?:^|[/_-])(?:sd15|sd1(?:[._-]?5)?|v1)(?:$|[/_.-])/i
        return paths.find((path) => familyPattern.test(path)) || ""
    }

    function refreshDefaults() {
        const family = selectedCheckpointFamily()
        if (family === "unsupported") {
            enabled.checked = false
            enabled.disabled = true
            status.textContent = "Native IP-Adapter supports Anima, SD 1.x and SDXL checkpoints."
            return
        }
        enabled.disabled = false
        model.setModelPredicate?.(path => compatibilityApi.adapterForFamily(compatibilityFromTags(path), family))
        const automaticModel = firstAdapterForFamily(family)
        const nextModel = !modelPathExists(model, model.value) ||
            !compatibilityApi.adapterForFamily(compatibilityFromTags(model.value), family) ||
            (automaticModel && model.value === lastAutomaticModel)
            ? automaticModel || "" : model.value
        lastAutomaticModel = automaticModel
        // ModelDropdown emits change even for the same value. Only assign a
        // different selection so this change handler cannot call itself forever.
        if (model.value !== nextModel) model.value = nextModel
        const compatibility = compatibilityFromTags(model.value)
        byId("ip-adapter-strength").min = family === "anima" ? "0" : "-10"
        clip.setModelPredicate?.((path) => isCompatibleClip(path, compatibility))
        const compatibleClip = firstCompatibleClip(compatibility)
        enabled.disabled = !compatibility || !compatibleClip
        if (!isCompatibleClip(clip.value, compatibility) && clip.value !== compatibleClip) {
            clip.value = compatibleClip
        }
        if (!compatibility) {
            enabled.checked = false
            status.textContent = "Select a supported Anima, base or Plus adapter. FaceID requires a separate InsightFace path."
        } else if (!compatibleClip) {
            enabled.checked = false
            status.textContent = compatibility.kind === "anima" ? "Install SigLIP2 base patch16-512 for this Anima adapter." : `No compatible CLIP-Vision model is installed. This ${compatibility.kind} adapter requires embedding width ${compatibility.dimension}.`
        } else {
            const source = compatibility.detected ? "tensor metadata" : "its filename"
            status.textContent = compatibility.kind === "anima" ? "Anima · SigLIP2 base patch16-512 · CPU/CUDA only. Strength 0 disables both image attention and adapter LoRA." : `${family === "sdxl" ? "SDXL" : "SD 1.x"} ${compatibility.kind} adapter requires width ${compatibility.dimension}, detected from ${source}. Incompatible vision encoders are hidden.`
        }
        saveState()
    }

    imageInput.addEventListener("change", () => {
        const file = imageInput.files?.[0]
        if (!file?.type.startsWith("image/")) return
        const reader = new FileReader()
        reader.addEventListener("load", () => setImage(reader.result))
        reader.readAsDataURL(file)
    })
    byId("ip-adapter-use-init").addEventListener("click", () => {
        const init = document.getElementById("init_image_preview")?.getAttribute("src") || ""
        if (!setImage(init)) notify("No Initial Image is loaded.", true)
    })
    byId("ip-adapter-image-clear").addEventListener("click", () => {
        imageInput.value = ""
        preview.removeAttribute("src")
        wrapper.classList.add("displayNone")
        enabled.checked = false
        saveState()
    })

    const state = readState()
    enabled.checked = Boolean(state.enabled) && hasImage()
    byId("ip-adapter-strength").value = state.strength ?? "1"
    byId("ip-adapter-start").value = state.start ?? "0"
    byId("ip-adapter-end").value = state.end ?? "100"
    if (state.model) model.value = state.model
    if (state.clip) clip.value = state.clip

    panel.querySelectorAll("input").forEach((input) => input.addEventListener("change", saveState))
    model.addEventListener("change", refreshDefaults)
    clip.addEventListener("change", refreshDefaults)
    document.getElementById("stable_diffusion_model")?.addEventListener("change", refreshDefaults)
    document.addEventListener("refreshModels", refreshDefaults)

    PLUGINS.TASK_CREATE.push(function (event) {
        if (!enabled.checked || enabled.disabled) return
        event.reqBody.ip_adapter_model = model.value
        event.reqBody.ip_adapter_clip_vision = clip.value
        event.reqBody.ip_adapter_image = hasImage() ? preview.src : null
        if (!model.value || !clip.value || !hasImage()) {
            notify("IP-Adapter is enabled, but its adapter, CLIP Vision model, or reference image is missing.", true)
            return
        }
        const compatibility = compatibilityFromTags(model.value)
        const rawStart = clamp(byId("ip-adapter-start").value, 0, 100, 0)
        const rawEnd = clamp(byId("ip-adapter-end").value, 0, 100, 100)
        event.reqBody.ip_adapter_strength = clamp(byId("ip-adapter-strength").value, compatibility?.kind === "anima" ? 0 : -10, 10, 1)
        event.reqBody.ip_adapter_start_percent = Math.min(rawStart, rawEnd)
        event.reqBody.ip_adapter_end_percent = Math.max(rawStart, rawEnd)
        if (!compatibility || !isCompatibleClip(clip.value, compatibility)) {
            notify("Select a CLIP-Vision model whose detected embedding width matches this IP-Adapter.", true)
            // Preserve the request so the native shape guard rejects stale or
            // manually restored values cleanly instead of silently generating
            // without the requested adapter.
            return
        }
        saveState()
    })

    PLUGINS.IMAGE_INFO_BUTTONS.push({
        text: "Use for IP-Adapter",
        on_click: function (_request, image) {
            if (setImage(image?.src || "")) panel.scrollIntoView({ behavior: "smooth", block: "center" })
        },
        filter: function () { return true },
    })

    refreshDefaults()
})()
