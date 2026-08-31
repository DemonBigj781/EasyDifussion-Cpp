// Built-in WD14 image tagger panel for Easy Diffusion.

;(function () {
    "use strict"
    if (window.__wd14TaggerPluginLoaded) return
    window.__wd14TaggerPluginLoaded = true

    const STATE_KEY = "easy-diffusion-wd14-tagger-v2"
    const DEFAULT_MODEL = "wd-v1-4-moat-tagger-v2"
    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode || typeof ModelDropdown !== "function" || typeof PLUGINS !== "object") {
        console.error("WD14 Tagger plugin: required Easy Diffusion UI APIs were not found")
        return
    }

    const panel = document.createElement("div")
    panel.id = "sdkit3-wd14-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/wdtagger_plugin/wd14-tagger.plugin.html")

    const decode = document.getElementById("sdkit3-decode-interpose-panel")
    if (decode) decode.after(panel)
    else editor.after(panel)
    window.orderSdkitSettingsPanels?.()
    setTimeout(() => window.orderSdkitSettingsPanels?.(), 250)
    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const byId = (id) => document.getElementById(id)
    const model = new ModelDropdown(byId("wd14-model"), "wd14-tagger", "None")
    const preview = byId("wd14-image-preview")

    function readState() {
        try { return JSON.parse(localStorage.getItem(STATE_KEY) || "{}") }
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
            threshold: byId("wd14-threshold").value,
            characterThreshold: byId("wd14-character-threshold").value,
            exclude: byId("wd14-exclude").value,
            replaceUnderscore: byId("wd14-replace-underscore").checked,
            trailingComma: byId("wd14-trailing-comma").checked,
        }))
    }
    function setImage(source) {
        if (!source || !/^data:image\//.test(source)) return false
        preview.src = source
        preview.style.display = "block"
        return true
    }

    byId("wd14-image-input").addEventListener("change", (event) => {
        const file = event.target.files?.[0]
        if (!file?.type.startsWith("image/")) return
        const reader = new FileReader()
        reader.addEventListener("load", () => setImage(reader.result))
        reader.readAsDataURL(file)
    })
    byId("wd14-use-init").addEventListener("click", () => {
        const init = document.getElementById("init_image_preview")
        if (!setImage(init?.getAttribute("src") || "")) notify("No Initial Image is loaded.", true)
    })

    const state = readState()
    model.value = state.model || DEFAULT_MODEL
    byId("wd14-threshold").value = state.threshold ?? "0.35"
    byId("wd14-character-threshold").value = state.characterThreshold ?? "0.85"
    byId("wd14-exclude").value = state.exclude ?? ""
    byId("wd14-replace-underscore").checked = Boolean(state.replaceUnderscore)
    byId("wd14-trailing-comma").checked = Boolean(state.trailingComma)
    panel.querySelectorAll("input").forEach((input) => input.addEventListener("change", saveState))

    async function tagImage(source, mode) {
        if (!source || !/^data:image\//.test(source)) throw new Error("Select an image to tag")
        const status = byId("wd14-status")
        status.textContent = "Tagging…"
        const response = await fetch("/tag", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image: source,
                model: model.value || DEFAULT_MODEL,
                threshold: clamp(byId("wd14-threshold").value, 0, 1, 0.35),
                character_threshold: clamp(byId("wd14-character-threshold").value, 0, 1, 0.85),
                exclude_tags: byId("wd14-exclude").value,
                replace_underscore: byId("wd14-replace-underscore").checked,
                trailing_comma: byId("wd14-trailing-comma").checked,
            }),
        })
        const result = await response.json()
        if (!response.ok) throw new Error(result.detail || `WD14 request failed (HTTP ${response.status})`)
        const prompt = byId("prompt")
        if (mode === "replace") prompt.value = result.tags
        else prompt.value = [prompt.value.trim(), result.tags].filter(Boolean).join(prompt.value.trim() ? ", " : "")
        prompt.dispatchEvent(new Event("input", { bubbles: true }))
        status.textContent = `${result.matches.length} tags (${result.model})`
        saveState()
    }

    async function tagPreview(mode) {
        try { await tagImage(preview.getAttribute("src") || "", mode) }
        catch (error) { byId("wd14-status").textContent = error.message; notify(error.message, true) }
    }
    byId("wd14-replace-prompt").addEventListener("click", () => tagPreview("replace"))
    byId("wd14-append-prompt").addEventListener("click", () => tagPreview("append"))

    PLUGINS.IMAGE_INFO_BUTTONS.push({
        text: "WD14 Tag → Prompt",
        on_click: async function (_request, image) {
            try { await tagImage(image.currentSrc || image.src, "append") }
            catch (error) { notify(error.message, true) }
        },
        filter: function () { return true },
    })
})()
