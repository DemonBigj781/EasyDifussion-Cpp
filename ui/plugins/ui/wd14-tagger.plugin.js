;(function () {
    "use strict"

    const STATE_KEY = "easy-diffusion-wd14-tagger-v1"
    const LEGACY_STATE_KEY = "easy-diffusion-controlled-image-generation-v1"
    const DEFAULT_MODEL = "wd-v1-4-moat-tagger-v2"
    const outputSettings = document.querySelector("#output-settings")
    if (!outputSettings?.parentNode || typeof ModelDropdown !== "function" || typeof PLUGINS !== "object" || document.querySelector("#sdkit3-wd14-panel")) return

    const panel = document.createElement("div")
    panel.id = "sdkit3-wd14-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel"
    panel.innerHTML = `
        <h4 class="collapsible">WD14 Tagger <small>sdkit3</small></h4>
        <div class="collapsible-content controlled-generation-content">
            <div class="controlled-generation-grid">
                <label>Model</label><input id="wd14-model" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="">
                <label>Image</label>
                <div>
                    <input id="wd14-image-input" type="file" accept="image/*">
                    <button id="wd14-use-init" type="button" class="tertiaryButton smallButton">Use init image</button>
                    <img id="wd14-image-preview" class="controlled-generation-preview" alt="WD14 input preview">
                </div>
                <label>General / character</label><div><input id="wd14-threshold" type="number" value="0.35" min="0" max="1" step="0.05"> / <input id="wd14-character-threshold" type="number" value="0.85" min="0" max="1" step="0.05"></div>
                <label>Exclude tags</label><input id="wd14-exclude" type="text" placeholder="tag_one, tag_two">
                <label>Formatting</label><div><label><input id="wd14-replace-underscore" type="checkbox"> spaces</label> <label><input id="wd14-trailing-comma" type="checkbox"> trailing comma</label></div>
            </div>
            <div class="controlled-generation-actions">
                <button id="wd14-replace-prompt" type="button" class="secondaryButton">Tag → replace prompt</button>
                <button id="wd14-append-prompt" type="button" class="secondaryButton">Tag → append prompt</button>
                <span id="wd14-status"></span>
            </div>
        </div>`

    const anchor = document.querySelector("#sdkit3-decode-interpose-panel") || document.querySelector("#sdkit3-encode-interpose-panel") || document.querySelector("#sdkit3-lllite-panel") || document.querySelector("#sdkit3-controlnet-preprocessor-panel") || document.querySelector("#sdkit3-controlnet-panel") || outputSettings
    anchor.parentNode.insertBefore(panel, anchor.nextSibling)

    const style = document.createElement("style")
    style.textContent = `
        #sdkit3-wd14-panel { margin-top: 10px; }
        #sdkit3-wd14-panel h4 { cursor: pointer; }
        #sdkit3-wd14-panel h4 small { float: right; }
        #sdkit3-wd14-panel .controlled-generation-content { padding-top: 8px; }
        #sdkit3-wd14-panel .controlled-generation-grid { display: grid; grid-template-columns: minmax(130px, auto) minmax(0, 1fr); gap: 6px 9px; align-items: center; margin: 7px 0; }
        #sdkit3-wd14-panel input[type="number"] { width: 74px; }
        #wd14-image-preview { display: none; margin-top: 6px; max-width: 190px; max-height: 140px; border-radius: var(--input-border-radius); object-fit: contain; }
        #sdkit3-wd14-panel .controlled-generation-actions { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
        #wd14-status { font-size: 0.85em; }
    `
    document.head.appendChild(style)

    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const byId = (id) => document.getElementById(id)
    const modelDropdown = new ModelDropdown(byId("wd14-model"), "wd14-tagger", "None")
    const clamp = (value, minimum, maximum, fallback) => {
        const number = Number(value)
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback
    }
    const notify = (message, isError) => {
        if (typeof showToast === "function") showToast(message, 5000, Boolean(isError))
        else if (isError) alert(message)
    }

    function readState() {
        try { return JSON.parse(localStorage.getItem(STATE_KEY) || localStorage.getItem(LEGACY_STATE_KEY) || "{}") }
        catch (_) { return {} }
    }

    function writeState() {
        localStorage.setItem(STATE_KEY, JSON.stringify({
            model: modelDropdown.value,
            threshold: byId("wd14-threshold").value,
            characterThreshold: byId("wd14-character-threshold").value,
            exclude: byId("wd14-exclude").value,
            replaceUnderscore: byId("wd14-replace-underscore").checked,
            trailingComma: byId("wd14-trailing-comma").checked,
        }))
    }

    const state = readState()
    modelDropdown.value = state.model || state.wd14Model || DEFAULT_MODEL
    byId("wd14-threshold").value = state.threshold ?? state.wd14Threshold ?? "0.35"
    byId("wd14-character-threshold").value = state.characterThreshold ?? state.wd14CharacterThreshold ?? "0.85"
    byId("wd14-exclude").value = state.exclude ?? state.wd14Exclude ?? ""
    byId("wd14-replace-underscore").checked = Boolean(state.replaceUnderscore ?? state.wd14ReplaceUnderscore)
    byId("wd14-trailing-comma").checked = Boolean(state.trailingComma ?? state.wd14TrailingComma)
    panel.querySelectorAll("input").forEach((input) => input.addEventListener("change", writeState))

    byId("wd14-image-input").addEventListener("change", () => {
        const file = byId("wd14-image-input").files?.[0]
        if (!file?.type.startsWith("image/")) return
        const reader = new FileReader()
        reader.addEventListener("load", () => {
            byId("wd14-image-preview").src = reader.result
            byId("wd14-image-preview").style.display = "block"
        })
        reader.readAsDataURL(file)
    })
    byId("wd14-use-init").addEventListener("click", () => {
        const source = document.querySelector("#init_image_preview")?.src || ""
        if (!/^data:image\//.test(source)) return notify("No init image is loaded.", true)
        byId("wd14-image-preview").src = source
        byId("wd14-image-preview").style.display = "block"
    })

    async function runTagger(image, mode) {
        if (!image) throw new Error("Select an image to tag")
        byId("wd14-status").textContent = "Tagging…"
        const response = await fetch("/tag", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image,
                model: modelDropdown.value || DEFAULT_MODEL,
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
        byId("wd14-status").textContent = `${result.matches.length} tags (${result.model})`
        writeState()
    }

    async function tagPreview(mode) {
        try { await runTagger(byId("wd14-image-preview").src, mode) }
        catch (error) { byId("wd14-status").textContent = error.message; notify(error.message, true) }
    }
    byId("wd14-replace-prompt").addEventListener("click", () => tagPreview("replace"))
    byId("wd14-append-prompt").addEventListener("click", () => tagPreview("append"))
    PLUGINS.IMAGE_INFO_BUTTONS.push({
        text: "WD14 Tag → Prompt",
        on_click: async function (_request, image) {
            try { await runTagger(image.currentSrc || image.src, "append") }
            catch (error) { notify(error.message, true) }
        },
        filter: function () { return true },
    })
})()
