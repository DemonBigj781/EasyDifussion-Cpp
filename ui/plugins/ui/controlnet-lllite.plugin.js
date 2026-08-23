;(function () {
    "use strict"

    const STATE_KEY = "easy-diffusion-controlnet-lllite-v1"
    const LEGACY_STATE_KEY = "easy-diffusion-controlled-image-generation-v1"
    const outputSettings = document.querySelector("#output-settings")
    if (!outputSettings?.parentNode || typeof ModelDropdown !== "function" || typeof PLUGINS !== "object" || document.querySelector("#sdkit3-lllite-panel")) return

    const panel = document.createElement("div")
    panel.id = "sdkit3-lllite-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel"
    panel.innerHTML = `
        <h4 class="collapsible">ControlNet-LLLite <small>image conditioning</small></h4>
        <div class="collapsible-content controlled-generation-content">
            <table class="controlled-generation-image-table"><tbody><tr class="pl-5">
                <td><label for="lllite-image-input">ControlNet-LLLite Image:</label></td>
                <td>
                    <div id="lllite-image-wrapper" class="preview_image_wrapper displayNone">
                        <img id="lllite-image-preview" class="image_preview" alt="LLLite condition preview">
                        <button id="lllite-image-clear" type="button" class="image_clear_btn"><i class="fa-solid fa-xmark"></i></button>
                    </div>
                    <input id="lllite-image-input" type="file" accept="image/*">
                    <button id="lllite-use-init" type="button" class="tertiaryButton smallButton">Use init image</button>
                    <a href="https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI" target="_blank"><i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">ControlNet-LLLite information</span></i></a>
                    <div id="lllite-config" class="displayNone controlled-generation-grid">
                        <label>Enable</label><input id="lllite-enabled" type="checkbox">
                        <label>Model</label><input id="lllite-model" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="">
                        <label>Strength</label><input id="lllite-strength" type="number" value="1" min="-10" max="10" step="0.05">
                        <label>Step range (%)</label><div><input id="lllite-start" type="number" value="0" min="0" max="100" step="1"> – <input id="lllite-end" type="number" value="100" min="0" max="100" step="1"></div>
                    </div>
                    <small>The condition is passed directly, matching the ComfyUI node. Apply depth/canny preprocessing first when the selected model expects it.</small>
                </td>
            </tr></tbody></table>
        </div>`

    const anchor = document.querySelector("#sdkit3-controlnet-preprocessor-panel") || document.querySelector("#sdkit3-controlnet-panel") || outputSettings
    anchor.parentNode.insertBefore(panel, anchor.nextSibling)

    const style = document.createElement("style")
    style.textContent = `
        #sdkit3-lllite-panel { margin-top: 10px; }
        #sdkit3-lllite-panel h4 { cursor: pointer; }
        #sdkit3-lllite-panel h4 small { float: right; }
        #sdkit3-lllite-panel .controlled-generation-content { padding-top: 8px; }
        #sdkit3-lllite-panel table { width: 100%; }
        #sdkit3-lllite-panel table > tbody > tr > td:first-child { padding-right: 4px; white-space: nowrap; vertical-align: top; text-align: right; }
        #sdkit3-lllite-panel .controlled-generation-grid { display: grid; grid-template-columns: minmax(105px, auto) minmax(0, 1fr); gap: 6px 9px; align-items: center; margin: 7px 0; }
        #sdkit3-lllite-panel input[type="number"] { width: 74px; }
    `
    document.head.appendChild(style)

    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const byId = (id) => document.getElementById(id)
    const model = new ModelDropdown(byId("lllite-model"), "controlnet-lllite", "None")
    const clamp = (value, minimum, maximum, fallback) => {
        const number = Number(value)
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback
    }
    const readJSON = (key) => {
        try { return JSON.parse(localStorage.getItem(key) || "{}") }
        catch (_) { return {} }
    }
    const legacy = readJSON(LEGACY_STATE_KEY)
    const state = { ...legacy, ...readJSON(STATE_KEY) }

    function notify(message, isError) {
        if (typeof showToast === "function") showToast(message, 5000, Boolean(isError))
        else if (isError) alert(message)
    }

    function writeState() {
        localStorage.setItem(STATE_KEY, JSON.stringify({
            llliteEnabled: byId("lllite-enabled").checked,
            llliteModel: model.value,
            llliteStrength: byId("lllite-strength").value,
            llliteStart: byId("lllite-start").value,
            llliteEnd: byId("lllite-end").value,
        }))
    }

    function updateImageUI() {
        const hasImage = /^data:image\//.test(byId("lllite-image-preview").src || "")
        byId("lllite-image-wrapper").classList.toggle("displayNone", !hasImage)
        byId("lllite-config").classList.toggle("displayNone", !hasImage)
    }

    byId("lllite-enabled").checked = Boolean(state.llliteEnabled)
    byId("lllite-strength").value = state.llliteStrength ?? "1"
    byId("lllite-start").value = state.llliteStart ?? "0"
    byId("lllite-end").value = state.llliteEnd ?? "100"
    model.value = state.llliteModel || "controlnet-lllite-depth-test"

    byId("lllite-image-preview").addEventListener("load", updateImageUI)
    byId("lllite-image-input").addEventListener("change", () => {
        const file = byId("lllite-image-input").files?.[0]
        if (!file?.type.startsWith("image/")) return
        const reader = new FileReader()
        reader.addEventListener("load", () => {
            byId("lllite-image-preview").src = reader.result
            byId("lllite-enabled").checked = true
            writeState()
        })
        reader.readAsDataURL(file)
    })
    byId("lllite-use-init").addEventListener("click", () => {
        const initImage = document.querySelector("#init_image_preview")
        if (!/^data:image\//.test(initImage?.src || "")) return notify("No init image is loaded.", true)
        byId("lllite-image-preview").src = initImage.src
        byId("lllite-enabled").checked = true
        writeState()
    })
    byId("lllite-image-clear").addEventListener("click", () => {
        byId("lllite-image-input").value = ""
        byId("lllite-image-preview").removeAttribute("src")
        byId("lllite-enabled").checked = false
        updateImageUI()
        writeState()
    })
    panel.querySelectorAll("input").forEach((input) => input.addEventListener("change", writeState))
    updateImageUI()

    PLUGINS.TASK_CREATE.push(function (event) {
        if (!byId("lllite-enabled").checked) return
        const image = byId("lllite-image-preview").src
        if (!model.value || !/^data:image\//.test(image)) return notify("ControlNet-LLLite is enabled, but its model or condition image is missing.", true)
        event.reqBody.control_net_lllite_model = model.value
        event.reqBody.control_net_lllite_image = image
        event.reqBody.control_net_lllite_strength = clamp(byId("lllite-strength").value, -10, 10, 1)
        event.reqBody.control_net_lllite_start_percent = clamp(byId("lllite-start").value, 0, 100, 0)
        event.reqBody.control_net_lllite_end_percent = clamp(byId("lllite-end").value, 0, 100, 100)
        writeState()
    })
})()
