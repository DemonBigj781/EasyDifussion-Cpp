// Dedicated panels for the native ControlNet architecture implementations.

;(function () {
    "use strict"
    if (window.__nativeControlNetArchitecturesPluginLoaded) return
    window.__nativeControlNetArchitecturesPluginLoaded = true

    if (typeof PLUGINS !== "object" || typeof ModelDropdown !== "function") {
        console.error("Native ControlNet panels: required Easy Diffusion APIs were not found")
        return
    }

    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode) return

    const TYPE_OPTIONS = {
        union: [
            ["canny", "Canny / line art"], ["mlsd", "MLSD / straight lines"],
            ["softedge", "HED / soft edge"], ["sketch", "Sketch / scribble"],
            ["openpose", "OpenPose"], ["depth", "Depth"], ["normal", "Normal"],
            ["segment", "Segmentation"],
        ],
        uni: [
            ["canny", "Canny"], ["mlsd", "MLSD / straight lines"],
            ["softedge", "HED / soft edge"], ["sketch", "Sketch / scribble"],
            ["openpose", "OpenPose"], ["depth", "Depth"],
            ["segment", "Segmentation"], ["global", "Global / CLIP image adapter"],
        ],
        lite: [
            ["canny", "Canny / line art"], ["mlsd", "MLSD / straight lines"],
            ["softedge", "HED / soft edge"], ["sketch", "Sketch / scribble"],
            ["openpose", "OpenPose"], ["depth", "Depth"], ["normal", "Normal"],
            ["segment", "Segmentation"],
        ],
    }

    const DEFINITIONS = [
        {
            key: "union", title: "ControlNet Union", subtitle: "multi-condition SDXL",
            help: "https://github.com/xinsir6/ControlNetPlus",
            hint: "Select an Xinsir Union checkpoint. The condition type is sent to the native Union routing layer.",
        },
        {
            key: "uni", title: "Uni-ControlNet", subtitle: "local + global adapters",
            help: "https://github.com/ShihaoZhaoZSH/Uni-ControlNet",
            hint: "Local types use the matching Uni slot. Global uses OpenAI CLIP ViT-L/14 when installed.",
        },
        {
            key: "lite", title: "ControlNet-LITE", subtitle: "five-stage lightweight control",
            help: "https://github.com/IronTony-Stark/ControlNet-LITE",
            hint: "This is ControlNet-LITE, not ControlNet-LLLite. Select the condition expected by the checkpoint.",
        },
    ]

    const instances = []

    function clamp(value, minimum, maximum, fallback) {
        const number = Number(value)
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback
    }

    function notify(message, isError = false) {
        if (typeof showToast === "function") showToast(message, 5000, isError)
        else if (isError) console.error(message)
        else console.log(message)
    }

    function readState(key) {
        try { return JSON.parse(localStorage.getItem(`easy-diffusion-controlnet-${key}-v1`) || "{}") }
        catch (_) { return {} }
    }

    function makePanel(definition) {
        const key = definition.key
        const prefix = `controlnet-${key}`
        const panel = document.createElement("div")
        panel.id = `sdkit3-controlnet-${key}-panel`
        panel.className = "settings-box panel-box sdkit3-extra-settings-panel gated-feature"
        panel.dataset.featureKeys = "backend_sdkit3"
        panel.innerHTML = `
            <h4 class="collapsible">${definition.title} <small>${definition.subtitle}</small></h4>
            <div class="collapsible-content settings-panel-entries">
                <div class="controlled-generation-grid">
                    <label for="${prefix}-enabled">Enable</label>
                    <input id="${prefix}-enabled" type="checkbox">
                    <label for="${prefix}-model">Model</label>
                    <div><input id="${prefix}-model" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="">
                    <a href="${definition.help}" target="_blank"><i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">${definition.title} project</span></i></a></div>
                    <label for="${prefix}-type">Condition type</label>
                    <select id="${prefix}-type">${TYPE_OPTIONS[key].map(([value, label]) => `<option value="${value}">${label}</option>`).join("")}</select>
                    <label for="${prefix}-strength">Strength</label>
                    <input id="${prefix}-strength" type="number" value="1" min="-10" max="10" step="0.05">
                    <label for="${prefix}-image">Condition image</label>
                    <div>
                        <div id="${prefix}-wrapper" class="preview_image_wrapper displayNone">
                            <img id="${prefix}-preview" class="image_preview" alt="${definition.title} condition">
                            <button id="${prefix}-clear" type="button" class="image_clear_btn" aria-label="Clear condition image"><i class="fa-solid fa-xmark"></i></button>
                        </div>
                        <input id="${prefix}-image" type="file" accept="image/*">
                        <button id="${prefix}-use-init" type="button" class="tertiaryButton smallButton">Use Initial Image</button>
                        <button id="${prefix}-use-control" type="button" class="tertiaryButton smallButton">Use shared ControlNet image</button>
                    </div>
                    <label for="${prefix}-preprocessor">Preprocessor</label>
                    <select id="${prefix}-preprocessor">
                        <option value="">None (image is ready)</option>
                        <option value="canny">Canny</option>
                        <option value="mlsd">MLSD</option>
                        <option value="softedge_hed">HED / soft edge</option>
                        <option value="scribble_hed">Scribble</option>
                        <option value="openpose">OpenPose</option>
                        <option value="depth_midas">Depth</option>
                        <option value="normal_bae">Normal</option>
                        <option value="segment">Segmentation</option>
                    </select>
                </div>
                <small>${definition.hint}</small>
            </div>`
        editor.parentNode.insertBefore(panel, editor.nextSibling)
        if (typeof createCollapsibles === "function") createCollapsibles(panel)
        if (typeof prettifyInputs === "function") prettifyInputs(panel)

        const find = (suffix) => panel.querySelector(`#${prefix}-${suffix}`)
        const enabled = find("enabled")
        const type = find("type")
        const strength = find("strength")
        const imageInput = find("image")
        const preview = find("preview")
        const wrapper = find("wrapper")
        const preprocessor = find("preprocessor")
        const model = new ModelDropdown(find("model"), "controlnet", "None", false)

        function hasImage() { return /^data:image\//.test(preview.getAttribute("src") || "") }
        function updateImage() { wrapper.classList.toggle("displayNone", !hasImage()) }
        function setImage(source) {
            if (!/^data:image\//.test(source || "")) return false
            preview.src = source
            enabled.checked = true
            updateImage()
            save()
            return true
        }
        function save() {
            localStorage.setItem(`easy-diffusion-controlnet-${key}-v1`, JSON.stringify({
                enabled: enabled.checked, model: model.value, type: type.value,
                strength: strength.value, preprocessor: preprocessor.value,
            }))
        }
        function disableOthers() {
            if (!enabled.checked) return
            for (const other of instances) {
                if (other.enabled !== enabled) other.enabled.checked = false
            }
            const standard = document.getElementById("controlnet_enabled")
            if (standard) standard.checked = false
        }

        const state = readState(key)
        enabled.checked = Boolean(state.enabled)
        model.value = state.model || ""
        type.value = state.type || TYPE_OPTIONS[key][0][0]
        strength.value = state.strength ?? "1"
        preprocessor.value = state.preprocessor || ""
        updateImage()

        enabled.addEventListener("change", () => { disableOthers(); save() })
        type.addEventListener("change", save)
        strength.addEventListener("change", save)
        preprocessor.addEventListener("change", save)
        imageInput.addEventListener("change", () => {
            const file = imageInput.files?.[0]
            if (!file?.type.startsWith("image/")) return
            const reader = new FileReader()
            reader.addEventListener("load", () => { setImage(reader.result); disableOthers() })
            reader.readAsDataURL(file)
        })
        find("clear").addEventListener("click", () => {
            imageInput.value = ""
            preview.removeAttribute("src")
            enabled.checked = false
            updateImage()
            save()
        })
        find("use-init").addEventListener("click", () => {
            if (!setImage(document.getElementById("init_image_preview")?.getAttribute("src") || "")) {
                notify("No Initial Image is loaded.", true)
            } else disableOthers()
        })
        find("use-control").addEventListener("click", () => {
            if (!setImage(document.getElementById("control_image_preview")?.getAttribute("src") || "")) {
                notify("No shared ControlNet image is loaded.", true)
            } else disableOthers()
        })
        preview.addEventListener("load", updateImage)

        const instance = { key, definition, panel, enabled, model, type, strength, preview, preprocessor, hasImage, save }
        instances.push(instance)
        return instance
    }

    DEFINITIONS.forEach(makePanel)

    const standardControlNet = document.getElementById("controlnet_enabled")
    standardControlNet?.addEventListener("change", () => {
        if (!standardControlNet.checked) return
        for (const instance of instances) {
            instance.enabled.checked = false
            instance.save()
        }
    })

    function orderPanels() {
        const ids = [
            "sdkit3-controlnet-panel",
            "sdkit3-controlnet-union-panel",
            "sdkit3-controlnet-uni-panel",
            "sdkit3-controlnet-lite-panel",
            "sdkit3-controlnet-preprocessor-panel",
            "sdkit3-lllite-panel",
            "sdkit3-encode-interpose-panel",
            "sdkit3-decode-interpose-panel",
            "sdkit3-wd14-panel",
        ]
        let previous = editor
        for (const id of ids) {
            const panel = document.getElementById(id)
            if (panel?.parentNode === editor.parentNode) {
                previous.after(panel)
                previous = panel
            }
        }
    }
    orderPanels()
    setTimeout(orderPanels, 0)
    setTimeout(orderPanels, 500)
    setTimeout(orderPanels, 2000)

    PLUGINS.TASK_CREATE.push(function (event) {
        const active = instances.filter((instance) => instance.enabled.checked)
        if (active.length === 0) return
        const selected = active[active.length - 1]
        for (const instance of active) {
            if (instance !== selected) instance.enabled.checked = false
        }
        if (!selected.model.value || !selected.hasImage()) {
            notify(`${selected.definition.title} is enabled, but its model or condition image is missing.`, true)
            return
        }

        // Replace any standard ControlNet values already copied by main.js.
        delete event.reqBody.use_controlnet_model
        delete event.reqBody.control_image
        delete event.reqBody.control_alpha
        delete event.reqBody.control_filter_to_apply
        event.reqBody.use_controlnet_model = selected.model.value
        event.reqBody.control_image = selected.preview.src
        event.reqBody.control_alpha = clamp(selected.strength.value, -10, 10, 1)
        event.reqBody.controlnet_union_type = selected.type.value
        if (selected.preprocessor.value) {
            event.reqBody.control_filter_to_apply = selected.preprocessor.value
        }
        selected.save()
    })
})()
