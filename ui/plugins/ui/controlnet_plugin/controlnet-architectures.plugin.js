// Native automatic Uni/Union routing for the unified ControlNet panel.

;(function () {
    "use strict"
    if (window.__nativeControlNetArchitecturesPluginLoaded) return
    window.__nativeControlNetArchitecturesPluginLoaded = true

    const controller = window.controlNetModeController
    const slot = document.getElementById("controlnet-extension-modes")
    if (!controller || !slot || typeof PLUGINS !== "object" || typeof ModelDropdown !== "function") {
        console.error("Native ControlNet modes: required Easy Diffusion APIs were not found")
        return
    }

    const TYPE_OPTIONS = {
        auto: [
            ["canny", "Canny / line art"], ["mlsd", "MLSD / straight lines"],
            ["softedge", "HED / soft edge"], ["sketch", "Sketch / scribble"],
            ["openpose", "OpenPose"], ["depth", "Depth"],
            ["segment", "Segmentation"],
        ],
    }

    const DEFINITIONS = [
        {
            key: "auto", title: "Automatic Uni / Union", automatic: true,
            hint: "Detects the base checkpoint from its weights: SD1.x uses Uni-ControlNet and SDXL uses ControlNet Union. Both paths are required in System Settings.",
        },
    ]

    function clamp(value, minimum, maximum, fallback) {
        const number = Number(value)
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback
    }

    function notify(message, isError = false) {
        if (typeof showToast === "function") showToast(message, 5000, isError)
        else if (isError) console.error(message)
        else console.log(message)
    }

    function hasSharedImage() {
        return /^data:image\//.test(controller.sharedImage())
    }

    const instances = new Map()
    for (const definition of DEFINITIONS) {
        const prefix = `controlnet-${definition.key}`
        const section = document.createElement("div")
        section.id = `controlnet-mode-${definition.key}`
        const modelMarkup = definition.automatic ? "" : `
                <label for="${prefix}-model">Model</label>
                <div><input id="${prefix}-model" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="">
                <a href="${definition.help}" target="_blank"><i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">${definition.title} project</span></i></a></div>`
        section.innerHTML = `
            <div class="controlled-generation-grid">
                ${modelMarkup}
                <label for="${prefix}-type">Condition type</label>
                <select id="${prefix}-type">${TYPE_OPTIONS[definition.key].map(([value, label]) => `<option value="${value}">${label}</option>`).join("")}</select>
                <label for="${prefix}-strength">Strength</label>
                <input id="${prefix}-strength" type="number" value="1" min="-10" max="10" step="0.05">
            </div>
            <small>${definition.hint}</small>`
        slot.appendChild(section)
        if (typeof prettifyInputs === "function") prettifyInputs(section)

        const model = definition.automatic
            ? null
            : new ModelDropdown(section.querySelector(`#${prefix}-model`), definition.modelType, "None", false)
        const type = section.querySelector(`#${prefix}-type`)
        const strength = section.querySelector(`#${prefix}-strength`)
        const stateKey = `easy-diffusion-controlnet-${definition.key}-v1`
        let state = {}
        try { state = JSON.parse(localStorage.getItem(stateKey) || "{}") }
        catch (_) { state = {} }
        if (model) model.value = state.model || ""
        type.value = state.type || TYPE_OPTIONS[definition.key][0][0]
        strength.value = state.strength ?? "1"

        function save() {
            localStorage.setItem(stateKey, JSON.stringify({
                model: model?.value || "", type: type.value, strength: strength.value,
            }))
        }
        section.querySelectorAll("input, select").forEach((element) => element.addEventListener("change", save))
        controller.register(definition.key, section)
        controller.registerValidator(definition.key, () => {
            if (!definition.automatic && !model.value) return `${definition.title}: choose a model.`
            if (!hasSharedImage()) return `${definition.title}: load the shared ControlNet image.`
            return ""
        })
        instances.set(definition.key, { definition, model, type, strength, save })
    }

    PLUGINS.TASK_BUILD.push(function (event) {
        const selected = instances.get(controller.mode)
        if (!selected) return
        if ((!selected.definition.automatic && !selected.model.value) || !hasSharedImage()) {
            const requirement = selected.definition.automatic ? "the shared ControlNet image" : "a model and the shared ControlNet image"
            notify(`${selected.definition.title} needs ${requirement}.`, true)
            return
        }
        delete event.reqBody.control_net_lllite_model
        event.reqBody.use_controlnet_model = selected.definition.automatic
            ? "__automatic_uni_union__"
            : `${selected.definition.modelFolder}/${selected.model.value}`
        event.reqBody.control_image = controller.sharedImage()
        event.reqBody.control_alpha = clamp(selected.strength.value, -10, 10, 1)
        event.reqBody.controlnet_union_type = selected.type.value
        const filter = controller.sharedFilter()
        if (filter) event.reqBody.control_filter_to_apply = filter
        else delete event.reqBody.control_filter_to_apply
        selected.save()
    })
})()
