// Per-model negative prompt and render-setting history.
;(function () {
    "use strict"
    if (window.__negativeHistoryPluginLoaded) return
    window.__negativeHistoryPluginLoaded = true

    document.body.insertAdjacentHTML(
        "beforeend",
        loadRequiredPluginHTML("/plugins/core/negative-history.plugin.html")
    )
    const template = document.getElementById("negative-history-settings-template")
    const settingsTable = document.querySelector(".parameters-table")
    if (!template || !settingsTable) throw new Error("Negative History: settings UI is unavailable")
    settingsTable.appendChild(template.content.cloneNode(true))
    if (typeof prettifyInputs === "function") prettifyInputs(settingsTable)

    const ENABLED_KEY = "negative-history_save_enabled"
    const HISTORY_KEY = "modelNegativePromptHistory"
    const enabled = document.getElementById("negative-history-save-enabled")
    if (localStorage.getItem(ENABLED_KEY) === null) localStorage.setItem(ENABLED_KEY, "true")
    enabled.checked = localStorage.getItem(ENABLED_KEY) !== "false"

    function modelName() {
        return stableDiffusionModelField?.dataset?.path || stableDiffusionModelField?.value || ""
    }
    function readHistory() {
        try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "{}") }
        catch (_) { return {} }
    }
    function save() {
        if (!enabled.checked || !modelName()) return
        const history = readHistory()
        history[modelName()] = {
            negativePrompt: negativePromptField.value,
            steps: numInferenceStepsField.value,
            guidance: guidanceScaleField.value,
            use_vae_model: vaeModelField.value,
            use_text_encoder_model: textEncoderModelField.value,
            timestamp: Date.now(),
        }
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history))
    }
    function restore() {
        if (!enabled.checked) return
        const entry = readHistory()[modelName()]
        if (!entry) return
        negativePromptField.value = entry.negativePrompt || ""
        if (entry.steps != null) numInferenceStepsField.value = entry.steps
        if (entry.guidance != null) guidanceScaleField.value = entry.guidance
        if (entry.use_vae_model != null) vaeModelField.value = entry.use_vae_model
        if (entry.use_text_encoder_model != null) textEncoderModelField.value = entry.use_text_encoder_model
        negativePromptField.dispatchEvent(new Event("input", { bubbles: true }))
    }

    enabled.addEventListener("change", () => {
        localStorage.setItem(ENABLED_KEY, enabled.checked)
        if (enabled.checked) restore()
    })
    stableDiffusionModelField.addEventListener("change", restore)
    makeImageBtn.addEventListener("click", save, true)
})()
