// Required standalone LoRA Settings panel for Easy Diffusion.
// The native MultiModelSelector binds to #lora_model later in main.js.

;(function () {
    "use strict"
    if (window.__loraSettingsPluginLoaded) return

    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode || typeof window.loadRequiredPluginHTML !== "function") {
        throw new Error("LoRA Settings plugin: Image Settings bootstrap was not found")
    }

    const panel = document.createElement("div")
    panel.id = "lora-settings-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel gated-feature"
    panel.dataset.featureKeys = "backend_ed_diffusers backend_webui backend_sdkit3"
    panel.dataset.pluginOwner = "lora-settings.plugin.js"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/lora_plugin/lora-settings.plugin.html")
    editor.after(panel)

    const addPanel = (id, fragment, anchor) => {
        const embeddingPanel = document.createElement("div")
        embeddingPanel.id = id
        embeddingPanel.className = "settings-box panel-box sdkit3-extra-settings-panel gated-feature"
        embeddingPanel.dataset.featureKeys = "backend_ed_diffusers backend_webui backend_sdkit3"
        embeddingPanel.dataset.pluginOwner = "lora-settings.plugin.js"
        embeddingPanel.innerHTML = window.loadRequiredPluginHTML(fragment)
        anchor.after(embeddingPanel)
        return embeddingPanel
    }

    const positivePanel = addPanel(
        "positive-embeddings-settings-panel",
        "/plugins/core/lora_plugin/positive-embeddings.plugin.html",
        panel
    )
    const negativePanel = addPanel(
        "negative-embeddings-settings-panel",
        "/plugins/core/lora_plugin/negative-embeddings.plugin.html",
        positivePanel
    )
    const setsPanel = addPanel(
        "embedding-sets-settings-panel",
        "/plugins/core/lora_plugin/embedding-sets.plugin.html",
        negativePanel
    )

    if (!panel.querySelector("#lora_model")) {
        throw new Error("LoRA Settings plugin: native LoRA selector anchor was not created")
    }
    if (!positivePanel.querySelector("#hidden_positive_prompt") ||
        !negativePanel.querySelector("#hidden_negative_prompt") ||
        !setsPanel.querySelector("#embedding-set-select")) {
        throw new Error("LoRA Settings plugin: embedding controls were not created")
    }

    window.__loraSettingsPluginLoaded = true
})()
