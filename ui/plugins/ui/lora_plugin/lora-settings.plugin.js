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

    if (!panel.querySelector("#lora_model")) {
        throw new Error("LoRA Settings plugin: native LoRA selector anchor was not created")
    }

    window.__loraSettingsPluginLoaded = true
})()
