// Standalone ControlNet preprocessor settings panel for Easy Diffusion.

;(function () {
    "use strict"
    if (window.__controlNetPreprocessorPluginLoaded) return
    window.__controlNetPreprocessorPluginLoaded = true

    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode) {
        console.error("ControlNet Preprocessor plugin: Image Settings anchor was not found")
        return
    }

    const panel = document.createElement("div")
    panel.id = "sdkit3-controlnet-preprocessor-panel"
    panel.className = "gated-feature"
    panel.dataset.featureKeys = "backend_ed_diffusers backend_webui backend_sdkit3"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/controlnet_plugin/controlnet-preprocessor.plugin.html")

    const filter = panel.querySelector("#control_image_filter")
    if (!filter) throw new Error("ControlNet Preprocessor plugin: selector was not created")
    filter.style.width = "100%"

    const sharedSlot = document.getElementById("controlnet-shared-slot")
    if (!sharedSlot) throw new Error("ControlNet Preprocessor plugin: shared slot was not created")
    sharedSlot.appendChild(panel)
})()
