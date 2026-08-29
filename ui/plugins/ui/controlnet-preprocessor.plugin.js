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
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel gated-feature"
    panel.dataset.featureKeys = "backend_ed_diffusers backend_webui backend_sdkit3"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/controlnet-preprocessor.plugin.html")

    const filter = panel.querySelector("#control_image_filter")
    if (!filter) throw new Error("ControlNet Preprocessor plugin: selector was not created")
    filter.style.width = "100%"

    const controlPanel = document.getElementById("sdkit3-controlnet-panel")
    if (controlPanel) controlPanel.after(panel)
    else editor.after(panel)
    window.orderSdkitSettingsPanels?.()
    setTimeout(() => window.orderSdkitSettingsPanels?.(), 250)
})()
