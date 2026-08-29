// Standalone native ControlNet settings panel for Easy Diffusion.

;(function () {
    "use strict"
    if (window.__controlNetSettingsPluginLoaded) return
    window.__controlNetSettingsPluginLoaded = true

    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode) {
        console.error("ControlNet plugin: Image Settings anchor was not found")
        return
    }

    const panel = document.createElement("div")
    panel.id = "sdkit3-controlnet-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel gated-feature"
    panel.dataset.featureKeys = "backend_ed_diffusers backend_webui backend_sdkit3"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/controlnet.plugin.html")
    editor.after(panel)

    function orderPanels() {
        const order = [
            "lora-settings-panel",
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
        let previous = document.getElementById("editor-settings")
        if (!previous?.parentNode) return
        for (const id of order) {
            const item = document.getElementById(id)
            if (item?.parentNode === previous.parentNode) {
                previous.after(item)
                previous = item
            }
        }
    }
    window.orderSdkitSettingsPanels = orderPanels
    orderPanels()
    setTimeout(orderPanels, 0)
    setTimeout(orderPanels, 250)

    if (!document.getElementById("controlled-generation-panel-style")) {
        const style = document.createElement("style")
        style.id = "controlled-generation-panel-style"
        style.textContent = `
            .sdkit3-extra-settings-panel h4 { cursor: pointer; }
            .sdkit3-extra-settings-panel h4 small { float: right; }
            .controlled-generation-image-table { width: 100%; }
            .controlled-generation-image-table > tbody > tr > td:first-child {
                padding-right: 4px; white-space: nowrap; vertical-align: top; text-align: right;
            }
            .controlled-generation-grid {
                display: grid; grid-template-columns: minmax(135px, auto) minmax(0, 1fr);
                gap: 6px 9px; align-items: center; margin: 7px 0;
            }
            .controlled-generation-grid input[type="number"] { width: 78px; }
            @media (max-width: 700px) {
                .controlled-generation-grid { grid-template-columns: 1fr; }
            }`
        document.head.appendChild(style)
    }
})()
