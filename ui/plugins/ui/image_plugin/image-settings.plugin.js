// Core Image Settings bootstrap for Easy Diffusion.
// This plugin runs before main.js so every native image field is present when
// Easy Diffusion binds its controls.

;(function () {
    "use strict"

    window.loadRequiredPluginHTML = window.loadRequiredPluginHTML || function (url) {
        const request = new XMLHttpRequest()
        request.open("GET", url, false)
        request.send(null)
        if (request.status < 200 || request.status >= 300) {
            throw new Error(`Required plugin HTML failed to load: ${url} (HTTP ${request.status})`)
        }
        return request.responseText
    }

    if (window.__imageSettingsPluginLoaded) return

    let panel = document.getElementById("editor-settings")
    if (!panel) {
        const renderSettings = document.getElementById("render-settings")
        if (!renderSettings?.parentNode) {
            throw new Error("Image Settings plugin: Render Settings anchor was not found")
        }

        // Core controls must exist before main.js is evaluated. Load the
        // plugin-owned fragment synchronously during the early bootstrap.
        renderSettings.insertAdjacentHTML(
            "beforebegin",
            window.loadRequiredPluginHTML("/plugins/core/image_plugin/image-settings.plugin.html")
        )
        panel = document.getElementById("editor-settings")
    }

    if (!panel) {
        throw new Error("Image Settings plugin: panel could not be created")
    }

    panel.dataset.pluginOwner = "image-settings.plugin.js"
    window.__imageSettingsPluginLoaded = true
})()
