(function () {
    "use strict"

    const tab = document.getElementById("tab-settings")
    if (!tab || tab.dataset.pluginBundle === "settings") return

    const source = window.loadRequiredPluginHTML?.("/plugins/core/ui_plugin/settings.tab.plugin.html")
    if (!source) return
    const parser = document.createElement("template")
    parser.innerHTML = source.trim()
    const definition = parser.content.querySelector("#settings-tab-plugin-template")
    const label = definition?.content.firstElementChild
    if (!label) return

    tab.replaceChildren(label.cloneNode(true))
    tab.dataset.pluginBundle = "settings"
})()
