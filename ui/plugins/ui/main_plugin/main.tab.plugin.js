(function () {
    "use strict"

    const tab = document.getElementById("tab-main")
    if (!tab || tab.dataset.pluginBundle === "main") return

    const source = window.loadRequiredPluginHTML?.("/plugins/core/main_plugin/main.tab.plugin.html")
    if (!source) return
    const parser = document.createElement("template")
    parser.innerHTML = source.trim()
    const definition = parser.content.querySelector("#main-tab-plugin-template")
    const label = definition?.content.firstElementChild
    if (!label) return

    tab.replaceChildren(label.cloneNode(true))
    tab.dataset.pluginBundle = "main"
})()
