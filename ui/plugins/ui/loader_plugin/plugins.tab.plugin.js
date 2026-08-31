(function () {
    "use strict"

    if (typeof createLocalPluginManagerTab !== "function") {
        throw new Error("Plugin Manager tab bootstrap is unavailable")
    }
    createLocalPluginManagerTab()
})()
