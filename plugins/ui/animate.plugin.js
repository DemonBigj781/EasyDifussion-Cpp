/* Compatibility filename for Easy Diffusion 3.5 plugin discovery. */
(function () {
    "use strict"
    if (window.__easyDiffusionAnimatePluginLoaded || window.__easyDiffusionAnimatePluginLoading) return
    window.__easyDiffusionAnimatePluginLoading = true

    const script = document.createElement("script")
    script.src = "/plugins/user/animate.plugin%20(1).js"
    script.addEventListener("error", () => {
        window.__easyDiffusionAnimatePluginLoading = false
        console.error("The complete Animate plugin could not be loaded.")
    })
    document.head.appendChild(script)
})()
