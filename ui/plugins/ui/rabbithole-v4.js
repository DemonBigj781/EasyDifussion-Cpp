/* Easy Diffusion 4.x Rabbit Hole entry point. */
(function () {
    "use strict"
    if (window.__easyDiffusionRabbitHoleLoaded || window.__easyDiffusionRabbitHolePortLoading) return
    window.__easyDiffusionRabbitHolePortLoading = true
    window.__easyDiffusionRabbitHoleRequestedVersion = "4"
    const script = document.createElement("script")
    script.src = new URL("rabbithole.plugins.js", document.currentScript?.src || window.location.href).href
    script.addEventListener("load", () => { window.__easyDiffusionRabbitHolePortLoading = false })
    script.addEventListener("error", () => {
        window.__easyDiffusionRabbitHolePortLoading = false
        window.__easyDiffusionRabbitHoleLoading = false
        console.error("Rabbit Hole 4 implementation could not be loaded.")
    })
    document.head.appendChild(script)
})()
