/* Easy Diffusion 3.5 discovers only files ending in .plugin.js. */
(function () {
    "use strict"
    if (window.__easyDiffusionRabbitHoleLoaded || window.__easyDiffusionRabbitHoleLoading) return
    window.__easyDiffusionRabbitHoleLoading = true

    const script = document.createElement("script")
    script.src = "/plugins/user/rabbithole.plugins.js"
    script.addEventListener("load", () => {
        window.__easyDiffusionRabbitHoleLoading = false
    })
    script.addEventListener("error", () => {
        window.__easyDiffusionRabbitHoleLoading = false
        console.error("Rabbit Hole compatibility script could not be loaded.")
    })
    document.head.appendChild(script)
})()
