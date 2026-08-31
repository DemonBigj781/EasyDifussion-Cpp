/* Rabbit Hole dispatcher for Easy Diffusion 3.5, 4, and 4.5. */
(function () {
    "use strict"
    if (window.__easyDiffusionRabbitHoleLoaded || window.__easyDiffusionRabbitHoleLoading) return
    window.__easyDiffusionRabbitHoleLoading = true

    function visibleVersionText() {
        const root = document.getElementById("version")
        if (!root) return ""
        const gatedVersions = Array.from(root.querySelectorAll("[data-feature-keys]"))
        if (gatedVersions.length === 0) return root.textContent || ""
        const visibleVersion = gatedVersions.find((element) => {
            const style = window.getComputedStyle?.(element)
            return element.style.display !== "none" && style?.display !== "none"
        })
        return visibleVersion?.textContent || ""
    }

    const sourceUrl = document.currentScript?.src || window.location.href
    let attempts = 0
    function loadMatchingPort() {
        const versionText = visibleVersionText()
        if (!versionText && attempts++ < 100) {
            window.setTimeout(loadMatchingPort, 100)
            return
        }
        const match = versionText.match(/v?(\d+)(?:\.(\d+))?/i)
        const major = Number(match?.[1] || 3)
        const minor = Number(match?.[2] || 5)
        const port = major > 4 || (major === 4 && minor >= 5) ? "4.5" : major >= 4 ? "4" : "3.5"
        const script = document.createElement("script")
        script.src = new URL(`rabbithole-v${port}.js`, sourceUrl).href
        script.addEventListener("error", () => {
            window.__easyDiffusionRabbitHoleLoading = false
            console.error(`Rabbit Hole ${port} compatibility script could not be loaded.`)
        })
        document.head.appendChild(script)
    }
    loadMatchingPort()
})()
