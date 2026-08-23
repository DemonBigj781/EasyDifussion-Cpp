/*
 * Hide Help & Community and What's new
 *
 * Keeps Easy Diffusion's help and release-notes tabs out of the UI.
 */
;(function () {
    "use strict"

    const hiddenTabIds = ["tab-about", "tab-news"]
    const hiddenContentIds = ["tab-content-about", "tab-content-news"]

    const style = document.createElement("style")
    style.id = "hide-help-and-news-css"
    style.textContent = `
        ${hiddenTabIds.map((id) => `#${id}`).join(",\n        ")},
        ${hiddenContentIds.map((id) => `#${id}`).join(",\n        ")} {
            display: none !important;
        }
    `
    document.head.appendChild(style)

    let redirecting = false

    function keepVisibleTabActive() {
        const hiddenTabIsActive = hiddenTabIds.some((id) =>
            document.getElementById(id)?.classList.contains("active")
        )

        if (!hiddenTabIsActive || redirecting) {
            return
        }

        const mainTab = document.getElementById("tab-main")
        if (!mainTab) {
            return
        }

        redirecting = true
        try {
            if (typeof selectTab === "function") {
                selectTab("tab-main")
            } else {
                mainTab.click()
            }
        } finally {
            queueMicrotask(() => {
                redirecting = false
            })
        }
    }

    keepVisibleTabActive()
    document.addEventListener("tabClick", keepVisibleTabActive)

    const observer = new MutationObserver(keepVisibleTabActive)
    const observerOptions = {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["class"],
    }

    const tabContainer = document.getElementById("tab-container")
    const tabContentWrapper = document.getElementById("tab-content-wrapper")

    if (tabContainer) {
        observer.observe(tabContainer, observerOptions)
    }
    if (tabContentWrapper) {
        observer.observe(tabContentWrapper, observerOptions)
    }
})()
