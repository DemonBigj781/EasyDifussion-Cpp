(function () {
    "use strict"

    if (window.PerchancePluginCore) return

    const ID_PREFIX = "perchance-generator"
    const STORAGE_KEY = "perchance_generator_settings_v1"

    function element(suffix) {
        return document.getElementById(`${ID_PREFIX}-${suffix}`)
    }

    function loadSettings() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}")
        } catch (_) {
            return {}
        }
    }

    function saveSettings(values) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...loadSettings(), ...values }))
    }

    async function requestJson(url, body) {
        const response = await fetch(url, {
            method: body === undefined ? "GET" : "POST",
            headers: body === undefined ? {} : { "Content-Type": "application/json" },
            body: body === undefined ? undefined : JSON.stringify(body),
        })
        const raw = await response.text()
        let data
        try {
            data = raw ? JSON.parse(raw) : {}
        } catch (_) {
            data = { detail: raw }
        }
        if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`)
        return data
    }

    function setStatus(message) {
        document.querySelectorAll("[data-perchance-status]").forEach((target) => {
            target.textContent = message
        })
    }

    function setOutputDirectory(path) {
        document.querySelectorAll("[data-perchance-output-dir]").forEach((target) => {
            target.textContent = path ? `Output: ${path}` : ""
        })
    }

    function setRelease(release) {
        document.querySelectorAll("[data-perchance-release]").forEach((target) => {
            if (!release?.url) {
                target.hidden = true
                return
            }
            target.hidden = false
            target.href = release.url
            target.textContent = release.name || release.tag || "Perchance release"
            target.title = release.sha256 ? `Expected SHA-256: ${release.sha256}` : ""
        })
    }

    function dispatchInput(target) {
        target.dispatchEvent(new Event("input", { bubbles: true }))
        target.dispatchEvent(new Event("change", { bubbles: true }))
    }

    function ensureTab(id, label, icon) {
        const tabContainer = document.getElementById("tab-container") || document.querySelector(".tab-container")
        const contentWrapper = document.getElementById("tab-content-wrapper")
        if (!tabContainer || !contentWrapper || typeof linkTabContents !== "function") return null

        let tab = document.getElementById(`tab-${id}`)
        let content = document.getElementById(`tab-content-${id}`)
        if (!tab) {
            tab = document.createElement("span")
            tab.id = `tab-${id}`
            tab.className = "tab"
            tab.innerHTML = `<span><i class="fa ${icon} icon"></i> ${label}</span>`
            tabContainer.appendChild(tab)
        }
        if (!content) {
            content = document.createElement("div")
            content.id = `tab-content-${id}`
            content.className = "tab-content"
            const inner = document.createElement("div")
            inner.className = "tab-content-inner"
            inner.dataset.perchanceTabHost = id
            content.appendChild(inner)
            contentWrapper.appendChild(content)
        }
        linkTabContents(tab)
        return content.querySelector("[data-perchance-tab-host]")
    }

    function ensureMainTab() {
        const host = ensureTab("perchance", "Perchance", "fa-wand-magic-sparkles")
        if (host) {
            host.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:12px;align-items:start;"
        }
        return host
    }

    function initializePanel(panel) {
        if (typeof createCollapsibles === "function") createCollapsibles(panel)
        requestJson("/perchance/status")
            .then((data) => {
                setOutputDirectory(data.output_directory)
                setRelease(data.release)
                setStatus(data.busy ? "Perchance is already busy." : (data.ready ? "Ready" : "Perchance launcher is unavailable."))
                if (!data.ready) panel.querySelectorAll("button[data-perchance-action]").forEach((button) => {
                    button.disabled = true
                })
            })
            .catch((error) => {
                setStatus(`Plugin server unavailable: ${error.message}`)
                panel.querySelectorAll("button[data-perchance-action]").forEach((button) => {
                    button.disabled = true
                })
            })
    }

    window.PerchancePluginCore = {
        ID_PREFIX,
        element,
        loadSettings,
        saveSettings,
        requestJson,
        setStatus,
        setOutputDirectory,
        setRelease,
        dispatchInput,
        ensureTab,
        ensureMainTab,
        initializePanel,
    }
})()
