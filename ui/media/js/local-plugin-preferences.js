// One catalog and preference store for the legacy and C++ plugin managers.
(() => {
    "use strict"
    if (window.LocalPluginPreferences) return

    const storageKey = "easy-diffusion-enabled-local-plugins-v1"
    const defaultsVersionKey = "easy-diffusion-local-plugin-defaults-version"
    const defaultsVersion = 3
    const catalog = Object.freeze([
        { id: "image-modifiers", name: "Image Modifiers (show/hide)", defaultEnabled: true, addedInDefaultsVersion: 3, port: "native" },
        { id: "perchance-image", name: "Perchance image", path: "/plugins/core/perchance_plugin/perchance-image.plugin.js", defaultEnabled: true, addedInDefaultsVersion: 2, port: "native", nativePath: "/cpp-ui/perchance/image" },
        { id: "perchance-text", name: "Perchance text", path: "/plugins/core/perchance_plugin/perchance-text.plugin.js", defaultEnabled: true, addedInDefaultsVersion: 2, port: "native", nativePath: "/cpp-ui/perchance/text" },
        { id: "perchance-gallery", name: "Perchance gallery", path: "/plugins/core/perchance_plugin/perchance-gallery.tab.plugin.js", defaultEnabled: true, addedInDefaultsVersion: 2, port: "native", nativePath: "/cpp-ui/perchance/gallery" },
        { id: "accessibility-improvements", name: "Accessibility improvements", path: "/plugins/core/ui_plugin/accessibility-improvements.plugin.js", port: "native" },
        { id: "animate", name: "Animate", path: "/plugins/core/animate_plugin/animate.plugin.js", port: "native" },
        { id: "daily-folders", name: "Daily output folders", path: "/plugins/core/ui_plugin/daily-folders.plugin.js", port: "native" },
        { id: "disable-source-image-zoom", name: "Disable source-image zoom", path: "/plugins/core/ui_plugin/disable-source-image-zoom.plugin.js", port: "native" },
        { id: "gpu-mode-quick-toggle", name: "GPU mode quick toggle", path: "/plugins/core/ui_plugin/gpu-mode-quick-toggle.plugin.js", port: "native" },
        { id: "make-image-always-visible", name: "Always-visible Make Image button", path: "/plugins/core/ui_plugin/make-image-button-always-visible.plugin.js", port: "native" },
        { id: "processing-order-quick-toggle", name: "Processing-order quick toggle", path: "/plugins/core/ui_plugin/processing-order-quick-toggle.plugin.js", port: "native" },
        { id: "prompt-diff", name: "Prompt diff", path: "/plugins/core/prompt_plugin/prompt-diff.plugin.js", port: "native" },
        { id: "prompt-translator", name: "Prompt translator (uses Google Translate)", path: "/plugins/core/prompt_plugin/prompt-translator.plugin.js", port: "native" },
        { id: "queue-counter", name: "Queue counter", path: "/plugins/core/ui_plugin/queue-counter.plugin.js", port: "native" },
        { id: "rabbit-hole", name: "Rabbit Hole UI (3.5 / 4 / 4.5)", path: "/plugins/core/rabithole_plugin/rabbithole.plugin.js", port: "native" },
        { id: "random-seed-quick-toggle", name: "Random-seed quick toggle", path: "/plugins/core/ui_plugin/random-seed-quick-toggle.plugin.js", port: "native" },
        { id: "seed-randomizer", name: "Batch seed randomizer", path: "/plugins/core/ui_plugin/seed-randomizer.plugin.js", port: "native" },
        { id: "prompt-assist", name: "Prompt assistance: token estimates, autocomplete and spellcheck (reload)", path: "/plugins/core/prompt_plugin/prompt-assist.plugin.js", port: "native" },
        { id: "spell-tokenizer", name: "Spell tokenizer and merged tag search", path: "/plugins/core/prompt_plugin/spell-tokenizer.plugin.js", port: "native" },
        { id: "stig-image-to-img2img", name: "Stig image-to-img2img tools", path: "/plugins/core/image_plugin/stig-image-to-img2img.plugin.js", port: "native" },
        { id: "stig-image-utilities", name: "Stig image utilities", path: "/plugins/core/image_plugin/stigs-image_utilities.plugin.js", port: "native" },
        { id: "stig-lora-shuttle", name: "Stig LoRA shuttle controls", path: "/plugins/core/lora_plugin/stigs-lora-shuttle-controls.plugin.js", port: "native" },
        { id: "stig-text-to-prompt", name: "Stig text-to-prompt", path: "/plugins/core/prompt_plugin/stig-text2prompt.plugin.js", port: "native" },
        { id: "storyteller", name: "Storyteller tab", path: "/plugins/core/prompt_plugin/storyteller.plugin.js", port: "native" },
        { id: "template-manager", name: "Template manager", path: "/plugins/core/prompt_plugin/template-manager.plugin.js", port: "native" },
        { id: "toggle-spellcheck", name: "Browser spellcheck toggle", path: "/plugins/core/prompt_plugin/toggle-spellcheck.plugin.js", port: "native" },
    ])
    const knownIds = new Set(catalog.map(plugin => plugin.id))
    let controlInstance = 0
    const visibilityStyle = document.createElement("style")
    visibilityStyle.textContent = `
        html[data-image-modifiers-hidden] #editor-inputs-tags-container,
        html[data-image-modifiers-hidden] #editor-modifiers,
        html[data-image-modifiers-hidden] #modifier-settings-config,
        html[data-image-modifiers-hidden] #cpp-image-modifiers { display: none !important; }
    `
    document.head.append(visibilityStyle)

    function getEnabled() {
        const defaults = new Set(catalog.filter(plugin => plugin.defaultEnabled).map(plugin => plugin.id))
        try {
            const saved = JSON.parse(localStorage.getItem(storageKey))
            if (Array.isArray(saved)) {
                const enabled = new Set(saved.filter(id => knownIds.has(id)))
                const version = Number.parseInt(localStorage.getItem(defaultsVersionKey) || "0", 10)
                if (version < defaultsVersion) {
                    catalog.forEach(plugin => {
                        if (plugin.defaultEnabled && Number(plugin.addedInDefaultsVersion || 0) > version) enabled.add(plugin.id)
                    })
                    localStorage.setItem(storageKey, JSON.stringify([...enabled]))
                    localStorage.setItem(defaultsVersionKey, String(defaultsVersion))
                }
                return enabled
            }
            localStorage.setItem(defaultsVersionKey, String(defaultsVersion))
        } catch (error) {
            console.warn("Could not read optional-plugin preferences", error)
        }
        return defaults
    }

    function refreshControls() {
        const enabled = getEnabled()
        document.documentElement.toggleAttribute("data-image-modifiers-hidden", !enabled.has("image-modifiers"))
        if (!enabled.has("image-modifiers")) {
            document.querySelectorAll("dialog#modifier-settings-config[open]").forEach(dialog => dialog.close())
        }
        document.querySelectorAll(".optional-ui-plugin-settings").forEach(container => {
            for (const plugin of catalog) {
                const checkbox = container.querySelector(`[data-optional-plugin-id="${plugin.id}"]`)
                const status = container.querySelector(`[data-optional-plugin-status="${plugin.id}"]`)
                if (checkbox) checkbox.checked = enabled.has(plugin.id)
                if (status) status.textContent = container.pluginStatusFor
                    ? container.pluginStatusFor(plugin, enabled.has(plugin.id))
                    : enabled.has(plugin.id) ? "enabled" : "disabled"
            }
        })
    }
    function notify() {
        refreshControls()
        window.dispatchEvent(new CustomEvent("local-plugin-preferences-changed", {detail: {enabled: [...getEnabled()]}}))
    }
    function saveEnabled(ids) {
        const enabled = new Set([...ids].filter(id => knownIds.has(id)))
        localStorage.setItem(storageKey, JSON.stringify([...enabled]))
        notify()
        return enabled
    }
    function setEnabled(id, enabled) {
        if (!knownIds.has(id)) throw new Error(`Unknown plugin: ${id}`)
        if (typeof enabled !== "boolean") throw new TypeError("Plugin state must be a boolean")
        const ids = getEnabled()
        if (enabled) ids.add(id)
        else ids.delete(id)
        return saveEnabled(ids)
    }
    function renderControls(options = {}) {
        const container = document.createElement("div")
        container.className = "optional-ui-plugin-settings"
        container.style.cssText = options.forTab ? "max-width:900px" : "min-width:min(620px,70vw);max-height:45vh;overflow:auto"
        container.pluginStatusFor = options.getStatus
        const instance = controlInstance++
        const enabled = getEnabled()
        const errorMessage = document.createElement("p")
        errorMessage.className = "optional-plugin-error"
        errorMessage.setAttribute("role", "status")
        for (const plugin of catalog) {
            const row = document.createElement("div")
            row.className = "optional-ui-plugin-row"
            row.dataset.pluginSearch = `${plugin.name} ${plugin.id}`.toLowerCase()
            row.style.cssText = "display:grid;grid-template-columns:auto 1fr auto;gap:.65rem;align-items:center;padding:.3rem 0"
            const toggle = document.createElement("div")
            toggle.className = "input-toggle"
            const checkbox = document.createElement("input")
            checkbox.id = `optional-plugin-${instance}-${plugin.id}`
            checkbox.type = "checkbox"
            checkbox.checked = enabled.has(plugin.id)
            checkbox.dataset.optionalPluginId = plugin.id
            const switchLabel = document.createElement("label")
            switchLabel.htmlFor = checkbox.id
            switchLabel.title = `Enable or disable ${plugin.name}`
            toggle.append(checkbox, switchLabel)
            const name = document.createElement("label")
            name.htmlFor = checkbox.id
            name.textContent = plugin.name
            const status = document.createElement("small")
            status.dataset.optionalPluginStatus = plugin.id
            status.textContent = options.getStatus ? options.getStatus(plugin, checkbox.checked) : checkbox.checked ? "enabled" : "disabled"
            row.append(toggle, name, status)
            container.append(row)
            checkbox.addEventListener("change", () => {
                errorMessage.textContent = ""
                Promise.resolve().then(() => options.onChange
                    ? options.onChange(plugin, checkbox.checked)
                    : setEnabled(plugin.id, checkbox.checked)
                ).catch(error => {
                    refreshControls()
                    errorMessage.textContent = `Plugin preference was not saved: ${error.message}`
                })
            })
        }
        container.append(errorMessage)
        return container
    }
    window.addEventListener("storage", event => {
        if (event.key === storageKey || event.key === defaultsVersionKey || event.key === null) notify()
    })
    window.LocalPluginPreferences = Object.freeze({catalog, storageKey, defaultsVersionKey, defaultsVersion,
        getEnabled, saveEnabled, setEnabled, isEnabled: id => getEnabled().has(id), renderControls, refreshControls})
    refreshControls()
})()
