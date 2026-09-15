// Per-module native backend routing. Controls live in System Settings.

;(function () {
    "use strict"
    if (window.NativeDeviceRouting) return

    const STATE_KEY = "easy-diffusion-native-device-routing-v1"
    const SETTINGS_ID = "native-device-routing-settings"
    const mountedGroups = new Set()
    const selectSets = new Map()
    let devices = []

    const IMAGE_MODULES = [
        ["diffusion", "KSampler / denoising"],
        ["te", "Conditioning / text encoders"],
        ["llm", "LLM conditioning"],
        ["vae_encode", "VAE encoding"],
        ["vae_decode", "VAE decoding"],
        ["controlnet", "ControlNet / LLLite"],
        ["clip_vision", "CLIP-Vision encoding"],
        ["ip_adapter", "IP-Adapter projection"],
        ["photomaker", "Identity encoding"],
        ["upscaler", "Latent / model upscaling"],
        ["detector", "Detection / detailing"],
        ["latent_interposer_encode", "Encode latent interposer"],
        ["latent_interposer_decode", "Decode latent interposer"],
    ]

    const VIDEO_MODULES = [
        ["diffusion", "KSampler / video denoising"],
        ["te", "Conditioning / text encoders"],
        ["llm", "LLM conditioning"],
        ["vae_encode", "Video VAE encoding"],
        ["vae_decode", "Video VAE decoding"],
        ["clip_vision", "Image / CLIP-Vision encoding"],
        ["latent_interposer_encode", "Encode latent interposer"],
        ["latent_interposer_decode", "Decode latent interposer"],
    ]

    function readState() {
        try { return JSON.parse(localStorage.getItem(STATE_KEY) || "{}") }
        catch (_) { return {} }
    }

    const state = readState()
    state.image = state.image && typeof state.image === "object" ? state.image : {}
    state.video = state.video && typeof state.video === "object" ? state.video : {}

    function saveState() {
        localStorage.setItem(STATE_KEY, JSON.stringify(state))
    }

    function humanBytes(bytes) {
        const value = Number(bytes)
        if (!Number.isFinite(value) || value <= 0) return ""
        return `${(value / 1073741824).toFixed(value >= 10737418240 ? 0 : 1)} GiB`
    }

    function deviceLabel(device) {
        const description = String(device.description || device.selector || "device")
        const backend = String(device.backend || "").toUpperCase()
        const memory = humanBytes(device.memory_total)
        return [description, backend, memory].filter(Boolean).join(" · ")
    }

    function selectKey(scope, module) {
        return `${scope}:${module}`
    }

    function populateSelect(select) {
        const current = select.value || select.dataset.savedValue || ""
        select.replaceChildren(new Option("Automatic", ""))
        for (const device of devices) {
            if (!device?.selector) continue
            select.appendChild(new Option(deviceLabel(device), String(device.selector)))
        }
        const exists = Array.from(select.options).some((option) => option.value === current)
        if (!exists && current) {
            const missing = new Option(`${current} · unavailable in this backend`, current)
            missing.disabled = true
            select.appendChild(missing)
        }
        select.value = current
        select.disabled = devices.length === 0
        select.title = devices.length
            ? "Select the native compute device for this inference module."
            : "No native backend devices are currently available."
    }

    function synchronize(scope, module, value, source) {
        state[scope] = state[scope] || {}
        state[scope][module] = value
        for (const select of selectSets.get(selectKey(scope, module)) || []) {
            if (select !== source) select.value = value
        }
        saveState()
    }

    function createSelect(scope, module, label) {
        const select = document.createElement("select")
        select.className = "native-device-select"
        select.dataset.scope = scope
        select.dataset.module = module
        select.dataset.savedValue = state[scope]?.[module] || ""
        select.setAttribute("aria-label", `${label} device`)
        const key = selectKey(scope, module)
        if (!selectSets.has(key)) selectSets.set(key, new Set())
        selectSets.get(key).add(select)
        select.addEventListener("change", () => synchronize(scope, module, select.value, select))
        populateSelect(select)
        return select
    }

    function mountGroup(container, scope, descriptors, heading = "Compute devices") {
        if (!container) return null
        const groupKey = `${scope}:${container.closest("[id]")?.id || container.id}:${descriptors.map((item) => item.module).join(",")}`
        if (mountedGroups.has(groupKey)) return null
        mountedGroups.add(groupKey)

        const section = document.createElement("div")
        section.className = "native-device-routing-group"
        const title = document.createElement("strong")
        title.className = "native-device-routing-title"
        title.textContent = heading
        section.appendChild(title)
        const grid = document.createElement("div")
        grid.className = "native-device-routing-grid"
        for (const descriptor of descriptors) {
            const label = document.createElement("label")
            label.textContent = descriptor.label
            const select = createSelect(scope, descriptor.module, descriptor.label)
            label.htmlFor = `${scope}-${descriptor.module}-device-${selectSets.get(selectKey(scope, descriptor.module)).size}`
            select.id = label.htmlFor
            grid.append(label, select)
        }
        section.appendChild(grid)
        container.appendChild(section)
        return section
    }

    async function refresh() {
        try {
            const response = await fetch("/get/backend_devices", { cache: "no-store" })
            if (!response.ok) throw new Error(`HTTP ${response.status}`)
            const payload = await response.json()
            devices = Array.isArray(payload.devices) ? payload.devices : []
        } catch (error) {
            devices = []
            console.warn("Native device discovery failed", error)
        }
        for (const selects of selectSets.values()) {
            for (const select of selects) populateSelect(select)
        }
        document.dispatchEvent(new CustomEvent("nativeBackendDevices", { detail: { devices } }))
        return devices
    }

    function assignmentFor(scope) {
        const assignments = state[scope] || {}
        return Object.entries(assignments)
            .filter(([, selector]) => typeof selector === "string" && selector)
            .map(([module, selector]) => `${module}=${selector}`)
            .join(",")
    }

    window.NativeDeviceRouting = Object.freeze({ mountGroup, refresh, assignmentFor })

    if (!document.getElementById("native-device-routing-style")) {
        const style = document.createElement("style")
        style.id = "native-device-routing-style"
        style.textContent = `
            #system-settings-table > [data-setting-id="native_device_routing"] { flex-wrap: wrap; }
            #system-settings-table > [data-setting-id="native_device_routing"] > div:nth-child(2) { min-width: 0; }
            #system-settings-table > [data-setting-id="native_device_routing"] > div:nth-child(3) {
                box-sizing: border-box;
                flex: 0 0 100%;
                justify-content: stretch;
                padding: 0 14px 14px 75px;
                text-align: left;
                width: 100%;
            }
            .native-device-routing-settings { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr)); gap: 12px; min-width: 0; width: 100%; }
            .native-device-routing-group { margin: 0; padding: 10px; border: 1px solid var(--background-color3, #444); border-radius: var(--input-border-radius, 6px); }
            .native-device-routing-title { display: block; margin-bottom: 8px; }
            .native-device-routing-grid { display: grid; grid-template-columns: minmax(145px, auto) minmax(180px, 1fr); gap: 7px 10px; align-items: center; }
            .native-device-routing-grid select { width: 100%; min-width: 0; }
            @media (max-width: 760px) {
                #system-settings-table > [data-setting-id="native_device_routing"] > div:nth-child(3) { padding: 0 10px 14px; }
                .native-device-routing-settings, .native-device-routing-grid { grid-template-columns: 1fr; }
            }
        `
        document.head.appendChild(style)
    }

    function mountSettings() {
        const settingsTable = document.getElementById("system-settings-table")
        if (!settingsTable || document.getElementById(SETTINGS_ID)) return Boolean(settingsTable)

        const row = document.createElement("div")
        row.id = SETTINGS_ID
        row.dataset.settingId = "native_device_routing"
        row.innerHTML = `
            <div><i class="fa fa-microchip"></i></div>
            <div>
                <label>Native compute devices</label>
                <small>Choose a backend device for each native image or video module. Automatic follows the backend defaults.</small>
            </div>`

        const controls = document.createElement("div")
        controls.className = "native-device-routing-settings"
        const imageControls = document.createElement("div")
        imageControls.id = "native-device-routing-image-settings"
        const videoControls = document.createElement("div")
        videoControls.id = "native-device-routing-video-settings"
        controls.append(imageControls, videoControls)
        row.appendChild(controls)
        settingsTable.appendChild(row)

        mountGroup(
            imageControls,
            "image",
            IMAGE_MODULES.map(([module, label]) => ({ module, label })),
            "Image pipeline"
        )
        mountGroup(
            videoControls,
            "video",
            VIDEO_MODULES.map(([module, label]) => ({ module, label })),
            "Video pipeline"
        )
        return true
    }

    mountSettings()

    PLUGINS.TASK_BUILD.push(function (event) {
        event.reqBody.backend_assignment = assignmentFor("image")
    })

    refresh()
    document.addEventListener("refreshModels", refresh)
})()
