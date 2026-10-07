/* Portable Settings/GPU pages use the native persisted configuration API. */
(() => {
    "use strict";
    const backend = document.getElementById("backend_platform");
    const save = document.getElementById("save-system-settings-btn");
    const extra = document.getElementById("system-settings-extra");
    if (!backend || !save || !extra) return;
    const info = document.getElementById("system-info");
    const status = document.createElement("p");
    status.id = "cosmo-config-status";
    status.setAttribute("role", "status");
    extra.appendChild(status);
    function field(id, label, kind = "select") {
        const row = document.createElement("label");
        row.textContent = `${label} `;
        const control = document.createElement(kind === "select" ? "select" : "input");
        control.id = id;
        if (kind !== "select") control.type = kind;
        row.appendChild(control);
        extra.appendChild(row);
        return control;
    }
    function option(control, value, label, disabled = false) {
        const entry = document.createElement("option");
        entry.value = value; entry.textContent = label; entry.disabled = disabled;
        control.appendChild(entry);
    }
    backend.replaceChildren();
    option(backend, "cpu", "CPU");
    option(backend, "webgpu", "WebGPU");
    const provider = field("cosmo-config-provider", "WebGPU provider");
    option(provider, "auto", "Automatic: prefer physical hardware; allow embedded software");
    option(provider, "native", "Native Vulkan: physical hardware required");
    option(provider, "embedded", "Embedded software Vulkan (CPU)");
    const device = field("cosmo-config-device", "Compute device");
    const port = field("cosmo-config-port", "Server port", "number");
    port.min = "1"; port.max = "65535"; port.step = "1";
    const models = field("cosmo-config-models", "Checkpoint directory", "text");
    const log = field("cosmo-config-log", "Log level");
    for (const value of ["verbose", "debug", "info", "warning", "error"]) option(log, value, value);
    const help = document.createElement("p");
    help.textContent = "These startup settings are saved on the server and require a restart to take effect. " +
        "Relative checkpoint directories resolve beside the configuration file. WebGPU operations without a supported kernel use GGML CPU fallback. " +
        "Exact device names can change when hardware changes; stable IDs are used only when the driver supplies one.";
    extra.appendChild(help);
    const inventory = document.createElement("p");
    inventory.id = "cosmo-config-device-inventory";
    extra.appendChild(inventory);
    for (const control of document.querySelectorAll("[data-native-scope]")) {
        control.disabled = true;
        control.title = "Per-module device routing is not saved by this portable Settings page. Use the compute device above.";
    }
    const routing = document.getElementById("native-device-routing-status");
    if (routing) routing.textContent = "Use the persisted compute device above. Per-module selectors below are unavailable here.";
    const refresh = document.getElementById("native-device-routing-refresh");
    if (refresh) refresh.hidden = true;
    const localPreferences = document.getElementById("cpp-input-saving");
    if (localPreferences) localPreferences.hidden = true;
    const controls = [backend, provider, device, port, models, log];
    let state = null, entries = [], enumerationError = "", pending = false;
    function enabled(ready) {
        for (const control of controls) control.disabled = !ready;
        save.disabled = !ready || !device.value || Array.from(device.options).some(x => x.selected && x.disabled);
    }
    async function json(url, options = {}) {
        const response = await fetch(url, {cache: "no-store", ...options});
        let value;
        try { value = await response.json(); } catch (_) { throw new Error(`Invalid JSON from ${url} (HTTP ${response.status})`); }
        if (!response.ok) throw new Error(value.message || value.detail || `HTTP ${response.status}`);
        return value;
    }
    function chooseDevice(wanted = "auto") {
        device.replaceChildren();
        option(device, "auto", backend.value === "cpu" ? "CPU (automatic)" : "Automatic compatible device");
        if (backend.value === "webgpu" && state && provider.value === state.effective.compute.provider) {
            for (const item of entries.filter(x => x.backend.toLowerCase() === "webgpu")) {
                const selector = item.stable_id_available && item.stable_id ? item.stable_id : item.selector;
                const physical = item.provider === "native" && item.software === false &&
                    (item.type === "gpu" || item.type === "integrated-gpu");
                const kind = item.software === true ? "software / CPU" : physical ? "physical hardware" : "unknown adapter type";
                const label = `${item.selector}: ${item.description} — ${kind} (${item.provider})`;
                option(device, selector, label);
                if (wanted === item.selector) wanted = selector;
            }
        }
        if (backend.value === "cpu" && wanted === "CPU") wanted = "auto";
        if (!Array.from(device.options).some(x => x.value === wanted))
            option(device, wanted, `${wanted} — not available under the running provider; choose auto or restart`, true);
        device.value = wanted;
        if (!pending) enabled(!!state);
    }
    function show(value) {
        if (value.schema !== 1 || !value.saved?.compute || !value.effective?.compute || typeof value.config_path !== "string")
            throw new Error("The server did not return the native configuration schema");
        state = value;
        backend.value = value.saved.compute.backend;
        provider.value = value.saved.compute.provider;
        port.value = String(value.saved.server.port);
        models.value = value.saved.models.checkpoint_dir;
        log.value = value.saved.server.log_level;
        chooseDevice(value.saved.compute.device);
        if (info) info.textContent = `Configuration: ${value.config_path}. Running: ${value.effective.compute.backend} / ${value.effective.compute.provider} / ${value.effective.compute.device}; port ${value.effective.server.port}.`;
        status.textContent = value.restart_required ? "Settings saved; restart the server to apply startup changes." : "Saved settings match this running server.";
        inventory.textContent = enumerationError || (entries.length ? entries.map(x => `${x.selector}: ${x.description}${x.software ? " (software running on CPU)" : ""}`).join("; ") : "No compute devices were reported.");
    }
    for (const control of [backend, provider]) control.addEventListener("change", () => chooseDevice("auto"));
    device.addEventListener("change", () => enabled(!!state && !pending));
    save.addEventListener("click", async () => {
        if (!state || pending || save.disabled) return;
        const number = Number(port.value);
        if (!Number.isInteger(number) || number < 1 || number > 65535 || !models.value.trim()) {
            status.textContent = "Enter a port from 1 to 65535 and a nonempty checkpoint directory.";
            return;
        }
        pending = true; enabled(false); status.textContent = "Saving configuration…";
        try {
            show(await json("/app_config", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
                compute: {backend: backend.value, provider: provider.value, device: device.value},
                server: {port: number, log_level: log.value}, models: {checkpoint_dir: models.value}
            })}));
        } catch (error) { status.textContent = `Configuration was not saved: ${error.message}`; }
        finally { pending = false; enabled(!!state); }
    });
    enabled(false); status.textContent = "Loading native configuration…";
    (async () => {
        try {
            if (!window.CppKiosk?.ready) throw new Error("Native access capabilities were not initialized");
            await window.CppKiosk.ready;
            const value = await json("/get/app_config");
            try {
                const devices = await json("/v1/sdapi/v1/backend-devices");
                if (!Array.isArray(devices.devices)) throw new Error("Invalid device inventory");
                entries = devices.devices.filter(x => typeof x.selector === "string" && typeof x.backend === "string");
            } catch (error) { enumerationError = `Device discovery unavailable: ${error.message}. CPU or automatic policy can still be saved.`; }
            show(value); enabled(true);
        } catch (error) { state = null; status.textContent = `Configuration unavailable: ${error.message}`; enabled(false); }
    })();
})();
