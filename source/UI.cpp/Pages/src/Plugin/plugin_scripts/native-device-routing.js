(() => {
    "use strict";

    const storageKey = "easy-diffusion-native-device-routing-v1";
    const status = document.getElementById("native-device-routing-status");
    const controls = Array.from(document.querySelectorAll("select[data-native-scope][data-native-module]"));
    if (!status || controls.length === 0) return;

    let state;
    try {
        state = JSON.parse(localStorage.getItem(storageKey) || "{}");
    } catch (_) {
        state = {};
    }
    state.image = state.image && typeof state.image === "object" ? state.image : {};
    state.video = state.video && typeof state.video === "object" ? state.video : {};

    function assignmentFor(scope) {
        return Object.entries(state[scope] || {})
            .filter(([, selector]) => typeof selector === "string" && selector)
            .map(([module, selector]) => `${module}=${selector}`).join(",");
    }

    window.NativeDeviceRouting = Object.freeze({ assignmentFor });

    async function refresh() {
        const button = document.getElementById("native-device-routing-refresh");
        if (button) button.disabled = true;
        status.dataset.state = "loading";
        status.textContent = "Loading native compute devices…";
        try {
            const response = await fetch("/get/backend_devices", { cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            const devices = Array.isArray(payload.devices) ? payload.devices : [];
            for (const select of controls) {
                const scope = select.dataset.nativeScope;
                const module = select.dataset.nativeModule;
                const selected = state[scope][module] || "";
                select.replaceChildren(new Option("Automatic", ""));
                for (const device of devices) {
                    if (!device || !device.selector) continue;
                    const memory = Number(device.memory_total);
                    const memoryLabel = memory > 0 ? ` · ${(memory / 1073741824).toFixed(memory >= 10737418240 ? 0 : 1)} GiB` : "";
                    const label = [device.description || device.selector, device.backend && String(device.backend).toUpperCase()]
                        .filter(Boolean).join(" · ") + memoryLabel;
                    select.add(new Option(label, String(device.selector)));
                }
                if (selected && !Array.from(select.options).some(option => option.value === selected)) {
                    const missing = new Option(`${selected} · unavailable in this backend`, selected);
                    missing.disabled = true;
                    select.add(missing);
                }
                select.value = selected;
                select.disabled = devices.length === 0;
            }
            status.dataset.state = "ready";
            status.textContent = devices.length
                ? `${devices.length} native compute device${devices.length === 1 ? "" : "s"} available. Selections are saved in this browser.`
                : "No native compute devices are available; all modules use Automatic.";
        } catch (error) {
            status.dataset.state = "error";
            status.textContent = `Could not load native compute devices (${error.name}).`;
            for (const select of controls) select.disabled = true;
        } finally {
            if (button) button.disabled = false;
        }
    }

    for (const select of controls) {
        select.addEventListener("change", () => {
            const scope = select.dataset.nativeScope;
            state[scope][select.dataset.nativeModule] = select.value;
            try {
                localStorage.setItem(storageKey, JSON.stringify(state));
            } catch (_) {
                status.dataset.state = "error";
                status.textContent = "Could not save device selection in this browser.";
            }
        });
    }
    document.getElementById("native-device-routing-refresh")?.addEventListener("click", refresh);
    refresh();
})();
