(() => {
    "use strict";
    if (window.CppKiosk) return;
    const root = document.documentElement;
    const banner = document.getElementById("kiosk-status-banner");
    root.dataset.kiosk = "loading";
    const state = {enabled: false, supported: false, allowed_models: []};
    window.CppKiosk = state;
    for (const id of ["kiosk-mode-enabled", "kiosk-mode-save"]) {
        const control = document.getElementById(id);
        if (control) {
            control.disabled = true;
            control.title = "Kiosk configuration is unavailable in this native build.";
        }
    }
    const settingStatus = document.getElementById("kiosk-mode-save-status");
    if (settingStatus) settingStatus.textContent = "This native build has no kiosk configuration. Its local generation interface is unrestricted.";
    state.ready = fetch("/v1/sdapi/v1/cosmopolitan-capabilities", {cache: "no-store"})
        .then(async response => {
            if (!response.ok) throw new Error(`Native capabilities returned HTTP ${response.status}`);
            const capabilities = await response.json();
            if (capabilities.protocol !== 1 || capabilities.mode !== "native-single-user" ||
                capabilities.kiosk_supported !== false || capabilities.kiosk_enabled !== false) {
                throw new Error("The server did not confirm the supported native access mode");
            }
            state.capabilities = capabilities;
            root.dataset.kiosk = "off";
            root.dataset.nativeMode = capabilities.mode;
            if (banner) {
                banner.hidden = false;
                banner.textContent = "Native local mode: one image request at a time. Kiosk restrictions and configuration are not supported.";
            }
            return state;
        }).catch(error => {
            root.dataset.kiosk = "unavailable";
            if (banner) {
                banner.hidden = false;
                banner.textContent = `${error.message}. Generation is disabled; reload after restoring the native server.`;
            }
            throw error;
        });
    state.ready.catch(() => {});
})();
