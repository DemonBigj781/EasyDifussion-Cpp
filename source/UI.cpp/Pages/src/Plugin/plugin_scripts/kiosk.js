(() => {
    "use strict";
    if (window.CppKiosk) return;
    const root = document.documentElement;
    const banner = document.getElementById("kiosk-status-banner");
    root.dataset.kiosk = "loading";
    const state = {enabled: false, allowed_models: []};
    window.CppKiosk = state;
    const blockedPages = ["/cpp-ui/training", "/cpp-ui/models/download", "/cpp-ui/gallery/datasets", "/cpp-ui/tagging", "/cpp-ui/perchance/image"];
    function apply(policy) {
        if (typeof policy.enabled !== "boolean" || !Array.isArray(policy.allowed_models)) throw new Error("Invalid kiosk policy");
        Object.assign(state, policy);
        root.dataset.kiosk = state.enabled ? "on" : "off";
        if (banner) {
            banner.hidden = !state.enabled;
            banner.textContent = state.enabled
                ? "Kiosk mode: base models only · LoRAs disabled · Perchance G filter · Destockd gallery" : "";
        }
        document.querySelectorAll("nav a").forEach(link => {
            if (blockedPages.includes(new URL(link.href).pathname)) link.hidden = state.enabled;
        });
        const toggle = document.getElementById("kiosk-mode-enabled");
        if (toggle) toggle.checked = state.enabled;
    }
    state.ready = fetch("/kiosk", {cache: "no-store"}).then(async response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        apply(await response.json());
        return state;
    }).catch(error => {
        root.dataset.kiosk = "unavailable";
        if (banner) { banner.hidden = false; banner.textContent = `Cannot verify kiosk policy (${error.message}). Reload the page to retry.`; }
        throw error;
    });
    state.ready.catch(() => {}); // Consumers stop initializing when the policy is unavailable.
    document.getElementById("kiosk-mode-save")?.addEventListener("click", async event => {
        const button = event.currentTarget, status = document.getElementById("kiosk-mode-save-status");
        button.disabled = true;
        status.textContent = "Saving kiosk mode…";
        try {
            const response = await fetch("/kiosk", {method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({enabled: document.getElementById("kiosk-mode-enabled").checked})});
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
            apply(result);
            status.textContent = "Kiosk mode saved. Other open pages will reload to apply the policy.";
            try { localStorage.setItem("easy-diffusion-kiosk-change", String(Date.now())); } catch (_) {}
        } catch (error) { status.textContent = `Kiosk mode was not saved: ${error.message}`; }
        finally { button.disabled = false; }
    });
    window.addEventListener("storage", event => {
        if (event.key === "easy-diffusion-kiosk-change") location.reload();
    });
    window.addEventListener("pageshow", event => { if (event.persisted) location.reload(); });
    // Also catch changes from another browser/device before leaving an old image visible.
    setInterval(async () => {
        try {
            await state.ready;
            const response = await fetch("/kiosk", {cache: "no-store"});
            if (!response.ok) throw new Error("Policy unavailable");
            const current = await response.json();
            if (typeof current.enabled !== "boolean") throw new Error("Invalid policy");
            if (current.enabled !== state.enabled) { root.dataset.kiosk = "loading"; location.reload(); }
        } catch (_) {
            root.dataset.kiosk = "unavailable";
            if (banner) { banner.hidden = false; banner.textContent = "Kiosk policy unavailable. Reload to retry."; }
        }
    }, 5000);
})();
