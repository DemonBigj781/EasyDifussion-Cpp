(async () => {
    "use strict";
    const host = document.getElementById("cpp-plugin-manager-controls");
    if (!host) return;
    try { await window.CppKiosk.ready; } catch (_) { return; }
    const preferences = window.LocalPluginPreferences;
    host.append(preferences.renderControls({forTab: true,
        getStatus: (plugin, enabled) => `${enabled ? "enabled" : "disabled"}${plugin.port === "native" ? "" : " · legacy UI"}`,
    }));
    document.getElementById("cpp-plugin-filter").addEventListener("input", event => {
        const query = event.target.value.trim().toLowerCase();
        host.querySelectorAll(".optional-ui-plugin-row").forEach(row => {
            row.style.display = !query || row.dataset.pluginSearch.includes(query) ? "grid" : "none";
        });
    });
})();
