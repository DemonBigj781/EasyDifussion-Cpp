(async () => {
    "use strict";
    try { await window.CppKiosk.ready; } catch (_) { return; }
    const preferences = window.LocalPluginPreferences;
    const native = preferences.catalog.filter(plugin => plugin.nativePath);
    const active = native.find(plugin => plugin.nativePath === location.pathname);
    let previous;
    function refresh() {
        for (const plugin of native) {
            document.querySelectorAll(`nav a[href="${plugin.nativePath}"]`).forEach(link => {
                // Keep this separate from kiosk's `hidden` attribute.
                link.toggleAttribute("data-plugin-disabled", !preferences.isEnabled(plugin.id));
            });
        }
        if (!active) return;
        const enabled = preferences.isEnabled(active.id);
        if (previous !== undefined && previous !== enabled) {
            document.documentElement.setAttribute("data-native-plugin-disabled", "");
            location.reload();
            return;
        }
        previous = enabled;
        document.documentElement.toggleAttribute("data-native-plugin-disabled", !enabled);
        if (!enabled && !document.getElementById("cpp-plugin-disabled-notice")) {
            const notice = document.createElement("section");
            notice.id = "cpp-plugin-disabled-notice";
            notice.className = "panel-box";
            notice.textContent = `${active.name} is disabled. `;
            const link = document.createElement("a");
            link.href = "/cpp-ui/settings/plugins";
            link.textContent = "Enable it in Plugin Config.";
            notice.append(link);
            document.querySelector("#page-content > h2").after(notice);
        }
    }
    window.addEventListener("local-plugin-preferences-changed", refresh);
    refresh();
})();
