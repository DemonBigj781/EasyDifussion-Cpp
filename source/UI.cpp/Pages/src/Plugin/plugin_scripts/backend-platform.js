(() => {
    "use strict";

    const select = document.getElementById("backend_platform");
    const save = document.getElementById("save-system-settings-btn");
    if (!select || !save) return;

    fetch("/get/app_config", { cache: "no-store" })
        .then((response) => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then((config) => {
            const platform = config.backend_config?.platform || "auto";
            select.value = Array.from(select.options).some((option) => option.value === platform)
                ? platform
                : "auto";
        })
        .catch((error) => console.warn("Could not load backend platform setting", error));

    save.addEventListener("click", async () => {
        try {
            const response = await fetch("/app_config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ backend_platform: select.value }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
        } catch (error) {
            console.error("Could not save backend platform setting", error);
            window.alert(`Backend platform was not saved: ${error.message}`);
        }
    });
})();
