(async () => {
    "use strict";
    const grid = document.getElementById("perchance-generator-gallery-results");
    if (!grid) return;
    let policy;
    try { policy = await window.CppKiosk.ready; } catch (_) { return; }
    if (!window.LocalPluginPreferences.isEnabled("perchance-gallery")) return;
    const status = document.querySelector("[data-perchance-gallery-status]");
    const list = document.getElementById("perchance-generator-gallery-list-button");
    const get = document.getElementById("perchance-generator-gallery-get-button");
    function showStatus(text) { if (status) status.textContent = text; }
    showStatus(policy.enabled ? "Kiosk: Perchance's G filter is enforced. Unrated and PG-13 content is excluded by the source filter."
        : "Choose List Gallery or enter an image ID.");
    async function load(single) {
        list.disabled = get.disabled = true;
        grid.replaceChildren();
        showStatus("Loading Perchance gallery…");
        try {
            const payload = {channel: document.getElementById("perchance-generator-gallery-channel").value,
                content_filter: policy.enabled ? "g" : "none", download: false, visible: false,
                limit: Number(document.getElementById("perchance-generator-gallery-limit").value),
                sort: document.getElementById("perchance-generator-gallery-sort").value};
            if (single) payload.gallery_id = document.getElementById("perchance-generator-gallery-id").value.trim();
            const response = await fetch(single ? "/perchance/gallery/get" : "/perchance/gallery/list",
                {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            if (policy.enabled && data.content_filter !== "g") throw new Error("Unverified Perchance response was blocked.");
            const entries = single ? [data] : data.entries;
            if (!Array.isArray(entries)) throw new Error("Invalid gallery response");
            let displayed = 0;
            for (const entry of entries) {
                if (typeof entry.preview_data_url !== "string" || !entry.preview_data_url.startsWith("data:image/jpeg;base64,")) continue;
                const card = document.createElement("figure"); card.className = "cpp-perchance-card";
                const image = document.createElement("img"); image.src = entry.preview_data_url;
                image.alt = "Perchance gallery image";
                const caption = document.createElement("figcaption"); caption.textContent = String(entry.prompt || "Perchance image");
                card.append(image, caption); grid.append(card); displayed++;
            }
            showStatus(`${displayed} image${displayed === 1 ? "" : "s"} displayed${policy.enabled ? " · G-filtered" : ""}${data.suppressed_count ? ` · ${data.suppressed_count} unverified or higher-rated images hidden` : ""}${displayed < entries.length ? " · some previews were unavailable" : ""}.`);
        } catch (error) { grid.replaceChildren(); showStatus(`Gallery unavailable: ${error.message}`); }
        finally { list.disabled = get.disabled = false; }
    }
    list.addEventListener("click", () => load(false));
    get.addEventListener("click", () => load(true));
})();
