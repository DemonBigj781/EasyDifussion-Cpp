(async () => {
    "use strict";
    if (window.CppImageGallery) return;
    let policy;
    try { policy = await window.CppKiosk.ready; } catch (_) { return; }
    window.CppImageGallery = true;

    const byId = id => document.getElementById(id);
    const container = byId("gallery-container");
    if (!container) return;
    const pageSize = 60;
    let page = 1, totalPages = 1, currentRecords = [], lightboxIndex = 0;
    const selected = new Set();
    const records = new Map();

    function message(text, error = false) {
        const node = byId("gallery-source");
        if (node) { node.textContent = text; node.dataset.state = error ? "error" : ""; }
    }
    function selectionChanged() {
        const count = selected.size;
        const selection = byId("gallery-selection-count");
        if (selection) selection.textContent = count ? `Selected: ${count}` : "";
        for (const id of ["gallery-collage-horizontal", "gallery-collage-vertical", "gallery-collage-grid", "gallery-delete-selected"]) {
            const button = byId(id); if (button) button.disabled = count === 0;
        }
        container.querySelectorAll(".cpp-gallery-card").forEach(card => card.classList.toggle("selected", selected.has(card.dataset.id)));
    }
    function safeUrl(path, route = "file") {
        return `/gallery/${route}/${String(path).split("/").map(encodeURIComponent).join("/")}`;
    }
    async function loadGallery() {
        const refresh = byId("gallery-refresh");
        if (refresh) refresh.disabled = true;
        container.replaceChildren();
        byId("gallery-lightbox")?.close();
        byId("gallery-lightbox-image")?.removeAttribute("src");
        message(policy.enabled ? "Loading Destockd film frames…" : "Loading configured gallery directory…");
        try {
            const response = await fetch(`/gallery/images?page=${page}&page_size=${pageSize}`, {cache:"no-store"});
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || `Gallery request failed (${response.status})`);
            if (policy.enabled && data.source !== "destockd") throw new Error("Kiosk gallery source mismatch; local images were blocked.");
            page = Number(data.page || 1); totalPages = Number(data.total_pages || 1);
            currentRecords = Array.isArray(data.images) ? data.images : [];
            for (const item of currentRecords) records.set(item.id, item);
            byId("gallery-count").textContent = String(data.total || 0);
            byId("gallery-page-status").textContent = `Page ${page} of ${totalPages}`;
            byId("gallery-prev-page").disabled = !data.has_previous;
            byId("gallery-next-page").disabled = !data.has_next;
            byId("gallery-empty").hidden = currentRecords.length !== 0;
            const range = data.total ? `images ${data.start_index}–${data.end_index} of ${data.total}` : "0 images";
            message(data.source === "destockd" ? `Destockd — ${range}. Refresh for another selection. Source filtering is not a PG certification.`
                : `${data.directory || "Gallery directory"} — ${range}${data.exists ? "" : " (directory unavailable)"}`, !data.exists);
            const directory = byId("gallery-directory-input");
            if (directory && document.activeElement !== directory) directory.value = data.directory || "";
            container.replaceChildren();
            for (const item of currentRecords) {
                const card = document.createElement("figure");
                card.className = "cpp-gallery-card";
                card.dataset.id = item.id;
                card.classList.toggle("selected", selected.has(item.id));
                const top = document.createElement("div"); top.className = "cpp-gallery-card-tools";
                const checkbox = document.createElement("input"); checkbox.type = "checkbox";
                checkbox.checked = selected.has(item.id); checkbox.setAttribute("aria-label", `Select ${item.filename}`);
                checkbox.addEventListener("change", () => {
                    if (checkbox.checked) selected.add(item.id); else selected.delete(item.id);
                    selectionChanged();
                });
                const name = document.createElement("span"); name.textContent = item.filename;
                if (!policy.enabled) top.append(checkbox);
                top.append(name);
                const image = document.createElement("img"); image.src = item.thumbnail_url || item.url;
                image.alt = item.filename; image.loading = "lazy"; image.title = "Open full-size preview";
                image.addEventListener("click", () => openLightbox(currentRecords.indexOf(item)));
                const download = document.createElement("a");
                download.href = policy.enabled ? item.source_url : item.url;
                if (policy.enabled) { download.target = "_blank"; download.rel = "noopener noreferrer"; download.textContent = "Source: Destockd"; }
                else { download.download = item.filename; download.textContent = "Download"; }
                card.append(top, image, download); container.append(card);
            }
            selectionChanged();
            applyZoom();
        } catch (error) {
            container.replaceChildren();
            currentRecords = []; selected.clear(); records.clear(); selectionChanged();
            message(error.message || "Could not load gallery.", true);
            byId("gallery-empty").hidden = false;
        } finally { if (refresh) refresh.disabled = false; }
    }
    function applyZoom() {
        const zoom = Math.max(25, Math.min(200, Number(byId("gallery-zoom-slider")?.value || 100)));
        container.style.setProperty("--gallery-card-min", `${Math.round(150 * zoom / 100)}px`);
    }
    async function loadSettings() {
        try {
            const response = await fetch("/gallery/settings", {cache:"no-store"});
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || `Settings request failed (${response.status})`);
            byId("gallery-directory-input").value = data.gallery_directory || "";
            byId("gallery-directory-status").textContent = data.exists ? "Directory is available." : "Directory does not exist.";
        } catch (error) { byId("gallery-directory-status").textContent = error.message; }
    }
    async function saveSettings() {
        if (policy.enabled) return;
        const input = byId("gallery-directory-input"), button = byId("gallery-directory-save"), status = byId("gallery-directory-status");
        button.disabled = true; status.textContent = "Saving directory…";
        try {
            const response = await fetch("/gallery/settings", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({gallery_directory:input.value.trim()})});
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || `Could not save directory (${response.status})`);
            input.value = data.gallery_directory; page = 1; selected.clear(); records.clear();
            status.textContent = "Gallery directory saved."; await loadGallery();
        } catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
    }
    async function deleteSelected() {
        if (policy.enabled) return;
        if (!selected.size || !window.confirm(`Permanently delete ${selected.size} selected image${selected.size === 1 ? "" : "s"}? This cannot be undone.`)) return;
        const button = byId("gallery-delete-selected"); button.disabled = true;
        try {
            for (const id of Array.from(selected)) {
                const item = records.get(id);
                if (!item) continue;
                const response = await fetch(item.url || safeUrl(item.relative_path), {method:"DELETE"});
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || `Delete failed (${response.status})`);
                records.delete(id); selected.delete(id);
            }
            await loadGallery();
        } catch (error) { message(error.message, true); }
        finally { button.disabled = false; selectionChanged(); }
    }
    function openLightbox(index) {
        lightboxIndex = Math.max(0, Math.min(index, currentRecords.length - 1)); updateLightbox();
        byId("gallery-lightbox")?.showModal();
    }
    function updateLightbox() {
        const item = currentRecords[lightboxIndex]; if (!item) return;
        byId("gallery-lightbox-image").src = item.url;
        byId("gallery-lightbox-counter").textContent = `${lightboxIndex + 1} / ${currentRecords.length} · ${item.filename}`;
    }
    function downloadCanvas(canvas, mode) {
        canvas.toBlob(blob => {
            if (!blob) { message("Could not create collage.", true); return; }
            const url = URL.createObjectURL(blob), link = document.createElement("a");
            link.href = url; link.download = `gallery-collage-${mode}-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
            link.click(); URL.revokeObjectURL(url);
        }, "image/png");
    }
    async function createCollage(mode) {
        if (policy.enabled) return;
        const items = Array.from(selected, id => records.get(id)).filter(Boolean);
        if (!items.length) return;
        if (items.length > 50) { message("Select 50 or fewer images for a collage.", true); return; }
        message("Building collage…");
        try {
            const images = await Promise.all(items.map(item => new Promise((resolve, reject) => {
                const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(new Error(`Could not load ${item.filename}`)); image.src = item.url;
            })));
            const gap = 8, maxDimension = 4096, canvas = document.createElement("canvas");
            let ctx;
            if (mode === "horizontal" || mode === "vertical") {
                const scaled = images.map(image => {
                    const scale = mode === "horizontal" ? 512 / image.height : 512 / image.width;
                    return {image, width: Math.max(1, Math.round(image.width * scale)), height: Math.max(1, Math.round(image.height * scale))};
                });
                let width = mode === "horizontal" ? scaled.reduce((n, x) => n + x.width, 0) + gap * (scaled.length - 1) : Math.max(...scaled.map(x => x.width));
                let height = mode === "vertical" ? scaled.reduce((n, x) => n + x.height, 0) + gap * (scaled.length - 1) : Math.max(...scaled.map(x => x.height));
                const scaleAll = Math.min(1, maxDimension / width, maxDimension / height);
                canvas.width = Math.max(1, Math.round(width * scaleAll)); canvas.height = Math.max(1, Math.round(height * scaleAll)); ctx = canvas.getContext("2d");
                let offset = 0;
                for (const item of scaled) {
                    const w = item.width * scaleAll, h = item.height * scaleAll;
                    ctx.drawImage(item.image, mode === "horizontal" ? offset : (canvas.width - w) / 2, mode === "vertical" ? offset : (canvas.height - h) / 2, w, h);
                    offset += (mode === "horizontal" ? w : h) + gap * scaleAll;
                }
            } else {
                const columns = Math.ceil(Math.sqrt(images.length)), rows = Math.ceil(images.length / columns), cell = 512;
                canvas.width = columns * cell; canvas.height = rows * cell; ctx = canvas.getContext("2d");
                ctx.fillStyle = "#202329"; ctx.fillRect(0, 0, canvas.width, canvas.height);
                images.forEach((image, i) => {
                    const x = (i % columns) * cell, y = Math.floor(i / columns) * cell;
                    const scale = Math.min((cell - gap * 2) / image.width, (cell - gap * 2) / image.height, 1);
                    const w = image.width * scale, h = image.height * scale;
                    ctx.drawImage(image, x + (cell - w) / 2, y + (cell - h) / 2, w, h);
                });
            }
            downloadCanvas(canvas, mode); message(`Created ${mode} collage from ${images.length} images.`);
        } catch (error) { message(error.message || "Could not create collage.", true); }
    }

    byId("gallery-refresh")?.addEventListener("click", () => loadGallery());
    byId("gallery-directory-save")?.addEventListener("click", saveSettings);
    byId("gallery-directory-input")?.addEventListener("keydown", event => { if (event.key === "Enter") saveSettings(); });
    byId("gallery-select-all")?.addEventListener("click", () => { currentRecords.forEach(item => { selected.add(item.id); records.set(item.id, item); }); selectionChanged(); container.querySelectorAll("input[type=checkbox]").forEach(box => { box.checked = true; }); });
    byId("gallery-deselect-all")?.addEventListener("click", () => { selected.clear(); selectionChanged(); container.querySelectorAll("input[type=checkbox]").forEach(box => { box.checked = false; }); });
    byId("gallery-prev-page")?.addEventListener("click", () => { if (page > 1) { page--; loadGallery(); } });
    byId("gallery-next-page")?.addEventListener("click", () => { if (page < totalPages) { page++; loadGallery(); } });
    byId("gallery-delete-selected")?.addEventListener("click", deleteSelected);
    byId("gallery-collage-horizontal")?.addEventListener("click", () => createCollage("horizontal"));
    byId("gallery-collage-vertical")?.addEventListener("click", () => createCollage("vertical"));
    byId("gallery-collage-grid")?.addEventListener("click", () => createCollage("grid"));
    byId("gallery-zoom-slider")?.addEventListener("input", applyZoom);
    byId("gallery-lightbox-close")?.addEventListener("click", () => byId("gallery-lightbox").close());
    byId("gallery-lightbox-prev")?.addEventListener("click", () => { lightboxIndex = (lightboxIndex - 1 + currentRecords.length) % currentRecords.length; updateLightbox(); });
    byId("gallery-lightbox-next")?.addEventListener("click", () => { lightboxIndex = (lightboxIndex + 1) % currentRecords.length; updateLightbox(); });
    byId("gallery-lightbox")?.addEventListener("click", event => { if (event.target === byId("gallery-lightbox")) byId("gallery-lightbox").close(); });
    document.addEventListener("keydown", event => {
        if (!byId("gallery-lightbox")?.open) return;
        if (event.key === "ArrowLeft") byId("gallery-lightbox-prev").click();
        if (event.key === "ArrowRight") byId("gallery-lightbox-next").click();
    });
    if (policy.enabled) loadGallery();
    else loadSettings().finally(loadGallery);
})();
