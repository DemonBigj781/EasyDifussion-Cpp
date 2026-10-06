(() => {
    "use strict";
    const root = document.getElementById("cpp-image-modifiers");
    if (!root || window.CppImageModifiers) return;

    const grid = root.querySelector("#cpp-modifier-grid");
    const tags = root.querySelector("#editor-inputs-tags-list");
    const search = root.querySelector("#cpp-modifier-search");
    const category = root.querySelector("#cpp-modifier-category");
    const status = root.querySelector("#cpp-modifier-status");
    const clear = root.querySelector("#cpp-modifier-clear");
    const retry = root.querySelector("#cpp-modifier-retry");
    const storageKey = "easy-diffusion-cpp-modifiers-v1";
    const selected = new Set();
    let catalog = [], shown = 0;

    function saveSelection() {
        try { localStorage.setItem(storageKey, JSON.stringify([...selected])); } catch (_) {}
    }
    function updateStatus() {
        status.dataset.state = "ready";
        status.textContent = !catalog.length ? "No image modifiers available."
            : !shown ? `No modifiers match your filters. ${selected.size} selected.`
            : `${shown} of ${catalog.length} modifiers · ${selected.size} selected`;
    }
    function updateSelection() {
        tags.replaceChildren();
        for (const name of selected) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "cpp-modifier-tag";
            button.textContent = `${name} ×`;
            button.setAttribute("aria-label", `Remove ${name}`);
            button.addEventListener("click", () => toggle(name));
            tags.append(button);
        }
        for (const button of grid.querySelectorAll(".cpp-modifier-card")) {
            button.setAttribute("aria-pressed", String(selected.has(button.dataset.modifier)));
        }
        clear.disabled = selected.size === 0;
        updateStatus();
    }
    function toggle(name) {
        if (selected.has(name)) selected.delete(name);
        else selected.add(name);
        saveSelection();
        updateSelection();
    }
    function thumbnailPath(item) {
        const previews = Array.isArray(item.previews) ? item.previews : [];
        const preview = previews.find(value => value.name === "landscape") || previews[0];
        if (typeof preview?.path !== "string" || !preview.path) return null;
        const prefix = "/media/modifier-thumbnails/";
        const url = new URL(prefix + preview.path, location.origin);
        return url.origin === location.origin && url.pathname.startsWith(prefix) ? url.href : null;
    }
    function renderGrid() {
        const query = search.value.trim().toLocaleLowerCase();
        const items = catalog.filter(item => (!category.value || category.value === item.category)
            && `${item.modifier} ${item.category}`.toLocaleLowerCase().includes(query));
        const fragment = document.createDocumentFragment();
        for (const item of items) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "cpp-modifier-card";
            button.dataset.modifier = item.modifier;
            button.setAttribute("aria-label", item.modifier);
            button.setAttribute("aria-pressed", String(selected.has(item.modifier)));
            const preview = thumbnailPath(item);
            if (preview) {
                const image = document.createElement("img");
                image.src = preview;
                image.alt = "";
                image.loading = "lazy";
                image.decoding = "async";
                image.addEventListener("error", () => { image.hidden = true; }, {once: true});
                button.append(image);
            }
            const name = document.createElement("span");
            name.className = "cpp-modifier-name";
            name.textContent = item.modifier;
            button.append(name);
            button.addEventListener("click", () => toggle(item.modifier));
            fragment.append(button);
        }
        grid.replaceChildren(fragment);
        shown = items.length;
        updateStatus();
    }
    async function load() {
        status.textContent = "Loading image modifiers…";
        status.dataset.state = "loading";
        grid.setAttribute("aria-busy", "true");
        retry.hidden = true;
        retry.disabled = true;
        try {
            const response = await fetch("/get/modifiers", {cache: "no-store"});
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            if (!Array.isArray(payload) || payload.some(group => typeof group.category !== "string"
                || !Array.isArray(group.modifiers) || group.modifiers.some(item => typeof item.modifier !== "string"))) {
                throw new Error("Invalid modifier catalog");
            }
            catalog = payload.flatMap(group => group.modifiers.map(item => ({...item, category: group.category})));
            category.replaceChildren(new Option("All categories", ""));
            for (const group of payload) category.append(new Option(group.category, group.category));
            const available = new Set(catalog.map(item => item.modifier));
            let stored = [];
            try { stored = JSON.parse(localStorage.getItem(storageKey) || "[]"); } catch (_) {}
            selected.clear();
            if (Array.isArray(stored)) for (const name of stored) if (available.has(name)) selected.add(name);
            renderGrid();
            updateSelection();
        } catch (error) {
            status.dataset.state = "error";
            status.textContent = `Could not load image modifiers (${error.message}).`;
            retry.hidden = false;
        } finally {
            grid.setAttribute("aria-busy", "false");
            retry.disabled = false;
        }
    }
    window.CppImageModifiers = {
        applyToRequest(request) {
            const original = request.prompt || "";
            const names = [...selected];
            request.original_prompt = original;
            request.active_tags = names;
            request.inactive_tags = [];
            request.prompt = [original.trim(), ...names].filter(Boolean).join(", ");
        },
    };
    search.addEventListener("input", renderGrid);
    category.addEventListener("change", renderGrid);
    clear.addEventListener("click", () => { selected.clear(); saveSelection(); updateSelection(); });
    retry.addEventListener("click", load);
    load();
})();
