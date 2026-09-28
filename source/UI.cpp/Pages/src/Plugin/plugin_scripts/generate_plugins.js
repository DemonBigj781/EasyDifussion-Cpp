(() => {
    "use strict";
    if (window.CppGeneratePlugins) return;

    const loraRoot = document.getElementById("lora_model");
    const mode = document.getElementById("controlnet_mode");
    const standard = document.getElementById("controlnet-mode-standard");
    const enabled = document.getElementById("controlnet_enabled");
    const loraKey = "easy-diffusion-cpp-lora-v1";
    const controlKey = "easy-diffusion-controlnet-mode-v1";
    const readJson = (key, fallback) => {
        try { return JSON.parse(localStorage.getItem(key) || JSON.stringify(fallback)); }
        catch (_) { return fallback; }
    };

    const storedLoras = readJson(loraKey, []);
    const loraEntries = [];
    function addLoraRow(initial = {}) {
        if (!loraRoot) return;
        const row = document.createElement("div");
        row.className = "model_entry cpp-lora-entry";
        const select = document.createElement("select");
        select.className = "model_name";
        select.setAttribute("aria-label", "LoRA model");
        select.add(new Option("Select a LoRA", ""));
        for (const item of window.CppGeneratePlugins.loraModels || []) {
            const value = String(item.model || item.name || item.rel_path || "");
            if (value) select.add(new Option(value, value));
        }
        select.value = initial.name || "";
        const weight = document.createElement("input");
        weight.className = "model_weight";
        weight.type = "number"; weight.min = "-2"; weight.max = "2"; weight.step = "0.02";
        weight.value = initial.weight ?? 0.5; weight.setAttribute("aria-label", "LoRA strength");
        const remove = document.createElement("button");
        remove.type = "button"; remove.textContent = "Remove";
        remove.addEventListener("click", () => { row.remove(); saveLoras(); });
        for (const control of [select, weight]) control.addEventListener("change", saveLoras);
        row.append(select, weight, remove); loraRoot.append(row);
        loraEntries.push({ row, select, weight });
    }
    function saveLoras() {
        const entries = loraEntries.filter(({row}) => row.isConnected)
            .map(({select, weight}) => ({name: select.value, weight: Number(weight.value)}))
            .filter(item => item.name);
        try { localStorage.setItem(loraKey, JSON.stringify(entries)); } catch (_) {}
    }
    const loraModels = [];
    window.CppGeneratePlugins = {
        loraModels,
        requestOptions() {
            const entries = loraEntries.filter(({row}) => row.isConnected && row.select.value);
            const request = {};
            if (entries.length) {
                request.use_lora_model = entries.length === 1 ? entries[0].select.value : entries.map(({select}) => select.value);
                request.lora_alpha = entries.length === 1 ? Number(entries[0].weight.value) : entries.map(({weight}) => Number(weight.value));
            }
            if (enabled?.checked && document.getElementById("controlnet_model")?.value) {
                request.use_controlnet_model = document.getElementById("controlnet_model").value;
                request.control_alpha = Number(document.getElementById("controlnet_alpha")?.value || 1);
                request.controlnet_union_type = document.getElementById("controlnet_union_type")?.value || "canny";
                const image = document.getElementById("control_image_preview");
                if (image?.src && image.dataset.ready === "true") request.control_image = image.src;
            }
            return request;
        },
        applyToRequest(request) {
            if (!request || typeof request !== "object") return request;
            Object.assign(request, this.requestOptions());
            return request;
        },
        refresh: loadModels,
    };

    async function loadModels() {
        if (!loraRoot) return;
        loraRoot.textContent = "Loading LoRA models…";
        try {
            const response = await fetch("/get/lora", {cache:"no-store"});
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            loraModels.splice(0, loraModels.length, ...(Array.isArray(payload.models) ? payload.models : []));
            loraRoot.replaceChildren();
            loraEntries.length = 0;
            for (const item of storedLoras) addLoraRow(item);
            if (!storedLoras.length) addLoraRow();
            const add = document.createElement("button"); add.type = "button"; add.textContent = "Add LoRA";
            add.addEventListener("click", () => addLoraRow()); loraRoot.append(add);
        } catch (error) {
            loraRoot.textContent = `Could not load LoRA models (${error.name}).`;
        }
    }

    function setMode(value, persist = true) {
        const valid = ["off", "standard", "auto", "lllite"];
        const next = valid.includes(value) ? value : "off";
        if (mode) mode.value = next;
        if (enabled) enabled.checked = next === "standard";
        if (standard) standard.hidden = next !== "standard";
        if (persist) try { localStorage.setItem(controlKey, next); } catch (_) {}
        document.dispatchEvent(new CustomEvent("controlnetModeChanged", {detail:{mode:next}}));
        return next;
    }
    mode?.addEventListener("change", () => setMode(mode.value));
    let savedMode = "off";
    try { savedMode = localStorage.getItem(controlKey) || "off"; } catch (_) {}
    setMode(savedMode, false);

    async function loadControlnetModels() {
        const datalist = document.getElementById("controlnet-model-options");
        if (!datalist) return;
        try {
            const response = await fetch("/get/models", {cache:"no-store"});
            if (!response.ok) return;
            const payload = await response.json();
            const models = Array.isArray(payload.models) ? payload.models : [];
            for (const item of models) {
                const tags = Array.isArray(item.tags) ? item.tags : [];
                if (!tags.some(tag => String(tag).includes("controlnet"))) continue;
                const value = String(item.model || item.name || "");
                if (value) datalist.append(new Option(value, value));
            }
        } catch (_) {}
    }
    const controlModel = document.getElementById("controlnet_model");
    controlModel?.addEventListener("change", () => {
        try { localStorage.setItem("easy-diffusion-cpp-controlnet-model-v1", controlModel.value); } catch (_) {}
    });
    try {
        if (controlModel) controlModel.value = localStorage.getItem("easy-diffusion-cpp-controlnet-model-v1") || "";
    } catch (_) {}
    document.getElementById("controlnet_union_type")?.addEventListener("change", event => {
        try { localStorage.setItem("easy-diffusion-cpp-controlnet-type-v1", event.target.value); } catch (_) {}
    });
    try {
        const savedType = localStorage.getItem("easy-diffusion-cpp-controlnet-type-v1");
        if (savedType && document.getElementById("controlnet_union_type"))
            document.getElementById("controlnet_union_type").value = savedType;
    } catch (_) {}
    loadControlnetModels();

    const range = document.getElementById("controlnet_alpha_slider");
    const strength = document.getElementById("controlnet_alpha");
    range?.addEventListener("input", () => { if (strength) strength.value = Number(range.value) / 10; });
    strength?.addEventListener("input", () => { if (range) range.value = String(Number(strength.value) * 10); });
    document.getElementById("control_image_file")?.addEventListener("change", event => {
        const file = event.target.files?.[0]; const image = document.getElementById("control_image_preview");
        if (!file || !image) return;
        const reader = new FileReader();
        reader.onload = () => { image.src = String(reader.result); image.dataset.ready = "true"; image.hidden = false; };
        reader.readAsDataURL(file);
    });
    loadModels();
})();
