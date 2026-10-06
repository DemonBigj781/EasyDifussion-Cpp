(async () => {
    "use strict";
    if (window.CppGeneratePlugins) return;
    let policy;
    try { policy = await window.CppKiosk.ready; } catch (_) { return; }

    const loraRoot = document.getElementById("lora_model");
    const mode = document.getElementById("controlnet_mode");
    const standard = document.getElementById("controlnet-mode-standard");
    const enabled = document.getElementById("controlnet_enabled");
    const llliteSection = document.createElement("div");
    llliteSection.id = "controlnet-mode-lllite";
    llliteSection.innerHTML = '<label for="lllite-model">LLLite model<input id="lllite-model" class="model-filter" data-path="" autocomplete="off"></label>'
        + '<label for="lllite-strength">Strength<input id="lllite-strength" type="number" min="-10" max="10" step="0.1" value="1"></label>'
        + '<label for="lllite-start">Start (%)<input id="lllite-start" type="number" min="0" max="100" value="0"></label>'
        + '<label for="lllite-end">End (%)<input id="lllite-end" type="number" min="0" max="100" value="100"></label>'
        + '<p id="lllite-model-status" role="status">Choose a checkpoint to see compatible RGB conditioning models. Upload its control image below; no automatic preprocessing is applied.</p>';
    document.getElementById("controlnet-extension-modes").append(llliteSection);
    const llliteModel = new ModelDropdown(document.getElementById("lllite-model"), null, "None");
    let availableModels = [];
    function compatibleLLLite(item) {
        const checkpoint = document.getElementById("stable_diffusion_model")?.dataset.path;
        const tags = availableModels.find(item => item.model === checkpoint)?.tags || [];
        const family = tags.includes("anima") ? "anima" : tags.some(tag => /^sd_(v1|v2|xl)/.test(tag)) ? "unet" : null;
        const compatibility = item?.lllite_compatibility;
        return family && compatibility?.architecture === family && compatibility.input_channels === 3 && !compatibility.aspp;
    }
    function refreshLLLite() {
        const items = availableModels.filter(item => item.tags?.includes("controlnet-lllite") && compatibleLLLite(item));
        const previous = llliteModel.value || window.CppInputPreferences.get("lllite_model") || "";
        llliteModel.inputModels = buildTree(items);
        llliteModel.populateModels();
        llliteModel.value = items.some(item => item.model === previous) ? previous : "";
        document.getElementById("lllite-model-status").textContent = `${items.length} compatible RGB LLLite models. Inpainting weights require a separate mask path and are not enabled here.`;
    }
    document.getElementById("stable_diffusion_model")?.addEventListener("change", refreshLLLite);
    const loraKey = "easy-diffusion-cpp-lora-v1";
    const controlKey = "easy-diffusion-controlnet-mode-v1";
    const preferences = window.CppInputPreferences;
    llliteModel.addEventListener("change", () => preferences.save("lllite_model", llliteModel.value));
    const readLoras = () => {
        try {
            const value = JSON.parse(preferences.get("lora_model", loraKey) || "[]");
            if (Array.isArray(value)) return value;
            return (value.modelNames || []).map((name, index) => ({name, weight: value.modelWeights?.[index] ?? .5}));
        } catch (_) { return []; }
    };

    const storedLoras = policy.enabled ? [] : readLoras();
    if (policy.enabled && loraRoot) {
        loraRoot.replaceChildren();
        const panel = loraRoot.closest(".generate-plugin-panel");
        if (panel) panel.hidden = true;
    }
    const loraEntries = [];
    let loraCounter = 0;
    function addLoraRow(initial = {}) {
        if (!loraRoot || policy.enabled) return;
        const row = document.createElement("div");
        row.className = "model_entry cpp-lora-entry";
        const select = document.createElement("input");
        select.type = "text";
        select.id = `cpp-lora-${loraCounter++}`;
        select.autocomplete = "off";
        select.dataset.path = initial.name || "";
        select.className = "model_name model-filter";
        select.setAttribute("aria-label", "LoRA model");
        const weight = document.createElement("input");
        weight.className = "model_weight";
        weight.type = "number"; weight.min = "-2"; weight.max = "2"; weight.step = "0.02";
        weight.value = initial.weight ?? 0.5; weight.setAttribute("aria-label", "LoRA strength");
        const remove = document.createElement("button");
        remove.type = "button"; remove.textContent = "Remove";
        remove.addEventListener("click", () => { row.remove(); saveLoras(); });
        for (const control of [select, weight]) control.addEventListener("change", saveLoras);
        row.append(select, weight, remove); loraRoot.append(row);
        select.field = new ModelDropdown(select, null, "None");
        select.field.inputModels = buildTree(window.CppGeneratePlugins.loraModels.map(item => ({model: String(item.model || item.name || item.rel_path || "")})));
        select.field.populateModels();
        loraEntries.push({ row, select, weight });
    }
    function saveLoras() {
        const entries = loraEntries.filter(({row}) => row.isConnected)
            .map(({select, weight}) => ({name: select.field.value, weight: Number(weight.value)}))
            .filter(item => item.name);
        preferences.save("lora_model", JSON.stringify({modelNames: entries.map(item => item.name), modelWeights: entries.map(item => item.weight)}));
    }
    const loraModels = [];
    window.CppGeneratePlugins = {
        loraModels,
        requestOptions() {
            if (policy.enabled) return {};
            const entries = loraEntries.filter(({row, select}) => row.isConnected && select.field.value);
            const request = {};
            if (mode?.value === "lllite") {
                const selected = availableModels.find(item => item.model === llliteModel.value);
                const image = document.getElementById("control_image_preview");
                if (!compatibleLLLite(selected)) throw new Error("Choose an LLLite model compatible with the checkpoint.");
                if (!image || image.dataset.ready !== "true") throw new Error("Upload an LLLite conditioning image first.");
                const readNumber = (id, min, max) => {
                    const value = Number(document.getElementById(id).value);
                    if (!Number.isFinite(value) || value < min || value > max) throw new Error(`Invalid LLLite setting: ${id}`);
                    return value;
                };
                request.control_net_lllite_model = llliteModel.value;
                request.control_net_lllite_image = image.src;
                request.control_net_lllite_strength = readNumber("lllite-strength", -10, 10);
                request.control_net_lllite_start_percent = readNumber("lllite-start", 0, 100);
                request.control_net_lllite_end_percent = readNumber("lllite-end", 0, 100);
                if (request.control_net_lllite_end_percent <= request.control_net_lllite_start_percent)
                    throw new Error("LLLite end must be greater than start.");
            }
            if (entries.length) {
                request.use_lora_model = entries.length === 1 ? entries[0].select.field.value : entries.map(({select}) => select.field.value);
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
        if (!loraRoot || policy.enabled) return;
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
        llliteSection.hidden = next !== "lllite";
        if (persist) preferences.save("controlnet_mode", next);
        document.dispatchEvent(new CustomEvent("controlnetModeChanged", {detail:{mode:next}}));
        return next;
    }
    mode?.addEventListener("change", () => setMode(mode.value));
    let savedMode = "off";
    savedMode = preferences.get("controlnet_mode", controlKey) || "off";
    setMode(savedMode, false);

    async function loadControlnetModels() {
        const datalist = document.getElementById("controlnet-model-options");
        if (!datalist) return;
        try {
            const response = await fetch("/get/models", {cache:"no-store"});
            if (!response.ok) return;
            const payload = await response.json();
            const models = Array.isArray(payload.models) ? payload.models : [];
            availableModels = models;
            refreshLLLite();
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
        preferences.save("controlnet_model", controlModel.value);
    });
    try {
        if (controlModel) controlModel.value = preferences.get("controlnet_model", "easy-diffusion-cpp-controlnet-model-v1") || "";
    } catch (_) {}
    document.getElementById("controlnet_union_type")?.addEventListener("change", event => {
        preferences.save("controlnet_union_type", event.target.value);
    });
    try {
        const savedType = preferences.get("controlnet_union_type", "easy-diffusion-cpp-controlnet-type-v1");
        if (savedType && document.getElementById("controlnet_union_type"))
            document.getElementById("controlnet_union_type").value = savedType;
    } catch (_) {}
    loadControlnetModels();

    const range = document.getElementById("controlnet_alpha_slider");
    const strength = document.getElementById("controlnet_alpha");
    range?.addEventListener("input", () => {
        if (strength) {
            strength.value = Number(range.value) / 10;
            strength.dispatchEvent(new Event("input", {bubbles: true}));
        }
    });
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
