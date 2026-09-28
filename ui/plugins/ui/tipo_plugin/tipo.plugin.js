(function() {
    "use strict";

    const ID_PREFIX = "z-tipo";
    const STORAGE_KEY = "z_tipo_settings_v1";
    const MODEL_PRESET_PREFIX = "model:";
    const CUSTOM_PRESET_PREFIX = "custom:";

    const DEFAULT_FORMAT = `<|special|>,
<|characters|>, <|copyrights|>,
<|artist|>,

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>`;

    const DEFAULTS = {
        tagLength: "long",
        nlLength: "long",
        temperature: "0.5",
        topP: "0.95",
        minP: "0.05",
        topK: "80",
        seed: "-1",
        device: "cuda",
        format: DEFAULT_FORMAT,
        model: "",
    };

    function normalizeCustomPresets(value) {
        if (!Array.isArray(value)) return [];
        return value.filter((preset) => (
            preset &&
            typeof preset.id === "string" &&
            typeof preset.name === "string" &&
            typeof preset.model === "string" &&
            typeof preset.format === "string"
        )).map((preset) => ({
            id: preset.id,
            name: preset.name.slice(0, 80),
            model: preset.model,
            tagLength: preset.tagLength || DEFAULTS.tagLength,
            nlLength: preset.nlLength || DEFAULTS.nlLength,
            temperature: String(preset.temperature ?? DEFAULTS.temperature),
            topP: String(preset.topP ?? DEFAULTS.topP),
            minP: String(preset.minP ?? DEFAULTS.minP),
            topK: String(preset.topK ?? DEFAULTS.topK),
            seed: String(preset.seed ?? DEFAULTS.seed),
            device: preset.device === "cpu" ? "cpu" : "cuda",
            format: preset.format,
        }));
    }

    function newPresetId() {
        return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    }

    function loadSettings() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return { ...DEFAULTS };
            const parsed = JSON.parse(raw);
            return { ...DEFAULTS, ...parsed };
        } catch (err) {
            return { ...DEFAULTS };
        }
    }

    function saveSettings(settings) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    }

    function getApiBase() {
        return "/tipo";
    }

    function dispatchInput(el) {
        ["input", "change"].forEach((evt) => {
            el.dispatchEvent(new Event(evt, { bubbles: true }));
        });
    }

    function setPrompt(text) {
        const promptField = document.querySelector("#prompt");
        if (!promptField) return;
        promptField.value = text;
        dispatchInput(promptField);
    }

    function getPrompt() {
        const promptField = document.querySelector("#prompt");
        return promptField ? promptField.value : "";
    }

    function getWidthHeight() {
        const widthField = document.querySelector("#width");
        const heightField = document.querySelector("#height");
        const width = widthField ? parseInt(widthField.value || "512", 10) : 512;
        const height = heightField ? parseInt(heightField.value || "512", 10) : 512;
        return { width, height };
    }

    async function postJson(url, bodyObj) {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyObj || {}),
        });
        if (!res.ok) {
            const text = await res.text();
            throw new Error(`${res.status} ${res.statusText} - ${text}`);
        }
        return res.json();
    }

    async function getJson(url) {
        const res = await fetch(url, { method: "GET" });
        if (!res.ok) {
            const text = await res.text();
            throw new Error(`${res.status} ${res.statusText} - ${text}`);
        }
        return res.json();
    }

    function createPanel() {
        const settings = loadSettings();

        const container = document.createElement("div");
        container.id = `${ID_PREFIX}-panel`;
        container.classList.add("panel-box");

        container.innerHTML = window.loadRequiredPluginHTML("/plugins/core/tipo_plugin/tipo.plugin.html");

        const tagsEl = container.querySelector(`#${ID_PREFIX}-tags`);
        const nlEl = container.querySelector(`#${ID_PREFIX}-nl`);
        const nlInfoEl = container.querySelector(`#${ID_PREFIX}-nl-info`);
        const banEl = container.querySelector(`#${ID_PREFIX}-ban`);
        const presetEl = container.querySelector(`#${ID_PREFIX}-preset`);
        const presetNameEl = container.querySelector(`#${ID_PREFIX}-preset-name`);
        const deletePresetEl = container.querySelector(`#${ID_PREFIX}-delete-preset`);
        const modelEl = container.querySelector(`#${ID_PREFIX}-model`);
        const deviceEl = container.querySelector(`#${ID_PREFIX}-device`);
        const modelInfoEl = container.querySelector(`#${ID_PREFIX}-model-info`);
        const tagLengthEl = container.querySelector(`#${ID_PREFIX}-tag-length`);
        const nlLengthEl = container.querySelector(`#${ID_PREFIX}-nl-length`);
        const tempEl = container.querySelector(`#${ID_PREFIX}-temp`);
        const seedEl = container.querySelector(`#${ID_PREFIX}-seed`);
        const topPEl = container.querySelector(`#${ID_PREFIX}-top-p`);
        const minPEl = container.querySelector(`#${ID_PREFIX}-min-p`);
        const topKEl = container.querySelector(`#${ID_PREFIX}-top-k`);
        const formatEl = container.querySelector(`#${ID_PREFIX}-format`);
        const statusEl = container.querySelector(`#${ID_PREFIX}-status`);
        const outputEl = container.querySelector(`#${ID_PREFIX}-output`);
        const outputRawEl = container.querySelector(`#${ID_PREFIX}-output-raw`);

        tagLengthEl.value = settings.tagLength;
        nlLengthEl.value = settings.nlLength;
        tempEl.value = settings.temperature;
        seedEl.value = settings.seed;
        topPEl.value = settings.topP;
        minPEl.value = settings.minP;
        topKEl.value = settings.topK;
        deviceEl.value = settings.device;
        formatEl.value = settings.format;

        let availableModels = [];
        let modelMetadata = {};
        let customPresets = normalizeCustomPresets(settings.presets);

        function updateSettings() {
            saveSettings({
                tagLength: tagLengthEl.value,
                nlLength: nlLengthEl.value,
                temperature: tempEl.value,
                topP: topPEl.value,
                minP: minPEl.value,
                topK: topKEl.value,
                seed: seedEl.value,
                device: deviceEl.value,
                format: formatEl.value,
                model: modelEl.value,
                presetId: presetEl.value,
                presets: customPresets,
            });
        }

        function modelPresetValue(model) {
            return `${MODEL_PRESET_PREFIX}${model}`;
        }

        function customPresetValue(id) {
            return `${CUSTOM_PRESET_PREFIX}${id}`;
        }

        function selectedCustomPreset() {
            if (!presetEl.value.startsWith(CUSTOM_PRESET_PREFIX)) return null;
            const id = presetEl.value.slice(CUSTOM_PRESET_PREFIX.length);
            return customPresets.find((preset) => preset.id === id) || null;
        }

        function updatePresetControls() {
            const preset = selectedCustomPreset();
            deletePresetEl.disabled = !preset;
            presetNameEl.value = preset ? preset.name : "";
        }

        function appendPresetOption(parent, value, label) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = label;
            parent.appendChild(option);
        }

        function renderPresets(selectedValue = "") {
            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = "Current custom settings";
            presetEl.replaceChildren(placeholder);

            if (availableModels.length) {
                const modelGroup = document.createElement("optgroup");
                modelGroup.label = "Available model defaults";
                availableModels.forEach((model) => {
                    appendPresetOption(modelGroup, modelPresetValue(model), model);
                });
                presetEl.appendChild(modelGroup);
            }
            if (customPresets.length) {
                const customGroup = document.createElement("optgroup");
                customGroup.label = "Saved custom presets";
                customPresets.forEach((preset) => {
                    appendPresetOption(customGroup, customPresetValue(preset.id), preset.name);
                });
                presetEl.appendChild(customGroup);
            }

            const exists = Array.from(presetEl.options).some((option) => option.value === selectedValue);
            presetEl.value = exists ? selectedValue : "";
            updatePresetControls();
        }

        function detachPreset() {
            if (presetEl.value.startsWith(MODEL_PRESET_PREFIX)) {
                presetEl.value = "";
                updatePresetControls();
            }
            updateSettings();
        }

        function getSelectedProtocol() {
            const metadata = modelMetadata[modelEl.value];
            return metadata && !metadata.error ? (metadata.protocol || "tipo") : "tipo";
        }

        function updateProtocolFields() {
            const supportsNaturalLanguage = getSelectedProtocol() === "tipo";
            nlEl.disabled = !supportsNaturalLanguage;
            nlLengthEl.disabled = !supportsNaturalLanguage;
            nlEl.title = supportsNaturalLanguage
                ? "Used by TIPO for sentence-to-tag and tag-to-sentence generation."
                : "DanTagGen accepts tag conditioning only.";
            nlLengthEl.title = nlEl.title;
            nlInfoEl.textContent = supportsNaturalLanguage
                ? "Used by native TIPO models as short/long text conditioning."
                : "Unavailable for DanTagGen, which accepts tag conditioning only.";
        }

        function applyModelSidecar() {
            const metadata = modelMetadata[modelEl.value];
            if (!metadata || metadata.error) {
                modelInfoEl.textContent = metadata && metadata.error ? metadata.error : "No sidecar; inferred defaults";
                updateProtocolFields();
                updateSettings();
                return;
            }
            tagLengthEl.value = metadata.tag_length || DEFAULTS.tagLength;
            nlLengthEl.value = metadata.nl_length || DEFAULTS.nlLength;
            tempEl.value = metadata.temperature ?? DEFAULTS.temperature;
            topPEl.value = metadata.top_p ?? DEFAULTS.topP;
            minPEl.value = metadata.min_p ?? DEFAULTS.minP;
            topKEl.value = metadata.top_k ?? DEFAULTS.topK;
            formatEl.value = metadata.format || DEFAULT_FORMAT;
            const sidecarLabel = metadata.sidecar ? ` · ${metadata.sidecar}` : " · inferred";
            modelInfoEl.textContent = `${metadata.protocol || "tipo"}${sidecarLabel}`;
            updateProtocolFields();
            updateSettings();
        }

        function applyCustomPreset(preset) {
            if (!preset || !availableModels.includes(preset.model)) return false;
            modelEl.value = preset.model;
            tagLengthEl.value = preset.tagLength;
            nlLengthEl.value = preset.nlLength;
            tempEl.value = preset.temperature;
            topPEl.value = preset.topP;
            minPEl.value = preset.minP;
            topKEl.value = preset.topK;
            seedEl.value = preset.seed;
            deviceEl.value = preset.device;
            formatEl.value = preset.format;
            const metadata = modelMetadata[preset.model];
            const sidecarLabel = metadata?.sidecar ? ` · ${metadata.sidecar}` : " · preset override";
            modelInfoEl.textContent = `${metadata?.protocol || "tipo"}${sidecarLabel}`;
            updateProtocolFields();
            updateSettings();
            return true;
        }

        [
            tagLengthEl,
            nlLengthEl,
            tempEl,
            seedEl,
            topPEl,
            minPEl,
            topKEl,
            deviceEl,
            formatEl,
        ].forEach((el) => el.addEventListener("change", detachPreset));

        modelEl.addEventListener("change", () => {
            if (!selectedCustomPreset()) presetEl.value = modelPresetValue(modelEl.value);
            updatePresetControls();
            applyModelSidecar();
        });

        presetEl.addEventListener("change", () => {
            if (presetEl.value.startsWith(MODEL_PRESET_PREFIX)) {
                const model = presetEl.value.slice(MODEL_PRESET_PREFIX.length);
                if (availableModels.includes(model)) {
                    modelEl.value = model;
                    applyModelSidecar();
                }
            } else {
                const preset = selectedCustomPreset();
                if (preset && !applyCustomPreset(preset)) {
                    statusEl.textContent = `Preset model is unavailable: ${preset.model}`;
                }
            }
            updatePresetControls();
            updateSettings();
        });

        container.querySelector(`#${ID_PREFIX}-save-preset`).addEventListener("click", () => {
            const name = presetNameEl.value.trim();
            if (!name) {
                statusEl.textContent = "Enter a name for the custom preset.";
                presetNameEl.focus();
                return;
            }
            if (!modelEl.value || !availableModels.includes(modelEl.value)) {
                statusEl.textContent = "Choose an available model before saving a preset.";
                return;
            }

            const selected = selectedCustomPreset();
            let index = selected ? customPresets.findIndex((preset) => preset.id === selected.id) : -1;
            if (index < 0) {
                index = customPresets.findIndex((preset) => preset.name.toLocaleLowerCase() === name.toLocaleLowerCase());
            }
            const preset = {
                id: index >= 0 ? customPresets[index].id : newPresetId(),
                name: name.slice(0, 80),
                model: modelEl.value,
                tagLength: tagLengthEl.value,
                nlLength: nlLengthEl.value,
                temperature: tempEl.value,
                topP: topPEl.value,
                minP: minPEl.value,
                topK: topKEl.value,
                seed: seedEl.value,
                device: deviceEl.value,
                format: formatEl.value,
            };
            if (index >= 0) customPresets[index] = preset;
            else customPresets.push(preset);
            renderPresets(customPresetValue(preset.id));
            updateSettings();
            statusEl.textContent = `Preset "${preset.name}" saved in this browser.`;
        });

        deletePresetEl.addEventListener("click", () => {
            const preset = selectedCustomPreset();
            if (!preset || !window.confirm(`Delete TIPO preset "${preset.name}"?`)) return;
            customPresets = customPresets.filter((candidate) => candidate.id !== preset.id);
            const fallback = availableModels.includes(modelEl.value) ? modelPresetValue(modelEl.value) : "";
            renderPresets(fallback);
            updateSettings();
            statusEl.textContent = `Preset "${preset.name}" deleted.`;
        });

        container.querySelector(`#${ID_PREFIX}-use-prompt`).addEventListener("click", () => {
            tagsEl.value = getPrompt();
        });

        container.querySelector(`#${ID_PREFIX}-apply`).addEventListener("click", () => {
            if (outputEl.value.trim() !== "") {
                setPrompt(outputEl.value.trim());
            }
        });

        container.querySelector(`#${ID_PREFIX}-apply-raw`).addEventListener("click", () => {
            if (outputRawEl.value.trim() !== "") {
                setPrompt(outputRawEl.value.trim());
            }
        });

        async function reloadModels() {
            const apiBase = getApiBase();
            const currentModel = modelEl.value;
            const currentPreset = presetEl.value;
            statusEl.textContent = "Loading models...";
            try {
                const data = await getJson(`${apiBase}/models`);
                const models = Array.isArray(data.models) ? data.models : [];
                availableModels = models;
                modelMetadata = data.metadata && typeof data.metadata === "object" ? data.metadata : {};
                modelEl.innerHTML = "";
                if (models.length === 0) {
                    const opt = document.createElement("option");
                    opt.value = "";
                    opt.textContent = "No models found";
                    modelEl.appendChild(opt);
                    renderPresets();
                } else {
                    models.forEach((model) => {
                        const opt = document.createElement("option");
                        opt.value = model;
                        opt.textContent = model;
                        modelEl.appendChild(opt);
                    });
                    if (currentModel && models.includes(currentModel)) {
                        modelEl.value = currentModel;
                    } else if (settings.model && models.includes(settings.model)) {
                        modelEl.value = settings.model;
                    }
                    const initialPresetValue = currentPreset || settings.presetId || modelPresetValue(modelEl.value);
                    renderPresets(initialPresetValue);
                    const customPreset = selectedCustomPreset();
                    if (!customPreset || !applyCustomPreset(customPreset)) {
                        presetEl.value = modelPresetValue(modelEl.value);
                        updatePresetControls();
                        applyModelSidecar();
                    }
                }
                statusEl.textContent = `Loaded ${models.length} models`;
            } catch (err) {
                statusEl.textContent = "TIPO server not reachable";
                console.warn("TIPO model load failed:", err);
            }
        }

        container.querySelector(`#${ID_PREFIX}-reload`).addEventListener("click", () => {
            reloadModels();
        });

        container.querySelector(`#${ID_PREFIX}-generate`).addEventListener("click", async () => {
            const apiBase = getApiBase();
            const { width, height } = getWidthHeight();
            statusEl.textContent = "Generating...";

            const payload = {
                tipo_model: modelEl.value,
                tags: tagsEl.value,
                nl_prompt: getSelectedProtocol() === "tipo" ? nlEl.value : "",
                ban_tags: banEl.value,
                format: formatEl.value,
                seed: parseInt(seedEl.value || "-1", 10),
                temperature: parseFloat(tempEl.value || "0.5"),
                top_p: parseFloat(topPEl.value || "0.95"),
                min_p: parseFloat(minPEl.value || "0.05"),
                top_k: parseInt(topKEl.value || "80", 10),
                tag_length: tagLengthEl.value,
                nl_length: nlLengthEl.value,
                width,
                height,
                device: deviceEl.value,
            };

            try {
                const data = await postJson(`${apiBase}/generate`, payload);
                outputEl.value = data.formatted_prompt || "";
                outputRawEl.value = data.unformatted_prompt || "";
                statusEl.textContent = "Done";
            } catch (err) {
                statusEl.textContent = "Generation failed";
                console.warn("TIPO generation failed:", err);
            }
        });

        reloadModels();
        return container;
    }

    function attachPanel() {
        if (document.querySelector(`#${ID_PREFIX}-panel`)) return;
        const editorInputs = document.querySelector("#editor-inputs");
        if (!editorInputs) return;
        const panel = createPanel();
        panel.style.flex = "0 0 auto";
        const separator =
            document.querySelector("#editor-inputs span.line-separator") ||
            document.querySelector("span.line-separator");
        if (separator && separator.parentNode) {
            separator.insertAdjacentElement("afterend", panel);
        } else {
            editorInputs.insertAdjacentElement("afterend", panel);
        }
        if (typeof createCollapsibles === "function") {
            createCollapsibles(panel);
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attachPanel);
    } else {
        attachPanel();
    }
})();
