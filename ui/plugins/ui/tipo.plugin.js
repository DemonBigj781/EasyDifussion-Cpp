(function() {
    "use strict";

    const ID_PREFIX = "z-tipo";
    const STORAGE_KEY = "z_tipo_settings_v1";

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
        device: "cpu",
        format: DEFAULT_FORMAT,
        model: "",
    };

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

        container.innerHTML = `
            <h4 class="collapsible">TIPO Prompt Generator</h4>
            <div class="collapsible-content" style="display:block;margin-top:15px;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px;">
                <div id="${ID_PREFIX}-status" style="font-size: 12px; opacity: 0.8;">Idle</div>
            </div>
            <div style="display: grid; gap: 10px; grid-template-columns: 1fr 1fr;">
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <label style="font-size: 12px;">Tag Prompt</label>
                    <textarea id="${ID_PREFIX}-tags" rows="4" placeholder="1girl, looking at viewer, ..."></textarea>
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <label style="font-size: 12px;">TIPO Natural Language Prompt</label>
                    <textarea id="${ID_PREFIX}-nl" rows="4" placeholder="Describe the scene in sentences"></textarea>
                    <small id="${ID_PREFIX}-nl-info" style="opacity:0.75;">Used by native TIPO models as short/long text conditioning.</small>
                </div>
            </div>
            <div style="margin-top: 10px; display: grid; gap: 10px; grid-template-columns: 1fr 1fr 1fr;">
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <label style="font-size: 12px;">Ban Tags (comma-separated)</label>
                    <input id="${ID_PREFIX}-ban" type="text" placeholder="nsfw, lowres" />
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <label style="font-size: 12px;">Model</label>
                    <select id="${ID_PREFIX}-model"></select>
                    <small id="${ID_PREFIX}-model-info" style="opacity:0.75;"></small>
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <label style="font-size: 12px;">Device</label>
                    <select id="${ID_PREFIX}-device">
                        <option value="cpu">cpu</option>
                        <option value="cuda">cuda</option>
                    </select>
                </div>
            </div>
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; font-size: 12px;">Advanced Options</summary>
                <div style="margin-top: 10px; display: grid; gap: 10px; grid-template-columns: repeat(4, 1fr);">
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <label style="font-size: 12px;">Tag Length</label>
                        <select id="${ID_PREFIX}-tag-length">
                            <option value="very_short">very short</option>
                            <option value="short">short</option>
                            <option value="long">long</option>
                            <option value="very_long">very long</option>
                        </select>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <label style="font-size: 12px;">NL Length</label>
                        <select id="${ID_PREFIX}-nl-length">
                            <option value="very_short">very short</option>
                            <option value="short">short</option>
                            <option value="long">long</option>
                            <option value="very_long">very long</option>
                        </select>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <label style="font-size: 12px;">Temperature</label>
                        <input id="${ID_PREFIX}-temp" type="number" step="0.01" min="0" max="2" />
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <label style="font-size: 12px;">Seed (-1 random)</label>
                        <input id="${ID_PREFIX}-seed" type="number" step="1" />
                    </div>
                </div>
                <div style="margin-top: 10px; display: grid; gap: 10px; grid-template-columns: repeat(3, 1fr);">
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <label style="font-size: 12px;">Top P</label>
                        <input id="${ID_PREFIX}-top-p" type="number" step="0.01" min="0" max="1" />
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <label style="font-size: 12px;">Min P</label>
                        <input id="${ID_PREFIX}-min-p" type="number" step="0.01" min="0" max="1" />
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <label style="font-size: 12px;">Top K</label>
                        <input id="${ID_PREFIX}-top-k" type="number" step="1" min="1" />
                    </div>
                </div>
                <div style="margin-top: 10px; display: flex; flex-direction: column; gap: 6px;">
                    <label style="font-size: 12px;">Prompt Format</label>
                    <textarea id="${ID_PREFIX}-format" rows="6"></textarea>
                </div>
                <div style="margin-top: 10px; display: flex; align-items: center; gap: 8px;">
                    <button id="${ID_PREFIX}-reload" class="secondaryButton">Reload Models</button>
                </div>
            </details>
            <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                <button id="${ID_PREFIX}-use-prompt" class="secondaryButton">Use Current Prompt as Tags</button>
                <button id="${ID_PREFIX}-generate" class="primaryButton">Generate with TIPO</button>
                <button id="${ID_PREFIX}-apply" class="secondaryButton">Apply Formatted Output</button>
                <button id="${ID_PREFIX}-apply-raw" class="secondaryButton">Apply Unformatted Output</button>
            </div>
            <div style="margin-top: 10px; display: grid; gap: 10px; grid-template-columns: 1fr 1fr;">
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <label style="font-size: 12px;">Formatted Output</label>
                    <textarea id="${ID_PREFIX}-output" rows="4" readonly></textarea>
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <label style="font-size: 12px;">Unformatted Output</label>
                    <textarea id="${ID_PREFIX}-output-raw" rows="4" readonly></textarea>
                </div>
            </div>
            </div>
        `;

        const tagsEl = container.querySelector(`#${ID_PREFIX}-tags`);
        const nlEl = container.querySelector(`#${ID_PREFIX}-nl`);
        const nlInfoEl = container.querySelector(`#${ID_PREFIX}-nl-info`);
        const banEl = container.querySelector(`#${ID_PREFIX}-ban`);
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
            });
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
        ].forEach((el) => el.addEventListener("change", updateSettings));

        let modelMetadata = {};

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

        modelEl.addEventListener("change", applyModelSidecar);

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
            statusEl.textContent = "Loading models...";
            try {
                const data = await getJson(`${apiBase}/models`);
                const models = Array.isArray(data.models) ? data.models : [];
                modelMetadata = data.metadata && typeof data.metadata === "object" ? data.metadata : {};
                modelEl.innerHTML = "";
                if (models.length === 0) {
                    const opt = document.createElement("option");
                    opt.value = "";
                    opt.textContent = "No models found";
                    modelEl.appendChild(opt);
                } else {
                    models.forEach((model) => {
                        const opt = document.createElement("option");
                        opt.value = model;
                        opt.textContent = model;
                        modelEl.appendChild(opt);
                    });
                    if (settings.model && models.includes(settings.model)) {
                        modelEl.value = settings.model;
                    }
                    applyModelSidecar();
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
