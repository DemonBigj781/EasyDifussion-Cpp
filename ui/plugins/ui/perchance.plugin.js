(function () {
    "use strict";

    const ID_PREFIX = "perchance-generator";
    const STORAGE_KEY = "perchance_generator_settings_v1";
    const PRESET_ENDING = " State the prompt as is.";
    const TEXT_PROMPT_PRESETS = [
        {
            id: "general-image-expander",
            label: "General image prompt expander",
            starter:
                "Expand, optimize, and improve the following image prompt. Return only the finished prompt, with a clear subject, setting, composition, lighting, color, mood, camera perspective, and useful visual details. Preserve the original intent, remove repetition and contradictions, and do not add explanations." + PRESET_ENDING,
        },
        {
            id: "danbooru-sdxl",
            label: "Danbooru tags for SDXL",
            starter:
                "Expand, optimize, and improve the following image prompt for an SDXL-based model using the Danbooru tag system. Return only one comma-separated list of concise Danbooru-style tags, ordered from the main subject and defining traits to pose, expression, clothing, environment, composition, camera, lighting, and style. Preserve the original intent, add useful visual detail, remove duplicates and contradictions, and avoid prose, explanations, unsupported details, and artist or model names." + PRESET_ENDING,
        },
        {
            id: "natural-language-sdxl",
            label: "Natural language for SDXL",
            starter:
                "Expand, optimize, and improve the following image prompt for an SDXL-based model using fluent natural language. Return only the final polished prompt as coherent descriptive prose. Clearly describe the main subject, important attributes, action, environment, composition, perspective, lighting, color, mood, and style while preserving the original intent. Avoid tag lists, headings, explanations, repetition, contradictions, and unsupported details." + PRESET_ENDING,
        },
        {
            id: "cinematic-scene",
            label: "Cinematic scene enhancer",
            starter:
                "Rewrite and expand the following image prompt as a cinematic scene. Return only the final prompt. Strengthen visual storytelling with shot type, camera angle, lens feel, subject placement, depth, lighting direction, atmosphere, color grading, environment, and mood while preserving the original subject and intent. Avoid explanations, repetition, and contradictory details." + PRESET_ENDING,
        },
        {
            id: "character-design",
            label: "Character design builder",
            starter:
                "Expand the following image prompt into a production-ready character design prompt. Return only the final prompt. Preserve the character concept while adding coherent details for appearance, age range, build, face, hair, expression, pose, clothing, materials, accessories, color palette, setting, composition, and lighting. Avoid explanations, repetition, copyrighted character names, and contradictory traits." + PRESET_ENDING,
        },
        {
            id: "negative-prompt",
            label: "SDXL negative prompt builder",
            starter:
                "Create a concise, optimized negative prompt for an SDXL-based image model from the following desired image prompt. Return only a comma-separated list of likely unwanted visual qualities, anatomy errors, composition problems, artifacts, and conflicting elements. Use only relevant exclusions, avoid excessive boilerplate, and do not exclude anything explicitly requested by the original prompt." + PRESET_ENDING,
        },
    ];
    const LEGACY_PRESET_SUFFIX = "\n\nInput prompt:";
    const REQUIRED_TEXT_FILTERS = [
        "**Negitive Prompt:**",
        "**Positive Prompt:**",
    ];
    let busy = false;

    function element(suffix) {
        return document.querySelector(`#${ID_PREFIX}-${suffix}`);
    }

    function dispatchInput(target) {
        ["input", "change"].forEach((eventName) => {
            target.dispatchEvent(new Event(eventName, { bubbles: true }));
        });
    }

    function autoResizeTextarea(target) {
        if (target.offsetParent === null) return;
        target.style.height = "auto";
        target.style.overflowY = "hidden";
        const measuredHeight = target.scrollHeight;
        if (measuredHeight > 0) {
            target.style.height = `${measuredHeight}px`;
        }
    }

    function autoResizeAllTextareas() {
        document.querySelectorAll(
            `#${ID_PREFIX}-image-panel textarea, #${ID_PREFIX}-text-panel textarea`
        ).forEach(autoResizeTextarea);
    }

    function bindTextareaAutoResize() {
        document.querySelectorAll(
            `#${ID_PREFIX}-image-panel textarea, #${ID_PREFIX}-text-panel textarea`
        ).forEach((textarea) => {
            textarea.addEventListener("input", () => autoResizeTextarea(textarea));
            autoResizeTextarea(textarea);
        });

        document.querySelectorAll(
            `#${ID_PREFIX}-image-panel details, #${ID_PREFIX}-text-panel details`
        ).forEach((details) => {
            details.addEventListener("toggle", () => {
                if (details.open) {
                    requestAnimationFrame(autoResizeAllTextareas);
                }
            });
        });

        window.addEventListener("resize", autoResizeAllTextareas);
    }

    function loadSettings() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
        } catch (_) {
            return {};
        }
    }

    function saveSettings() {
        const values = {
            imagePrompt: element("image-prompt").value,
            shape: element("shape").value,
            negativePrompt: element("negative-prompt").value,
            seed: element("seed").value,
            guidanceScale: element("guidance-scale").value,
            textPrompt: element("text-prompt").value,
            textPreset: element("text-preset").value,
            startWith: element("start-with").value,
            stops: element("stops").value,
            filterStrings: element("filters").value,
            timeout: element("timeout").value,
            galleryId: element("gallery-id").value,
            galleryChannel: element("gallery-channel").value,
            galleryContentFilter: element("gallery-content-filter").value,
            galleryLimit: element("gallery-limit").value,
            galleryCursor: element("gallery-cursor").value,
            gallerySort: element("gallery-sort").value,
            galleryTimeRange: element("gallery-time-range").value,
            galleryDownload: element("gallery-download").checked,
            galleryVisible: element("gallery-visible").checked,
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(values));
    }

    function setGalleryRating(value) {
        const select = element("gallery-content-filter");
        const rating = String(value || "g").trim() || "g";
        select.querySelectorAll("option[data-custom-rating]").forEach((option) => option.remove());
        if (!Array.from(select.options).some((option) => option.value === rating)) {
            const option = document.createElement("option");
            option.value = rating;
            option.textContent = `${rating} (custom)`;
            option.dataset.customRating = "true";
            select.appendChild(option);
        }
        select.value = rating;
    }

    function setStatus(message) {
        document.querySelectorAll("[data-perchance-status]").forEach((target) => {
            target.textContent = message;
        });
    }

    function setOutputDirectory(path) {
        document.querySelectorAll("[data-perchance-output-dir]").forEach((target) => {
            target.textContent = `Output: ${path}`;
        });
    }

    function setBusy(nextBusy, message) {
        busy = nextBusy;
        element("image-button").disabled = nextBusy;
        element("text-button").disabled = nextBusy;
        element("gallery-list-button").disabled = nextBusy;
        element("gallery-get-button").disabled = nextBusy;
        element("gallery-save-button").disabled = nextBusy;
        setStatus(message || (nextBusy ? "Perchance is busy…" : "Ready"));
    }

    function showTextError(message) {
        element("text-result").style.display = "block";
        const textError = element("text-error");
        textError.textContent = message;
        textError.style.display = "block";
        element("text-output-label").style.display = "none";
        element("text-output-container").style.display = "none";
        element("text-output-actions").style.display = "none";
    }

    function showTextOutput() {
        element("text-result").style.display = "block";
        const textError = element("text-error");
        textError.textContent = "";
        textError.style.display = "none";
        element("text-output-label").style.display = "";
        element("text-output-container").style.display = "";
        element("text-output-actions").style.display = "flex";
    }

    async function requestJson(url, body) {
        const response = await fetch(url, {
            method: body === undefined ? "GET" : "POST",
            headers: body === undefined ? {} : { "Content-Type": "application/json" },
            body: body === undefined ? undefined : JSON.stringify(body),
        });
        const raw = await response.text();
        let data;
        try {
            data = raw ? JSON.parse(raw) : {};
        } catch (_) {
            data = { detail: raw };
        }
        if (!response.ok) {
            throw new Error(data.detail || `${response.status} ${response.statusText}`);
        }
        return data;
    }

    function promptValue(kind) {
        const prompt = element(`${kind}-prompt`).value.trim();
        if (!prompt) {
            throw new Error("Enter a prompt first.");
        }
        return prompt;
    }

    function importGeneratorPrompt(kind) {
        const easyPrompt = document.querySelector("#prompt");
        if (!easyPrompt) {
            setStatus("Easy Diffusion prompt field is unavailable.");
            return;
        }

        const prompt = element(`${kind}-prompt`);
        prompt.value = easyPrompt.value;
        autoResizeTextarea(prompt);

        if (kind === "image") {
            const easyNegativePrompt = document.querySelector("#negative_prompt");
            if (easyNegativePrompt) {
                const negativePrompt = element("negative-prompt");
                negativePrompt.value = easyNegativePrompt.value;
                autoResizeTextarea(negativePrompt);
            }
        }

        saveSettings();
        setStatus(kind === "image"
            ? "Imported positive and negative prompts from Easy Diffusion."
            : "Imported prompt from Easy Diffusion.");
        prompt.focus();
    }

    function exportGeneratorPrompt(kind) {
        const easyPrompt = document.querySelector("#prompt");
        if (!easyPrompt) {
            setStatus("Easy Diffusion prompt field is unavailable.");
            return;
        }

        easyPrompt.value = element(`${kind}-prompt`).value;
        dispatchInput(easyPrompt);

        if (kind === "image") {
            const easyNegativePrompt = document.querySelector("#negative_prompt");
            if (easyNegativePrompt) {
                easyNegativePrompt.value = element("negative-prompt").value;
                dispatchInput(easyNegativePrompt);
            }
        }

        setStatus(kind === "image"
            ? "Exported positive and negative prompts to Easy Diffusion."
            : "Exported prompt to Easy Diffusion.");
    }

    function applyTextFilters(text, filterStrings) {
        let filteredText = text;
        let skippedCount = 0;

        filterStrings.forEach((filterString) => {
            const nextText = filteredText.split(filterString).join("");
            if (filteredText.trim() && !nextText.trim()) {
                skippedCount += 1;
                return;
            }
            filteredText = nextText;
        });

        return { filteredText, skippedCount };
    }

    function migratePresetStarter(text) {
        for (const preset of TEXT_PROMPT_PRESETS) {
            const previousStarter = preset.starter.slice(0, -PRESET_ENDING.length);
            const legacyStarters = [
                previousStarter + LEGACY_PRESET_SUFFIX,
                previousStarter,
            ];
            const matchingStarter = legacyStarters.find(
                (starter) => text === starter || text.startsWith(`${starter}\n\n`)
            );
            if (matchingStarter) {
                return preset.starter + text.slice(matchingStarter.length);
            }
        }
        return text;
    }

    function applyTextPromptPreset() {
        const selectedPresetId = element("text-preset").value;
        const selectedPreset = TEXT_PROMPT_PRESETS.find(
            (preset) => preset.id === selectedPresetId
        );
        if (!selectedPreset) {
            setStatus("Choose a prompt starter preset first.");
            return;
        }

        const startWith = element("start-with");
        let starterBody = migratePresetStarter(startWith.value).trim();
        const existingPreset = TEXT_PROMPT_PRESETS.find((preset) =>
            starterBody.startsWith(preset.starter)
        );
        if (existingPreset) {
            starterBody = starterBody.slice(existingPreset.starter.length).trim();
        }

        startWith.value =
            selectedPreset.starter + (starterBody ? `\n\n${starterBody}` : "");
        const textOptions = startWith.closest("details");
        if (textOptions) {
            textOptions.open = true;
        }
        requestAnimationFrame(() => autoResizeTextarea(startWith));
        saveSettings();
        setStatus(`Applied preset to Start with: ${selectedPreset.label}`);
        startWith.focus();
    }

    function clearStartWithOnPresetChange() {
        const startWith = element("start-with");
        startWith.value = "";
        autoResizeTextarea(startWith);
        saveSettings();
        setStatus(
            element("text-preset").value
                ? "Start with cleared. Click Insert into Start With to apply the preset."
                : "Start with cleared."
        );
    }

    function createPanels() {
        const settings = loadSettings();
        if (typeof settings.startWith === "string") {
            const migratedStartWith = migratePresetStarter(settings.startWith);
            if (migratedStartWith !== settings.startWith) {
                settings.startWith = migratedStartWith;
                localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
            }
        }
        const textPresetOptions = TEXT_PROMPT_PRESETS.map(
            (preset) => `<option value="${preset.id}">${preset.label}</option>`
        ).join("");
        const imagePanel = document.createElement("div");
        imagePanel.id = `${ID_PREFIX}-image-panel`;
        imagePanel.className = "panel-box";
        imagePanel.style.flex = "0 0 auto";
        imagePanel.innerHTML = `
            <h4 class="collapsible">Perchance Image</h4>
            <div class="collapsible-content" style="display:block;margin-top:15px;">
                <div style="display:flex;justify-content:space-between;gap:10px;margin-bottom:8px;">
                    <span data-perchance-status style="font-size:12px;opacity:.8;">Checking launcher…</span>
                    <span data-perchance-output-dir style="font-size:11px;opacity:.65;overflow-wrap:anywhere;"></span>
                </div>

                <label for="${ID_PREFIX}-image-prompt" style="font-size:12px;">Positive prompt</label>
                <div>
                    <textarea id="${ID_PREFIX}-image-prompt" rows="4" style="display:block;width:100%;" placeholder="Describe the image to generate"></textarea>
                </div>

                <label style="display:block;font-size:12px;margin-top:8px;">Negative prompt
                    <textarea id="${ID_PREFIX}-negative-prompt" rows="2" style="display:block;width:100%;"></textarea>
                </label>

                <div id="${ID_PREFIX}-image-result" style="display:none;margin-top:12px;">
                    <img id="${ID_PREFIX}-image" alt="Generated by Perchance" style="display:block;max-width:100%;max-height:640px;border-radius:6px;">
                    <a id="${ID_PREFIX}-image-link" target="_blank" rel="noopener" style="display:block;margin-top:6px;overflow-wrap:anywhere;"></a>
                </div>

                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:8px;margin-top:8px;">
                    <label style="font-size:12px;">Shape
                        <select id="${ID_PREFIX}-shape" style="display:block;width:100%;">
                            <option value="square">square</option>
                            <option value="portrait">portrait</option>
                            <option value="landscape">landscape</option>
                        </select>
                    </label>
                    <label style="font-size:12px;">Seed
                        <input id="${ID_PREFIX}-seed" type="number" step="1" value="-1" style="display:block;width:100%;">
                    </label>
                    <label style="font-size:12px;">Guidance scale
                        <input id="${ID_PREFIX}-guidance-scale" type="number" step="0.1" value="7" style="display:block;width:100%;">
                    </label>
                </div>

                <div style="display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:8px;margin-top:8px;">
                    <button id="${ID_PREFIX}-import-image-prompt" class="secondaryButton" style="grid-column:1;justify-self:start;">Import Prompt</button>
                    <button id="${ID_PREFIX}-image-button" class="primaryButton" style="grid-column:2;justify-self:center;">Generate Image</button>
                    <button id="${ID_PREFIX}-export-image-prompt" class="secondaryButton" style="grid-column:3;justify-self:end;">Export Prompt</button>
                </div>
            </div>
        `;

        const textPanel = document.createElement("div");
        textPanel.id = `${ID_PREFIX}-text-panel`;
        textPanel.className = "panel-box";
        textPanel.style.flex = "0 0 auto";
        textPanel.innerHTML = `
            <h4 class="collapsible">Perchance Text</h4>
            <div class="collapsible-content" style="display:block;margin-top:15px;">
                <div style="display:flex;justify-content:space-between;gap:10px;margin-bottom:8px;">
                    <span data-perchance-status style="font-size:12px;opacity:.8;">Checking launcher…</span>
                    <span data-perchance-output-dir style="font-size:11px;opacity:.65;overflow-wrap:anywhere;"></span>
                </div>

                <label for="${ID_PREFIX}-text-prompt" style="font-size:12px;">Text prompt</label>
                <div>
                    <textarea id="${ID_PREFIX}-text-prompt" rows="4" style="display:block;width:100%;" placeholder="Describe the text to generate"></textarea>
                </div>
                <div style="margin-top:8px;text-align:left;">
                    <label for="${ID_PREFIX}-text-preset" style="font-size:12px;">Prompt starter preset</label>
                    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                        <select id="${ID_PREFIX}-text-preset" style="display:block;flex:1 1 220px;min-width:0;width:100%;">
                            <option value="">Choose a preset…</option>
                            ${textPresetOptions}
                        </select>
                        <button id="${ID_PREFIX}-apply-text-preset" class="secondaryButton">Insert into Start With</button>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:8px;margin-top:8px;">
                    <button id="${ID_PREFIX}-import-text-prompt" class="secondaryButton" style="grid-column:1;justify-self:start;">Import Prompt</button>
                    <button id="${ID_PREFIX}-text-button" class="primaryButton" style="grid-column:2;justify-self:center;">Generate Text</button>
                    <button id="${ID_PREFIX}-export-text-prompt" class="secondaryButton" style="grid-column:3;justify-self:end;">Export Prompt</button>
                </div>

                <details style="margin-top:12px;">
                    <summary style="cursor:pointer;font-size:12px;">Text options</summary>
                    <label style="display:block;font-size:12px;margin-top:8px;">Start with
                        <textarea id="${ID_PREFIX}-start-with" rows="2" style="display:block;width:100%;"></textarea>
                    </label>
                    <label style="display:block;font-size:12px;margin-top:8px;">Stop sequences (one per line)
                        <textarea id="${ID_PREFIX}-stops" rows="2" style="display:block;width:100%;"></textarea>
                    </label>
                    <label style="display:block;font-size:12px;margin-top:8px;">Filter out strings (one per line)
                        <textarea id="${ID_PREFIX}-filters" rows="2" style="display:block;width:100%;"></textarea>
                    </label>
                    <label style="display:block;font-size:12px;margin-top:8px;">Per-chunk timeout (milliseconds)
                        <input id="${ID_PREFIX}-timeout" type="number" min="1" step="1000" placeholder="Perchance default" style="display:block;width:100%;">
                    </label>
                </details>

                <div id="${ID_PREFIX}-text-result" style="display:none;margin-top:12px;">
                    <div id="${ID_PREFIX}-text-error" role="alert" style="display:none;padding:8px;border:1px solid currentColor;border-radius:4px;color:#d9534f;background:rgba(217,83,79,.1);white-space:pre-wrap;overflow-wrap:anywhere;text-align:left;"></div>
                    <label id="${ID_PREFIX}-text-output-label" for="${ID_PREFIX}-text-output" style="font-size:12px;">Generated text</label>
                    <div id="${ID_PREFIX}-text-output-container">
                        <textarea id="${ID_PREFIX}-text-output" rows="7" style="display:block;width:100%;" readonly></textarea>
                    </div>
                    <div id="${ID_PREFIX}-text-output-actions" style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;">
                        <button id="${ID_PREFIX}-apply-text" class="secondaryButton">Use as Easy Diffusion Prompt</button>
                        <button id="${ID_PREFIX}-apply-negative-text" class="secondaryButton">Use as Easy Diffusion Negative Prompt</button>
                        <button id="${ID_PREFIX}-copy-text" class="secondaryButton">Copy Text</button>
                    </div>
                </div>
            </div>
        `;

        const galleryPanel = document.createElement("div");
        galleryPanel.id = `${ID_PREFIX}-gallery-panel`;
        galleryPanel.className = "panel-box";
        galleryPanel.style.flex = "0 0 auto";
        galleryPanel.innerHTML = `
            <h4 class="collapsible">Perchance Gallery</h4>
            <div class="collapsible-content" style="display:block;margin-top:15px;">
                <div style="display:flex;justify-content:space-between;gap:10px;margin-bottom:8px;">
                    <span data-perchance-status style="font-size:12px;opacity:.8;">Checking launcher…</span>
                    <span data-perchance-output-dir style="font-size:11px;opacity:.65;overflow-wrap:anywhere;"></span>
                </div>
                <label style="display:block;font-size:12px;">Gallery image ID or supported URL
                    <input id="${ID_PREFIX}-gallery-id" type="text" spellcheck="false" style="display:block;width:100%;" placeholder="64-character image ID or Perchance gallery/image URL">
                </label>
                <label style="display:block;font-size:12px;margin-top:8px;">Generator channel / URL ID
                    <input id="${ID_PREFIX}-gallery-channel" type="text" spellcheck="false" style="display:block;width:100%;" value="ai-text-to-image-generator">
                </label>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-top:8px;">
                    <label style="font-size:12px;">Rating
                        <select id="${ID_PREFIX}-gallery-content-filter" style="display:block;width:100%;">
                            <option value="g">G</option>
                            <option value="pg13">PG-13</option>
                            <option value="none">None / unfiltered</option>
                        </select>
                    </label>
                    <label style="font-size:12px;">Limit
                        <input id="${ID_PREFIX}-gallery-limit" type="number" min="1" max="100" step="1" value="20" style="display:block;width:100%;">
                    </label>
                    <label style="font-size:12px;">Sort
                        <select id="${ID_PREFIX}-gallery-sort" style="display:block;width:100%;">
                            <option value="recent">recent</option>
                            <option value="top">top</option>
                            <option value="trending">trending</option>
                        </select>
                    </label>
                    <label style="font-size:12px;">Time range
                        <input id="${ID_PREFIX}-gallery-time-range" type="text" placeholder="CLI default" style="display:block;width:100%;">
                    </label>
                </div>
                <label style="display:block;font-size:12px;margin-top:8px;">Continuation cursor
                    <input id="${ID_PREFIX}-gallery-cursor" type="text" spellcheck="false" style="display:block;width:100%;" placeholder="Filled after listing when another page exists">
                </label>
                <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:12px;">
                    <label><input id="${ID_PREFIX}-gallery-download" type="checkbox"> Save images into Easy Diffusion outputs</label>
                    <label><input id="${ID_PREFIX}-gallery-visible" type="checkbox"> Show browser</label>
                </div>
                <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:10px;">
                    <button id="${ID_PREFIX}-gallery-save-button" class="secondaryButton">Save ID + Channel</button>
                    <button id="${ID_PREFIX}-gallery-list-button" class="primaryButton">List Gallery</button>
                    <button id="${ID_PREFIX}-gallery-get-button" class="primaryButton">Get Image</button>
                </div>
                <div id="${ID_PREFIX}-gallery-results" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:12px;"></div>
            </div>
        `;

        imagePanel.querySelector(`#${ID_PREFIX}-image-prompt`).value = settings.imagePrompt || "";
        imagePanel.querySelector(`#${ID_PREFIX}-shape`).value = settings.shape || "square";
        imagePanel.querySelector(`#${ID_PREFIX}-negative-prompt`).value = settings.negativePrompt || "";
        imagePanel.querySelector(`#${ID_PREFIX}-seed`).value = settings.seed || "-1";
        imagePanel.querySelector(`#${ID_PREFIX}-guidance-scale`).value = settings.guidanceScale || "7";
        textPanel.querySelector(`#${ID_PREFIX}-text-prompt`).value = settings.textPrompt || "";
        textPanel.querySelector(`#${ID_PREFIX}-text-preset`).value = settings.textPreset || "";
        textPanel.querySelector(`#${ID_PREFIX}-start-with`).value = settings.startWith || "";
        textPanel.querySelector(`#${ID_PREFIX}-stops`).value = settings.stops || "";
        textPanel.querySelector(`#${ID_PREFIX}-filters`).value = settings.filterStrings || "";
        textPanel.querySelector(`#${ID_PREFIX}-timeout`).value = settings.timeout || "";
        galleryPanel.querySelector(`#${ID_PREFIX}-gallery-id`).value = settings.galleryId || "";
        galleryPanel.querySelector(`#${ID_PREFIX}-gallery-channel`).value = settings.galleryChannel || "ai-text-to-image-generator";
        setGalleryRating(settings.galleryContentFilter || "g");
        galleryPanel.querySelector(`#${ID_PREFIX}-gallery-limit`).value = settings.galleryLimit || "20";
        galleryPanel.querySelector(`#${ID_PREFIX}-gallery-cursor`).value = settings.galleryCursor || "";
        galleryPanel.querySelector(`#${ID_PREFIX}-gallery-sort`).value = settings.gallerySort || "recent";
        galleryPanel.querySelector(`#${ID_PREFIX}-gallery-time-range`).value = settings.galleryTimeRange || "";
        galleryPanel.querySelector(`#${ID_PREFIX}-gallery-download`).checked = Boolean(settings.galleryDownload);
        galleryPanel.querySelector(`#${ID_PREFIX}-gallery-visible`).checked = Boolean(settings.galleryVisible);
        return { imagePanel, textPanel, galleryPanel };
    }

    async function generateImage() {
        if (busy) return;
        try {
            const payload = {
                prompt: promptValue("image"),
                shape: element("shape").value,
                negative_prompt: element("negative-prompt").value,
                seed: element("seed").value,
                guidance_scale: element("guidance-scale").value,
            };
            saveSettings();
            setBusy(true, "Generating image with Perchance…");
            const data = await requestJson("/perchance/image", payload);
            const image = element("image");
            image.src = `${data.url}?t=${Date.now()}`;
            const link = element("image-link");
            link.href = data.url;
            link.textContent = data.path;
            element("image-result").style.display = "block";
            setBusy(false, "Image generated and saved to Easy Diffusion outputs.");
        } catch (error) {
            setBusy(false, `Image generation failed: ${error.message}`);
            console.warn("[Perchance Plugin] Image generation failed", error);
        }
    }

    async function generateText() {
        if (busy) return;
        try {
            const timeout = element("timeout").value.trim();
            const textPrompt = promptValue("text");
            const startWithValue = element("start-with").value;
            const startWithPreset = TEXT_PROMPT_PRESETS.find((preset) =>
                startWithValue.trim().startsWith(preset.starter)
            );
            const filterStrings = element("filters").value
                .split(/\r?\n/)
                .map((value) => value.trim())
                .filter((value) => value.length);
            const payload = {
                prompt: startWithPreset
                    ? `${startWithValue.trim()}\n\n${textPrompt}`
                    : textPrompt,
                start_with: startWithPreset ? "" : startWithValue,
                stop: element("stops").value.split(/\r?\n/).filter((value) => value.length),
            };
            if (timeout) payload.timeout_ms = timeout;
            saveSettings();
            setBusy(true, "Generating text with Perchance…");
            const data = await requestJson("/perchance/text", payload);
            const generatedText = data.text || "";
            if (!generatedText.trim()) {
                throw new Error("Perchance returned an empty text response.");
            }
            const textOutput = element("text-output");
            showTextOutput();
            const headingFilteredCandidate = REQUIRED_TEXT_FILTERS.reduce(
                (text, filterString) => text.split(filterString).join(""),
                generatedText
            );
            const headingFilterSkipped =
                Boolean(generatedText.trim()) && !headingFilteredCandidate.trim();
            const headingFilteredText = headingFilterSkipped
                ? generatedText
                : headingFilteredCandidate;
            const { filteredText, skippedCount } = applyTextFilters(
                headingFilteredText,
                filterStrings
            );
            const cleanedCandidate = filteredText.replace(
                /^(?:[.():][^\S\r\n]*)*(?:[^\S\r\n]*(?:\r\n|\r|\n)){2}/,
                ""
            );
            const leadingCleanupSkipped =
                Boolean(filteredText.trim()) && !cleanedCandidate.trim();
            const cleanedText = leadingCleanupSkipped
                ? filteredText
                : cleanedCandidate;
            const totalSkippedCount =
                skippedCount
                + Number(headingFilterSkipped)
                + Number(leadingCleanupSkipped);
            textOutput.value = cleanedText.replace(/\./g, ",");
            requestAnimationFrame(() => autoResizeTextarea(textOutput));
            setBusy(
                false,
                totalSkippedCount
                    ? `Text generated. Skipped ${totalSkippedCount} filter/cleanup rule(s) that would empty the response.`
                    : "Text generated."
            );
        } catch (error) {
            showTextError(`Text generation failed: ${error.message}`);
            setBusy(false, `Text generation failed: ${error.message}`);
            console.warn("[Perchance Plugin] Text generation failed", error);
        }
    }

    function galleryPayload() {
        const payload = {
            gallery_id: element("gallery-id").value.trim(),
            channel: element("gallery-channel").value.trim(),
            content_filter: element("gallery-content-filter").value.trim(),
            limit: element("gallery-limit").value,
            cursor: element("gallery-cursor").value.trim(),
            sort: element("gallery-sort").value,
            time_range: element("gallery-time-range").value.trim(),
            download: element("gallery-download").checked,
            visible: element("gallery-visible").checked,
        };
        saveSettings();
        return payload;
    }

    function useGalleryPrompt(prompt) {
        const easyPrompt = document.querySelector("#prompt");
        if (!easyPrompt) return;
        easyPrompt.value = prompt || "";
        dispatchInput(easyPrompt);
        setStatus("Gallery prompt copied to Easy Diffusion.");
    }

    function renderGalleryEntries(entries) {
        const results = element("gallery-results");
        results.replaceChildren();
        entries.forEach((entry) => {
            const card = document.createElement("div");
            card.style.cssText = "padding:8px;border:1px solid var(--border-color,rgba(127,127,127,.35));border-radius:6px;min-width:0;";
            const imageUrl = entry.local_url || entry.imageUrl || "";
            if (imageUrl) {
                const image = document.createElement("img");
                image.src = imageUrl;
                image.alt = "Perchance gallery image";
                image.loading = "lazy";
                image.style.cssText = "display:block;width:100%;height:180px;object-fit:contain;border-radius:4px;background:rgba(0,0,0,.12);";
                card.appendChild(image);
            }
            const prompt = document.createElement("div");
            prompt.textContent = entry.prompt || "(No prompt returned)";
            prompt.style.cssText = "font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;margin-top:7px;max-height:120px;overflow:auto;";
            card.appendChild(prompt);
            const id = document.createElement("div");
            id.textContent = entry.imageId || "";
            id.style.cssText = "font-size:10px;opacity:.65;overflow-wrap:anywhere;margin-top:5px;";
            card.appendChild(id);
            const actions = document.createElement("div");
            actions.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;";
            const promptButton = document.createElement("button");
            promptButton.className = "secondaryButton";
            promptButton.textContent = "Use Prompt";
            promptButton.addEventListener("click", () => useGalleryPrompt(entry.prompt));
            actions.appendChild(promptButton);
            if (entry.imageId) {
                const idButton = document.createElement("button");
                idButton.className = "secondaryButton";
                idButton.textContent = "Select ID";
                idButton.addEventListener("click", () => {
                    element("gallery-id").value = entry.imageId;
                    saveSettings();
                    setStatus("Gallery image ID selected. Use Save ID + Channel to persist it in Easy Diffusion.");
                });
                actions.appendChild(idButton);
            }
            card.appendChild(actions);
            results.appendChild(card);
        });
        if (!entries.length) {
            const empty = document.createElement("div");
            empty.textContent = "No gallery entries returned.";
            empty.style.opacity = ".7";
            results.appendChild(empty);
        }
    }

    async function saveGallerySettings() {
        try {
            setBusy(true, "Saving Perchance gallery settings…");
            const data = await requestJson("/perchance/settings", {
                gallery_id: element("gallery-id").value.trim(),
                channel: element("gallery-channel").value.trim(),
            });
            element("gallery-id").value = data.gallery_id || "";
            element("gallery-channel").value = data.channel || "ai-text-to-image-generator";
            saveSettings();
            setBusy(false, "Perchance gallery ID and channel saved.");
        } catch (error) {
            setBusy(false, `Saving gallery settings failed: ${error.message}`);
        }
    }

    async function listGallery() {
        if (busy) return;
        try {
            setBusy(true, "Loading Perchance gallery…");
            const data = await requestJson("/perchance/gallery/list", galleryPayload());
            renderGalleryEntries(Array.isArray(data.entries) ? data.entries : []);
            element("gallery-cursor").value = data.nextCursor || "";
            saveSettings();
            setBusy(false, data.nextCursor ? "Gallery loaded; continuation cursor saved." : "Gallery loaded.");
        } catch (error) {
            setBusy(false, `Gallery list failed: ${error.message}`);
        }
    }

    async function getGalleryImage() {
        if (busy) return;
        try {
            const payload = galleryPayload();
            if (!payload.gallery_id) throw new Error("Enter a gallery image ID or supported URL.");
            setBusy(true, "Loading Perchance gallery image…");
            const data = await requestJson("/perchance/gallery/get", payload);
            renderGalleryEntries([data]);
            setBusy(false, "Gallery image loaded.");
        } catch (error) {
            setBusy(false, `Gallery lookup failed: ${error.message}`);
        }
    }

    function bindPanel() {
        bindTextareaAutoResize();

        [
            "image-prompt",
            "shape",
            "negative-prompt",
            "seed",
            "guidance-scale",
            "text-prompt",
            "start-with",
            "stops",
            "filters",
            "timeout",
            "gallery-id",
            "gallery-channel",
            "gallery-content-filter",
            "gallery-limit",
            "gallery-cursor",
            "gallery-sort",
            "gallery-time-range",
            "gallery-download",
            "gallery-visible",
        ].forEach((suffix) => {
            element(suffix).addEventListener("input", saveSettings);
            element(suffix).addEventListener("change", saveSettings);
        });

        element("image-button").addEventListener("click", generateImage);
        element("text-button").addEventListener("click", generateText);
        element("gallery-save-button").addEventListener("click", saveGallerySettings);
        element("gallery-list-button").addEventListener("click", listGallery);
        element("gallery-get-button").addEventListener("click", getGalleryImage);
        ["image", "text"].forEach((kind) => {
            element(`import-${kind}-prompt`).addEventListener(
                "click",
                () => importGeneratorPrompt(kind)
            );
            element(`export-${kind}-prompt`).addEventListener(
                "click",
                () => exportGeneratorPrompt(kind)
            );
        });
        element("text-preset").addEventListener("change", clearStartWithOnPresetChange);
        element("apply-text-preset").addEventListener("click", applyTextPromptPreset);
        element("apply-text").addEventListener("click", () => {
            const easyPrompt = document.querySelector("#prompt");
            if (!easyPrompt) return;
            easyPrompt.value = element("text-output").value;
            dispatchInput(easyPrompt);
        });
        element("apply-negative-text").addEventListener("click", () => {
            const easyNegativePrompt = document.querySelector("#negative_prompt");
            if (!easyNegativePrompt) return;
            easyNegativePrompt.value = element("text-output").value;
            dispatchInput(easyNegativePrompt);
        });
        element("copy-text").addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(element("text-output").value);
                setStatus("Generated text copied.");
            } catch (error) {
                setStatus(`Copy failed: ${error.message}`);
            }
        });

        requestJson("/perchance/status")
            .then((data) => {
                setOutputDirectory(data.output_directory);
                if (data.settings) {
                    element("gallery-id").value = data.settings.gallery_id || element("gallery-id").value;
                    element("gallery-channel").value = data.settings.channel || element("gallery-channel").value;
                    saveSettings();
                }
                setBusy(Boolean(data.busy), data.busy ? "Perchance is already busy." : (
                    data.ready ? "Ready" : "Perchance launcher is unavailable."
                ));
                if (!data.ready) {
                    element("image-button").disabled = true;
                    element("text-button").disabled = true;
                    element("gallery-list-button").disabled = true;
                    element("gallery-get-button").disabled = true;
                }
            })
            .catch((error) => {
                setBusy(false, `Plugin server unavailable: ${error.message}`);
                element("image-button").disabled = true;
                element("text-button").disabled = true;
                element("gallery-list-button").disabled = true;
                element("gallery-get-button").disabled = true;
            });
    }

    function attachPanel() {
        if (
            document.querySelector(`#${ID_PREFIX}-image-panel`) ||
            document.querySelector(`#${ID_PREFIX}-text-panel`) ||
            document.querySelector(`#${ID_PREFIX}-gallery-panel`)
        ) return true;
        const editorInputs = document.querySelector("#editor-inputs");
        if (!editorInputs) return false;
        const { imagePanel, textPanel, galleryPanel } = createPanels();
        const separator =
            document.querySelector("#editor > span.line-separator") ||
            document.querySelector("span.line-separator");
        if (separator && separator.parentNode) {
            separator.insertAdjacentElement("afterend", imagePanel);
            imagePanel.insertAdjacentElement("afterend", textPanel);
            textPanel.insertAdjacentElement("afterend", galleryPanel);
        } else {
            editorInputs.insertAdjacentElement("afterend", imagePanel);
            imagePanel.insertAdjacentElement("afterend", textPanel);
            textPanel.insertAdjacentElement("afterend", galleryPanel);
        }
        if (typeof createCollapsibles === "function") {
            createCollapsibles(imagePanel);
            createCollapsibles(textPanel);
            createCollapsibles(galleryPanel);
        }
        bindPanel();
        return true;
    }

    if (!attachPanel()) {
        let attempts = 0;
        const timer = window.setInterval(() => {
            attempts += 1;
            if (attachPanel() || attempts >= 60) {
                window.clearInterval(timer);
            }
        }, 500);
    }
})();
