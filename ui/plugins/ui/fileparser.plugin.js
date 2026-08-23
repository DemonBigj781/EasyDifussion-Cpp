/**
 * LoRa Metadata Reader
 * v.1.4, last updated: 07/29/2026
 * By The Stig
 *
 * Free to use with the CMDR2 Stable Diffusion UI.
 */

(function() { "use strict";

    const DEBUG = false; // set true to enable console logs
    const log = (...args) => DEBUG && console.log(...args);

    const VERSION = "1.5";
    const ID_PREFIX = "TheStig-LoRa-Metadata-Reader";
    log('%s Embed Metadata Version: %s', ID_PREFIX, VERSION);

    const ENDPOINT_LIST = "/files/list_lora";
    const ENDPOINT_TRIGGERS = "/meta/get_triggers";
    const ENDPOINT_SCAN = "/meta/scan_loras";
    const AUTO_IMPORT_STORAGE_KEY = `${ID_PREFIX}-auto-import`;

    const defaultTrigger = true;
    const defaultClickToKeywords = true;

    let modifierImprovementPlugin = false;

    watchForPlugin().then((loaded) => {
        modifierImprovementPlugin = !!loaded;
        log("Modifier plugin loaded?", loaded);
    });

    injectLoaderCSS();
    initializeMetadataReader();
    injectLoraReloadButton();
    watchForLoraManager();

    /* =========================
       Helpers
    ========================= */

    function applyWordToPrompt(selector, word) {
        const field = document.querySelector(selector);
        if (!field) return;

        let words = field.value
            .split(',')
            .map(w => w.trim())
            .filter(Boolean);

        const idx = words.indexOf(word);
        if (idx > -1) {
            words.splice(idx, 1);
        } else {
            words.push(word); // always append
        }

        field.value = words.join(', ');

        ["input", "change"].forEach(evt =>
            field.dispatchEvent(new Event(evt, { bubbles: true }))
        );
    }

    function toggleWordInField(field, word, appendEnd) {
        if (!field) return null;

        const w = String(word || "").trim();
        if (!w) return null;

        let words = field.value
            .split(',')
            .map(s => s.trim())
            .filter(Boolean);

        const idx = words.indexOf(w);
        const removed = idx > -1;

        if (removed) {
            words.splice(idx, 1);
        } else {
            appendEnd ? words.push(w) : words.unshift(w);
        }

        field.value = words.join(', ');

        ["input", "change"].forEach(evt =>
            field.dispatchEvent(new Event(evt, { bubbles: true }))
        );

        // return whether the word exists AFTER the operation
        return !removed;
    }

    function fieldHasWord(field, word) {
        if (!field) return false;
        const w = String(word || "").trim();
        if (!w) return false;

        return field.value
            .split(',')
            .map(s => s.trim())
            .filter(Boolean)
            .includes(w);
    }

    function getPositivePromptField() {
        return (
            document.querySelector("#prompt") ||
            document.querySelector("textarea#prompt") ||
            document.querySelector("textarea[name='prompt']") ||
            document.querySelector("textarea[data-testid='prompt']") ||
            document.querySelector("textarea")
        );
    }

    function getNegativePromptField() {
        return (
            document.querySelector("#negative_prompt") ||
            document.querySelector("#negativePrompt") ||
            document.querySelector("textarea#negative_prompt") ||
            document.querySelector("textarea[name='negative_prompt']") ||
            document.querySelector("textarea[name='negativePrompt']") ||
            document.querySelector("textarea[data-testid='negative_prompt']") ||
            null
        );
    }



    async function postJSON(url, bodyObj) {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: bodyObj ? JSON.stringify(bodyObj) : "{}",
        });
        const data = await res.json().catch(() => null);
        if (!res.ok) {
            const detail = data?.detail || data?.error || res.statusText;
            throw new Error(detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status}`);
        }
        return data;
    }

    function normalizeListResponse(data) {
        if (Array.isArray(data)) return data;
        if (data && typeof data === "object") {
            for (const k of ["items","loras","files","results","data"]) {
                if (Array.isArray(data[k])) return data[k];
            }
        }
        return [];
    }

    function buildOptions(list) {
        return list
            .filter(v =>
                typeof v === "string" &&
                v.toLowerCase().endsWith(".safetensors")
            )
            .sort((a,b)=>a.localeCompare(b,undefined,{sensitivity:"base"}))
            .map(fp => ({
                label: fp.split("/").pop(),
                value: fp
            }));
    }

    function injectLoraReloadButton() {
        const container = document.querySelector("#lora_model");
        if (!container) return;
        const parent = container.parentElement;
        if (!parent) return;
        if (parent.querySelector("#lora_model_reload")) return;

        const btn = document.createElement("button");
        btn.id = "lora_model_reload";
        btn.className = "secondaryButton reloadModels";
        btn.type = "button";
        btn.title = "Reload LoRA list";
        btn.innerHTML = '<i class="fa-solid fa-rotate"></i>';
        btn.addEventListener("click", () => {
            document.dispatchEvent(new Event("refreshModels"));
        });

        if (container.nextSibling) {
            parent.insertBefore(btn, container.nextSibling);
        } else {
            parent.appendChild(btn);
        }
    }

    async function loadLoraList(selectEl, statusEl) {
        statusEl.textContent = "Loading LoRAs…";
        selectEl.disabled = true;
        try {
            const data = await postJSON(ENDPOINT_LIST);
            const opts = buildOptions(normalizeListResponse(data));

            selectEl.innerHTML = "";
            const ph = document.createElement("option");
            ph.value = "";
            ph.textContent = opts.length ? "Select a LoRA…" : "No LoRAs found";
            selectEl.appendChild(ph);

            opts.forEach(o=>{
                const el=document.createElement("option");
                el.value=o.value;
                el.textContent=o.label;
                selectEl.appendChild(el);
            });

            statusEl.textContent = `Loaded ${opts.length} LoRAs`;
        } catch (e) {
            statusEl.textContent = "Failed to load LoRAs";
            log("loadLoraList error:", e);
        }
        selectEl.disabled = false;
    }

    // Normalizes the built-in API's { meta: {...} } response for the renderer.
    async function fetchMetadata(filepath) {
        const data = await postJSON(ENDPOINT_TRIGGERS, {
            filepath,
            include_metadata: true
        });

        if (data?.__metadata__) return data;

        const record = data?.meta ?? data?.metadata ?? data;
        if (record && typeof record === "object") {
            const embedded = record.meta ?? record.embedded_metadata;
            return {
                __metadata__: {
                    ...(embedded && typeof embedded === "object"
                        ? embedded
                        : {}),
                    ss_trigger_words: record.trigger_words ?? record.triggers ?? null,
                    model_name: record.model_name ?? null
                }
            };
        }

        return { __metadata__: {} };
    }
    function selectLoraInMainTab(filepath) {
        if (!filepath) return;

        // Easy Diffusion uses paths like: "sdxl/absol_illust" (NO extension)
        // Your API gives full path ending in ".safetensors"
        const normalized = String(filepath)
            .replaceAll("\\", "/")
            .replace(/^.*?(?=(?:sdxl|flux|1\.5)\b)/i, "") // strip anything before known roots if present
            .replace(/\.safetensors$/i, "")
            .trim();

        const input = document.querySelector("#loraModel");
        const list = document.querySelector("#loraModel-model-list");
        const arrow = document.querySelector("#loraModel-model-filter-arrow");

        if (!input || !list) return;

        // Open dropdown if it's closed
        const openDropdown = () => {
            // Some builds toggle via arrow click, some via input focus/click
            input.focus();
            input.click();
            if (arrow) arrow.click();
            list.style.display = "block"; // harmless if ED manages it; helps if it doesn't
        };

        openDropdown();

        // Find the matching LI by data-path
        const tryClick = () => {
            const items = Array.from(list.querySelectorAll("li.model-file[data-path]"));

            // Exact match
            let li = items.find(x => (x.getAttribute("data-path") || "") === normalized);

            // Fallback: sometimes API returns only filename, so try endsWith match
            if (!li) {
                li = items.find(x => (x.getAttribute("data-path") || "").endsWith("/" + normalized.split("/").pop()));
            }

            // As a last fallback, match by visible text (model name)
            if (!li) {
                const wantName = normalized.split("/").pop();
                li = items.find(x => (x.textContent || "").trim() === wantName);
            }

            if (!li) return false;

            // Click the item — this is what actually selects in ED
            li.click();

            // Also sync the input fields (some builds read these)
            const dp = li.getAttribute("data-path") || normalized;
            input.value = (li.textContent || "").trim();
            input.setAttribute("data-path", dp);

            ["input", "change"].forEach(t => input.dispatchEvent(new Event(t, { bubbles: true })));

            // Close dropdown
            list.style.display = "none";
            return true;
        };

        // ED may rebuild the list asynchronously; try a few times quickly
        let attempts = 0;
        const timer = setInterval(() => {
            attempts++;
            if (tryClick() || attempts >= 10) clearInterval(timer);
        }, 50);
    }

    function getPromptField() {
        // Easy Diffusion has changed IDs across versions; try several safe candidates.
        return (
            document.querySelector("#prompt") ||
            document.querySelector("textarea#prompt") ||
            document.querySelector("textarea[name='prompt']") ||
            document.querySelector("textarea[placeholder*='prompt' i]") ||
            document.querySelector("textarea")
        );
    }

    function findMainLoraSelect() {
        // Try known selectors first
        const known = [
            "#lora-select",
            "#lora_model",
            "select[name='lora']",
            "select[data-testid='lora-select']"
        ];
        for (const sel of known) {
            const el = document.querySelector(sel);
            if (el && el.tagName === "SELECT") return el;
        }

        // Fallback: pick the first SELECT that contains LoRA-ish options
        const selects = Array.from(document.querySelectorAll("select"));
        for (const s of selects) {
            const opts = Array.from(s.options || []);
            if (!opts.length) continue;
            const hit = opts.some(o => {
                const t = `${o.value} ${o.text}`.toLowerCase();
                return t.includes("lora") || t.includes(".safetensors");
            });
            if (hit) return s;
        }
        return null;
    }

    function normalizeModelKey(value) {
        let model = String(value || "")
            .trim()
            .replaceAll("\\", "/");

        const loraMarker = model.toLowerCase().lastIndexOf("/lora/");
        if (loraMarker >= 0) {
            model = model.slice(loraMarker + "/lora/".length);
        }

        return model
            .replace(/^\/+/, "")
            .replace(/\.safetensors$/i, "");
    }

    function getSelectedLoraReference() {
        const input = document.querySelector("#loraModel");
        if (!input) return "";
        return (
            input.getAttribute("data-path") ||
            input.dataset?.path ||
            input.value ||
            ""
        );
    }

    function normalizeKeywords(keywords) {
        const result = [];
        const seen = new Set();

        (Array.isArray(keywords) ? keywords : []).forEach(value => {
            const keyword = String(value || "").trim();
            const normalized = keyword.toLocaleLowerCase();
            if (!keyword || seen.has(normalized)) return;
            seen.add(normalized);
            result.push(keyword);
        });

        return result;
    }

    function mergeKeywords(existing, imported) {
        return normalizeKeywords([
            ...(Array.isArray(existing) ? existing : []),
            ...(Array.isArray(imported) ? imported : [])
        ]);
    }

    async function storeLoraKeywords(model, keywords) {
        if (typeof Bucket === "undefined") {
            throw new Error("Easy Diffusion model storage is not ready");
        }

        const modelKey = normalizeModelKey(model);
        if (!modelKey) throw new Error("No LoRA model selected");

        const bucketPath = `modelinfo/lora/${modelKey}`;
        const existingInfo = await Bucket.retrieve(bucketPath);
        const info = existingInfo && typeof existingInfo === "object"
            ? { ...existingInfo }
            : {};

        info.keywords = mergeKeywords(info.keywords, keywords);
        if (typeof info.notes !== "string") info.notes = "";

        const response = await Bucket.store(bucketPath, info);
        if (response && "ok" in response && !response.ok) {
            throw new Error(`Failed to save LoRA keywords (HTTP ${response.status})`);
        }

        const selectedModel = normalizeModelKey(getSelectedLoraReference());
        const keywordsField = document.querySelector("#lora-manager-keywords");
        if (keywordsField && selectedModel === modelKey) {
            keywordsField.value = info.keywords.join("\n");
            ["input", "change"].forEach(eventType => {
                keywordsField.dispatchEvent(new Event(eventType, { bubbles: true }));
            });
        }

        document.dispatchEvent(new CustomEvent("loraMetadataImported", {
            detail: { model: modelKey, keywords: info.keywords }
        }));

        return info.keywords;
    }

    async function fetchLoraRecord(filepath) {
        const data = await postJSON(ENDPOINT_TRIGGERS, {
            filepath,
            include_metadata: false
        });
        const record = data?.meta ?? data?.metadata;
        if (!record || typeof record !== "object") {
            throw new Error("The metadata API returned no metadata");
        }
        if (record.error) throw new Error(record.error);
        return record;
    }

    function refreshOpenEmbeddingsDialog() {
        const dialog = document.querySelector("#embeddings-dialog");
        if (
            dialog?.open &&
            typeof updateEmbeddingsList === "function"
        ) {
            updateEmbeddingsList();
        }
    }

    async function importSelectedLora(statusEl) {
        const filepath = getSelectedLoraReference();
        if (!filepath) {
            statusEl.textContent = "Select a LoRA first.";
            return;
        }

        statusEl.textContent = "Retrieving embedded LoRA metadata…";
        const record = await fetchLoraRecord(filepath);
        const model = record.model || normalizeModelKey(filepath);
        const imported = normalizeKeywords(record.trigger_words);
        const stored = await storeLoraKeywords(model, imported);
        refreshOpenEmbeddingsDialog();

        statusEl.textContent = imported.length
            ? `Imported ${imported.length} metadata keywords for ${model}. ${stored.length} total available in + Embeddings.`
            : `No embedded trigger or training-tag metadata found for ${model}.`;
    }

    async function importAllLoras(statusEl, selectedButton, allButton) {
        selectedButton.disabled = true;
        allButton.disabled = true;
        statusEl.textContent = "Scanning all LoRA metadata…";

        try {
            const data = await postJSON(ENDPOINT_SCAN);
            const items = Array.isArray(data)
                ? data
                : (Array.isArray(data?.meta)
                    ? data.meta
                    : (Array.isArray(data?.items) ? data.items : []));

            let importedModels = 0;
            let importedKeywords = 0;
            let skippedModels = 0;

            for (let index = 0; index < items.length; index++) {
                const record = items[index];
                const model = record?.model;
                const keywords = normalizeKeywords(record?.trigger_words);

                statusEl.textContent =
                    `Adding LoRA metadata to + Embeddings… ${index + 1}/${items.length}`;

                if (!model || record?.error || keywords.length === 0) {
                    skippedModels++;
                    continue;
                }

                await storeLoraKeywords(model, keywords);
                importedModels++;
                importedKeywords += keywords.length;
            }

            refreshOpenEmbeddingsDialog();
            statusEl.textContent =
                `Imported ${importedKeywords} metadata keywords from ${importedModels} LoRAs` +
                (skippedModels ? `; ${skippedModels} had no usable metadata.` : ".");
        } finally {
            selectedButton.disabled = false;
            allButton.disabled = false;
        }
    }

    function addLoraManagerMetadataControls() {
        const tab = document.querySelector("#tab-content-model-loraUI");
        if (!tab || tab.querySelector("#lora-metadata-auto-import")) return false;

        const selectorArea =
            tab.querySelector(".lora-manager-grid-selector") ||
            tab.querySelector(".tab-content-inner");
        if (!selectorArea) return false;

        const controls = document.createElement("div");
        controls.id = "lora-metadata-auto-import";
        controls.innerHTML = `
            <button id="lora-metadata-import-selected" class="tertiaryButton smallButton" type="button">
                Import selected metadata
            </button>
            <button id="lora-metadata-import-all" class="tertiaryButton smallButton" type="button">
                Import all LoRAs
            </button>
            <label>
                <input id="lora-metadata-auto-toggle" type="checkbox">
                Auto-import metadata when a LoRA is selected
            </label>
            <small id="lora-metadata-import-status">
                Imported keywords appear in <b>+ Embeddings</b> while their LoRA is selected.
            </small>
        `;
        selectorArea.appendChild(controls);

        const selectedButton = controls.querySelector("#lora-metadata-import-selected");
        const allButton = controls.querySelector("#lora-metadata-import-all");
        const autoToggle = controls.querySelector("#lora-metadata-auto-toggle");
        const statusEl = controls.querySelector("#lora-metadata-import-status");
        const modelInput = tab.querySelector("#loraModel");

        let savedAutoImport = null;
        try {
            savedAutoImport = localStorage.getItem(AUTO_IMPORT_STORAGE_KEY);
        } catch (e) {
            log("Unable to read auto-import setting:", e);
        }
        autoToggle.checked = savedAutoImport !== "false";

        const runSelectedImport = () => {
            selectedButton.disabled = true;
            importSelectedLora(statusEl)
                .catch(error => {
                    statusEl.textContent = `Metadata import failed: ${error.message}`;
                })
                .finally(() => {
                    selectedButton.disabled = false;
                });
        };

        selectedButton.addEventListener("click", runSelectedImport);
        allButton.addEventListener("click", () => {
            importAllLoras(statusEl, selectedButton, allButton).catch(error => {
                statusEl.textContent = `Metadata import failed: ${error.message}`;
            });
        });

        autoToggle.addEventListener("change", () => {
            try {
                localStorage.setItem(AUTO_IMPORT_STORAGE_KEY, String(autoToggle.checked));
            } catch (e) {
                log("Unable to save auto-import setting:", e);
            }
            if (autoToggle.checked && getSelectedLoraReference()) {
                runSelectedImport();
            }
        });

        if (modelInput) {
            let importTimer;
            modelInput.addEventListener("change", () => {
                if (!autoToggle.checked) return;
                clearTimeout(importTimer);
                importTimer = setTimeout(runSelectedImport, 100);
            });
        }

        return true;
    }

    function watchForLoraManager() {
        if (addLoraManagerMetadataControls()) return;

        const observer = new MutationObserver(() => {
            if (addLoraManagerMetadataControls()) {
                observer.disconnect();
            }
        });
        observer.observe(document.documentElement, { childList: true, subtree: true });
    }



    /* =========================
       UI
    ========================= */

    function initializeMetadataReader() {
        if (addReaderOptions()) return;

        const observer = new MutationObserver(() => {
            if (addReaderOptions()) observer.disconnect();
        });
        observer.observe(document.documentElement, { childList: true, subtree: true });
    }

    function addReaderOptions() {
        if (document.getElementById("lora-metadata-reader")) return true;

        const editorSettings = document.getElementById("editor-settings");
        if (!editorSettings?.parentNode) return false;

        log('Add Metadata Reader Settings');

        const box = document.createElement('div');
        box.id = "lora-metadata-reader";
        box.classList.add('panel-box');
        box.innerHTML = `
            <h4 class="collapsible">LoRa Metadata Reader</h4>
            <div class="collapsible-content" style="display:block;margin-top:15px;">
                <div style="display:flex;gap:8px;">
                    <select id="loraSelect" style="flex:1;"></select>
                    <button class="metadatabtn" id="refreshLora">Refresh</button>
                </div>
                <small id="loraStatus"></small>
                <p></p>

                <label>
                    <input type="checkbox" id="loraClickToKeywordsToggle">
                    Clicking a trigger adds it to LoRa Keywords
                </label>
                <p></p>

                <label><input type="checkbox" id="loraTriggerToggle"> Show only trigger words</label>
                <p></p>

                <label>
                    <input type="radio" name="loraClickPromptTarget" id="loraClickPositiveToggle" checked>
                    Clicking a trigger adds it to positive prompt
                </label>
                <p></p>

                <label>
                    <input type="radio" name="loraClickPromptTarget" id="loraClickNegativeToggle">
                    Clicking a trigger adds it to negative prompt
                </label>
                <p></p>


                <div id="lora-metadata-results"></div>
            </div>
        `;

        editorSettings.parentNode.insertBefore(box, editorSettings.nextSibling);
        createCollapsibles(box);

        document.getElementById("loraTriggerToggle").checked = defaultTrigger;
        document.getElementById("loraClickToKeywordsToggle").checked = defaultClickToKeywords;

        const sel = box.querySelector("#loraSelect");
        const stat = box.querySelector("#loraStatus");

        loadLoraList(sel, stat);

        box.querySelector("#refreshLora").onclick = () => loadLoraList(sel, stat);

        sel.onchange = async () => {
            if (!sel.value) return;

            try {
                stat.textContent = "Reading metadata…";
                selectLoraInMainTab(sel.value);
                const meta = await fetchMetadata(sel.value);
                showMetadata(meta, sel.value.split("/").pop());
                stat.textContent = "Metadata loaded";
            } catch (error) {
                stat.textContent = `Metadata scan failed: ${error.message}`;
            }
        };

        return true;
    }

    function injectLoaderCSS() {
        log('Inject CSS');
        const style = document.createElement('style');
        style.textContent = `
            .metadatabtn{background:#2168bf}
            .trigger-word{cursor:pointer;padding:2px 4px;border-radius:3px;background:#333;color:#fff;margin:1px;display:inline-block;user-select:none}
            .trigger-word:hover,.trigger-word.clicked{background:#2e8b57}
            #lora-metadata-auto-import{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:8px 0}
            #lora-metadata-auto-import label{display:inline-flex;align-items:center;gap:5px}
            #lora-metadata-import-status{display:block;flex-basis:100%;color:var(--small-label-color)}
        `;
        document.head.appendChild(style);
    }

    /* =========================
       Renderer (trigger words only)
    ========================= */

    function showMetadata(metadata, fileName) {
        const results = document.querySelector('#lora-metadata-results');
        results.innerHTML = '';

        const panel = document.createElement('div');
        panel.style.background = '#222';
        panel.style.color = '#fff';
        panel.style.padding = '10px';
        panel.style.border = '1px solid #444';
        panel.style.marginTop = '10px';
        panel.style.whiteSpace = "pre-wrap";
        panel.style.wordBreak = "break-word";
        panel.style.overflowWrap = "break-word";
        panel.style.maxWidth = "100%";

        const onlyImportant = document.querySelector('#loraTriggerToggle')?.checked;

        let html = `<br><button class="metadatabtn" id="clearMetadataBtn">Close Panel</button>`;
        html += `<p></p>`;
        html += `<b>File:</b> ${escapeHtml(fileName)}<br><br>`;

        if (metadata && metadata.__metadata__) {
            const meta = metadata.__metadata__;

            if (onlyImportant) {
                // Trigger words only
                if (meta.ss_trigger_words) {
                    const triggers = Array.isArray(meta.ss_trigger_words)
                        ? meta.ss_trigger_words
                        : String(meta.ss_trigger_words).split(/[,\s]+/);

                    const triggerHtml = triggers
                        .map(t => (t || "").trim())
                        .filter(Boolean)
                        .map(word => `<span class="trigger-word" data-word="${escapeHtml(word)}">${escapeHtml(word)}</span>`)
                        .join(' ');

                    html += `<b>Trigger Words:</b><br>${triggerHtml || "<i>None</i>"}`;
                } else {
                    html += `<b>Trigger Words:</b><br><i>None found.</i>`;
                }
            } else {
                // Full metadata dump (still no ss_tag_frequency special-casing)
                html += `<b>Embedded Metadata:</b><br>`;
                for (const key in meta) {
                    html += `<b>${escapeHtml(key)}:</b> ${escapeHtml(String(meta[key]))}<br>`;
                }
            }
        } else {
            html += `<i>No embedded metadata found.</i><br>`;
            html += `<small>This LoRA may have been compiled without captions or trigger words.</small>`;
        }

        html += `<p></p><button class="metadatabtn" id="clearMetadataBtn2">Close Panel</button>`;

        panel.innerHTML = html;
        results.appendChild(panel);

        // Hook clicks AFTER inserting HTML
        wireTriggerClicks(panel);

        // Close buttons
        const close = () => { results.innerHTML = ''; };
        panel.querySelector('#clearMetadataBtn')?.addEventListener('click', close);
        panel.querySelector('#clearMetadataBtn2')?.addEventListener('click', close);
    }

    function wireTriggerClicks(panel) {
        panel.querySelectorAll('.trigger-word').forEach(span => {

            // SINGLE CLICK = positive / negative prompt
span.addEventListener('click', (e) => {
    e.preventDefault();

    const word = span.textContent.trim();

    span.classList.toggle('clicked');

    const addToPositive = document.getElementById('loraClickPositiveToggle')?.checked;
    const addToNegative = document.getElementById('loraClickNegativeToggle')?.checked;

    // With radios, exactly one should be selected, but keep this guard anyway.

    if (addToPositive) {
        applyWordToPrompt('#prompt', word);
    } else if (addToNegative) {
        applyWordToPrompt('#negative_prompt', word);
    }
});

            // DOUBLE CLICK = add to LoRa Keywords
            span.addEventListener('dblclick', (e) => {
                e.preventDefault();
                e.stopPropagation();

                const word = span.textContent.trim();
                if (!word) return;

                appendToLoRaKeywords(word);
            });
        });
    }





    function appendToLoRaKeywords(text) {
        const textarea = document.getElementById("lora-manager-keywords");
        if (!textarea) return;

        const line = String(text || "").trim();
        if (!line) return;

        // Prevent duplicates (line-based)
        const existing = textarea.value
            .split("\n")
            .map(s => s.trim())
            .filter(Boolean);

        if (existing.includes(line)) return;

        textarea.value = (textarea.value.trim() !== "")
            ? (textarea.value.replace(/\s*$/, "") + "\n" + line + "\n")
            : (line + "\n");

        ["input", "change"].forEach(evtType => {
            textarea.dispatchEvent(new Event(evtType, { bubbles: true }));
        });
    }

    // Tiny HTML escape so tags like "<" don't break the panel
    function escapeHtml(s) {
        return String(s)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function watchForPlugin(timeout = 5000) {
        return new Promise(resolve=>{
            if (document.getElementById("exportModifiers")) return resolve(true);
            const obs=new MutationObserver(()=>{
                if (document.getElementById("exportModifiers")){
                    obs.disconnect(); resolve(true);
                }
            });
            obs.observe(document.body,{childList:true,subtree:true});
            setTimeout(()=>{obs.disconnect();resolve(false)},timeout);
        });
    }


})();
