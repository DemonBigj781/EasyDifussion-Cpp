(() => {
    "use strict";
    function crc32(text) {
        let crc = -1;
        for (const byte of new TextEncoder().encode(text)) {
            crc ^= byte;
            for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
        }
        return (crc ^ -1) >>> 0;
    }
    function seedFor(request, mode) {
        if (mode === "high") return crc32(`${request.prompt}_${request.seed}`);
        if (mode === "very_high") return crc32(`${request.prompt}_${request.negative_prompt || ""}_${request.seed}`);
        if (mode === "extreme") return Math.floor(Math.random() * 10000000);
        return request.seed;
    }
    function expand(text, limit = 256) {
        const match = /\{([^{}]+)\}/.exec(text);
        if (!match) return [text];
        const suffixes = expand(text.slice(match.index + match[0].length), limit);
        const choices = match[1].split(",").map(value => value.trim());
        if (choices.length * suffixes.length > limit) throw new Error(`Prompt expansion limit is ${limit}.`);
        return choices.flatMap(choice => suffixes.map(suffix => text.slice(0, match.index) + choice + suffix));
    }
    function diff(before, after) {
        let first = 0, last = 0;
        while (first < Math.min(before.length, after.length) && before[first] === after[first]) first++;
        while (last < Math.min(before.length, after.length) - first && before.at(-1-last) === after.at(-1-last)) last++;
        return [
            {type: "equal", text: before.slice(0, first)},
            {type: "delete", text: before.slice(first, before.length - last)},
            {type: "insert", text: after.slice(first, after.length - last)},
            {type: "equal", text: last ? before.slice(-last) : ""},
        ].filter(part => part.text);
    }
    const math = {crc32, seedFor, expand, diff};
    if (typeof module !== "undefined" && module.exports) { module.exports = math; return; }
    window.CppOptionalPluginMath = math;

    (async () => {
        await window.CppKiosk.ready;
        const prefs = window.LocalPluginPreferences, byId = id => document.getElementById(id);
        const enabled = id => prefs.isEnabled(id);
        const originalTitle = document.title;
        const controls = document.createElement("section");
        controls.id = "cpp-optional-plugins";
        controls.className = "panel-box";
        const heading = document.createElement("h3"); heading.textContent = "Generation plugins"; controls.append(heading);
        document.querySelector(".generation-queue-controls").after(controls);
        const rows = [];
        const read = (key, fallback) => { try { return localStorage.getItem(key) ?? fallback; } catch (_) { return fallback; } };
        function field(plugin, id, label, options, fallback, type = "select") {
            const row = document.createElement("label"); row.htmlFor = id; row.textContent = label;
            const input = document.createElement(type === "select" ? "select" : "input"); input.id = id;
            if (type === "select") for (const [value, text] of options) input.add(new Option(text, value));
            else input.type = type;
            if (type === "checkbox") input.checked = read(id, fallback) === "true";
            else input.value = read(id, fallback);
            input.addEventListener("change", () => {
                localStorage.setItem(id, type === "checkbox" ? input.checked : input.value); refresh();
            });
            row.append(input); controls.append(row); rows.push([plugin,row]); return input;
        }
        const invocation = field("accessibility-improvements", "contextual_menu_invocation", "Image actions", [
            ["hover","Hover / keyboard focus"],["left_click","Left-click"],["middle_click","Middle-click"],
            ["right_click","Right-click"],["ctrl_left_click","Ctrl + left-click"],
            ["ctrl_middle_click","Ctrl + middle-click"],["ctrl_right_click","Ctrl + right-click"],
        ], "hover");
        const rightClick = field("accessibility-improvements", "disable_right_click", "Disable page context menu (text fields remain available)", [], "false", "checkbox");
        const zoom = field("disable-source-image-zoom", "disable_source_image_zoom", "Zoom source image on hover", [], "true", "checkbox");
        const gpu = field("gpu-mode-quick-toggle", "vram_usage_level", "GPU memory mode", [["balanced","Balanced"],["low","Low memory"]], "balanced");
        const order = field("processing-order-quick-toggle", "cpp-processing-order", "Processing order", [["oldest","Oldest first"],["newest","Newest first"]], "oldest");
        const random = field("random-seed-quick-toggle", "cpp-random-seed", "Random seed", [], "true", "checkbox");
        const variability = field("seed-randomizer", "seed_randomizer_behavior", "Batch seed variability", [["normal","Normal"],["high","High (prompt)"],["very_high","Very high (both prompts)"],["extreme","Extreme (random)"]], "normal");
        const language = field("prompt-translator", "prompt_language", "Prompt language → English (sends both prompts to Google Translate)", [["en","English — no translation"],["auto","Detect language"]], "en");
        // Same language list as the legacy translator, supplied as a local static resource.
        fetch("/cpp-ui/scripts/translation_languages.json").then(response => {
            if (!response.ok) throw Error("Could not load translation languages."); return response.json();
        }).then(languages => {
            for (const [label, code] of Object.entries(languages)) if (!["en","auto"].includes(code)) language.add(new Option(label, code));
            language.value = read("prompt_language", "en");
        }).catch(error => { language.title = error.message; });
        const consent = field("prompt-translator", "cpp-translation-consent", "Allow sending prompts to Google Translate", [], "false", "checkbox");
        const daily = document.createElement("p"); daily.textContent = "Daily folders: outputs saved by the server use today's YYYY-MM-DD session folder.";
        controls.append(daily); rows.push(["daily-folders",daily]);
        const count = document.createElement("output"); count.id = "cpp-queue-counter"; count.setAttribute("aria-live", "polite");
        controls.append(count); rows.push(["queue-counter",count]);
        const sticky = document.createElement("button"); sticky.id = "cpp-floating-generate"; sticky.type = "button"; sticky.textContent = "Generate";
        sticky.addEventListener("click", () => byId("makeImage").click()); document.body.append(sticky);
        const style = document.createElement("style"); style.textContent = `
            #cpp-optional-plugins[hidden], #cpp-optional-plugins [hidden], #cpp-floating-generate[hidden] {display:none!important}
            #cpp-optional-plugins label {display:flex;flex-wrap:wrap;align-items:center;gap:.6rem;margin:.6rem 0}
            #cpp-optional-plugins select {max-width:100%}
            #cpp-floating-generate {position:fixed;right:1rem;bottom:1rem;z-index:1000;box-shadow:0 2px 12px #0008}
            html[data-kiosk=loading] #cpp-floating-generate,html[data-kiosk=unavailable] #cpp-floating-generate {display:none!important}
            .cpp-image-actions {display:flex;gap:.4rem;flex-wrap:wrap}
            html[data-cpp-actions] .cpp-image-actions {visibility:hidden}
            html[data-cpp-actions] .cpp-generated-image:focus-within .cpp-image-actions,
            html[data-cpp-actions] .cpp-generated-image[data-actions-open] .cpp-image-actions,
            html[data-cpp-actions=hover] .cpp-generated-image:hover .cpp-image-actions {visibility:visible}
            .cpp-source-preview {max-width:80px;max-height:80px;object-fit:contain;transition:transform .15s}
            html:not([data-cpp-no-source-zoom]) .cpp-source-preview:hover {transform:scale(2);transform-origin:top left}
            .cpp-prompt-diff {overflow-wrap:anywhere;margin:.5rem 0}
            .cpp-prompt-diff ins {color:#77d8a5} .cpp-prompt-diff del {color:#ffb3b3}
        `; document.head.append(style);
        let state = {active:false,pending:0,images:0}, lastPrompt = null, lastFixed = "0";
        function refresh() {
            for (const [id,row] of rows) row.hidden = !enabled(id);
            controls.hidden = rows.every(([,row]) => row.hidden);
            sticky.hidden = !enabled("make-image-always-visible");
            document.documentElement.toggleAttribute("data-cpp-no-source-zoom", enabled("disable-source-image-zoom") && !zoom.checked);
            if (enabled("accessibility-improvements")) document.documentElement.dataset.cppActions = invocation.value;
            else document.documentElement.removeAttribute("data-cpp-actions");
            if (window.CppGeneration) window.CppGeneration.newestFirst = enabled("processing-order-quick-toggle") && order.value === "newest";
            count.textContent = `${state.images} images / ${state.pending + Number(state.active)} tasks remaining`;
            document.title = enabled("queue-counter") && state.images ? `(${state.images}/${state.pending+Number(state.active)}) ${originalTitle}` : originalTitle;
        }
        random.addEventListener("change", () => {
            if (random.checked) { lastFixed = byId("seed").value; byId("seed").value = "-1"; }
            else byId("seed").value = lastFixed === "-1" ? "0" : lastFixed;
            byId("seed").dispatchEvent(new Event("change",{bubbles:true}));
        });
        byId("seed").addEventListener("change", () => {random.checked = Number(byId("seed").value) < 0;});
        window.addEventListener("cpp-generation-state", event => {state = event.detail; refresh();});
        window.addEventListener("cpp-generation-ready", refresh);
        window.addEventListener("local-plugin-preferences-changed", refresh);
        window.addEventListener("contextmenu", event => {
            if (!enabled("accessibility-improvements")) return;
            if ((rightClick.checked && !event.target.closest("input,textarea,img")) ||
                (event.target.closest(".cpp-generated-image") && invocation.value.includes("right_click"))) event.preventDefault();
        });
        window.addEventListener("mouseup", event => {
            if (!enabled("accessibility-improvements") || invocation.value === "hover") return;
            if (event.target.closest("button,a,input,select,textarea")) return;
            const button = invocation.value.includes("middle") ? 1 : invocation.value.includes("right") ? 2 : 0;
            if (event.button !== button || event.ctrlKey !== invocation.value.startsWith("ctrl_")) return;
            const card = event.target.closest(".cpp-generated-image");
            const open = card?.hasAttribute("data-actions-open");
            document.querySelectorAll("[data-actions-open]").forEach(item => item.removeAttribute("data-actions-open"));
            if (card && !open) card.setAttribute("data-actions-open", "");
        });
        const translations = new Map();
        async function translate(text, locale, signal) {
            if (!text.trim() || locale === "en") return text;
            const key = `${locale}:${text}`;
            if (translations.has(key)) return translations.get(key);
            const params = new URLSearchParams({client:"gtx",sl:locale,tl:"en",dt:"t",q:text});
            const response = await fetch(`https://translate.googleapis.com/translate_a/single?${params}`, {signal});
            if (!response.ok) throw Error(`Google Translate failed (HTTP ${response.status}).`);
            const data = await response.json();
            if (!Array.isArray(data[0]) || !data[0].every(part => typeof part[0] === "string")) throw Error("Invalid translation response.");
            const result = data[0].map(part => part[0]).join("");
            if (translations.size >= 128) translations.clear();
            translations.set(key,result); return result;
        }
        window.CppOptionalPlugins = {
            expandRequests(request) {
                const positives = expand(request.prompt), negatives = expand(request.negative_prompt || "");
                if (positives.length * negatives.length > 256) throw Error("Prompt expansion limit is 256.");
                return positives.flatMap(prompt => negatives.map(negative_prompt => ({...request,prompt,negative_prompt})));
            },
            async prepare(request, signal) {
                if (enabled("gpu-mode-quick-toggle")) request.vram_usage_level = gpu.value;
                if (enabled("daily-folders")) {
                    const now = new Date();
                    request.session_id = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,"0")}-${String(now.getDate()).padStart(2,"0")}`;
                }
                if (enabled("seed-randomizer")) request.seed = seedFor(request, variability.value);
                if (enabled("prompt-translator") && language.value !== "en") {
                    if (!language.value) throw Error("Wait for the translation language list to load, or reload if loading failed.");
                    if (!consent.checked) throw Error("Allow sending prompts to Google Translate before using translation.");
                    request.prompt = await translate(request.prompt,language.value,signal);
                    request.negative_prompt = await translate(request.negative_prompt || "",language.value,signal);
                }
            },
            decorate(card, request) {
                if (enabled("prompt-diff") && lastPrompt) {
                    const section = document.createElement("div"); section.className = "cpp-prompt-diff";
                    for (const key of ["prompt","negative_prompt"]) {
                        const line = document.createElement("p"); line.append(key === "prompt" ? "Prompt diff: " : "Negative prompt diff: ");
                        for (const part of diff(lastPrompt[key] || "", request[key] || "")) {
                            const node = document.createElement(part.type === "insert" ? "ins" : part.type === "delete" ? "del" : "span");
                            node.textContent = part.text; line.append(node);
                        }
                        section.append(line);
                    }
                    card.append(section);
                }
                lastPrompt = {...request};
            },
        };
        refresh();
    })().catch(error => console.error("Modern optional plugins:", error));
})();
