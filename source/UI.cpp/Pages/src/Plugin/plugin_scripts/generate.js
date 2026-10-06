(async () => {
    "use strict";
    if (window.CppGeneratePage) return;
    let policy;
    try { policy = await window.CppKiosk.ready; } catch (_) { return; }
    window.CppGeneratePage = true;

    const byId = id => document.getElementById(id);
    for (const id of ["width", "height"]) {
        const slider = byId(id), output = byId(`${id}-value`);
        const update = () => { if (output) output.textContent = `${slider.value} px`; };
        slider.addEventListener("input", update);
        slider.addEventListener("change", update);
        update();
    }
    const status = byId("generation-queue-status");
    const progress = byId("generation-progress");
    const preview = byId("preview-content");
    const start = byId("makeImage");
    const stop = byId("stopImage");
    const checkpointPicker = new ModelDropdown(byId("stable_diffusion_model"), null, "Select a checkpoint");
    const vaePicker = new ModelDropdown(byId("vae_model"), null, "Use checkpoint VAE");
    const textEncoderPicker = new ModelDropdown(byId("text_encoder_model"), null, "Use checkpoint text encoder");
    let companionsReady = false;
    let activeTask = null;
    let stopped = false;
    let preparation = null;
    const queue = window.CppGenerationQueue.createQueue({run: runRequest, changed: state => {
        if (stop) stop.disabled = !state.active && !state.pending;
        window.dispatchEvent(new CustomEvent("cpp-generation-state", {detail: state}));
    }});
    if (stop) stop.disabled = true;
    const sessionKey = "easy-diffusion-cpp-session-v1";
    let sessionId = "";
    try {
        sessionId = localStorage.getItem(sessionKey) || (crypto.randomUUID ? crypto.randomUUID() : `cpp-${Date.now()}`);
        localStorage.setItem(sessionKey, sessionId);
    } catch (_) { sessionId = `cpp-${Date.now()}`; }

    function setStatus(text, isError = false) {
        if (!status) return;
        status.textContent = text;
        status.dataset.state = isError ? "error" : "";
    }
    function inputValue(id, fallback) {
        const value = byId(id)?.value;
        return value === undefined || value === "" ? fallback : value;
    }
    function integer(id, fallback, minimum, maximum) {
        const value = Number.parseInt(inputValue(id, String(fallback)), 10);
        return Math.min(maximum, Math.max(minimum, Number.isFinite(value) ? value : fallback));
    }
    function number(id, fallback, minimum, maximum) {
        const value = Number.parseFloat(inputValue(id, String(fallback)));
        return Math.min(maximum, Math.max(minimum, Number.isFinite(value) ? value : fallback));
    }
    async function loadCheckpoints() {
        const select = byId("stable_diffusion_model");
        const modelStatus = byId("generation-model-status");
        if (!select) return;
        try {
            const response = await fetch("/get/model", {cache:"no-store"});
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            const models = Array.isArray(payload.models) ? payload.models : [];
            checkpointPicker.inputModels = buildTree(models.map(model => ({model: String(model.model || model.name || "")})));
            checkpointPicker.populateModels();
            try {
                const saved = window.CppInputPreferences.get("stable_diffusion_model", "easy-diffusion-cpp-checkpoint-v1");
                if (saved && (!policy.enabled || models.some(model => model.model === saved))) checkpointPicker.value = saved;
                else if (policy.enabled && models.length) checkpointPicker.value = models[0].model;
            } catch (_) {}
            select.addEventListener("change", () => {
                window.CppInputPreferences.save("stable_diffusion_model", checkpointPicker.value);
            });
            if (modelStatus) modelStatus.textContent = models.length ? `${models.length} checkpoints available.` : "No checkpoints found.";
        } catch (error) {
            select.disabled = true;
            select.placeholder = "Could not load checkpoints";
            if (modelStatus) modelStatus.textContent = `Checkpoint list unavailable (${error.name}).`;
        }
    }

    async function loadCompanions() {
        const message = byId("generation-companion-status");
        try {
            for (const [picker, endpoint, type, key] of [
                [vaePicker, "/get/vae", "vae", "easy-diffusion-cpp-vae-v1"],
                [textEncoderPicker, "/get/other", "text-encoder", "easy-diffusion-cpp-text-encoder-v1"],
            ]) {
                const response = await fetch(endpoint, {cache: "no-store"});
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const payload = await response.json();
                const models = (payload.models || []).filter(model => model.tags?.[0] === type);
                picker.inputModels = buildTree(models);
                picker.populateModels();
                try {
                    const saved = window.CppInputPreferences.get(type === "vae" ? "vae_model" : "text_encoder_model", key);
                    if (saved && models.some(model => model.model === saved)) picker.value = saved;
                } catch (_) {}
                picker.addEventListener("change", () => {
                    window.CppInputPreferences.save(type === "vae" ? "vae_model" : "text_encoder_model", picker.value || "");
                });
            }
            companionsReady = true;
        } catch (error) {
            if (message) message.textContent = `Could not load companion models: ${error.message}. Reload before generating.`;
        }
    }

    function splitJsonValues(text) {
        const values = [];
        let startAt = -1, depth = 0, quoted = false, escaped = false;
        for (let i = 0; i < text.length; i++) {
            const c = text[i];
            if (startAt < 0) {
                if (c === "{" || c === "[") { startAt = i; depth = 1; }
                continue;
            }
            if (quoted) {
                if (escaped) escaped = false;
                else if (c === "\\") escaped = true;
                else if (c === '"') quoted = false;
                continue;
            }
            if (c === '"') quoted = true;
            else if (c === "{" || c === "[") depth++;
            else if (c === "}" || c === "]") {
                depth--;
                if (depth === 0) {
                    try { values.push(JSON.parse(text.slice(startAt, i + 1))); } catch (_) {}
                    startAt = -1;
                }
            }
        }
        return values;
    }
    function showImages(result, request) {
        if (!preview) return;
        const outputs = Array.isArray(result.output) ? result.output : [];
        for (const [index, item] of outputs.entries()) {
            if (!item?.data) continue;
            const card = document.createElement("figure");
            card.className = "cpp-generated-image";
            const image = document.createElement("img");
            const format = result.task_data?.output_format || "jpeg";
            image.src = item.data.startsWith("data:") ? item.data : `data:image/${format};base64,${item.data}`;
            image.alt = `Generated image ${index + 1}, seed ${item.seed ?? "unknown"}`;
            const caption = document.createElement("figcaption");
            caption.textContent = `Seed ${item.seed ?? "unknown"}`;
            card.append(image, caption);
            card.tabIndex = 0;
            const actions = document.createElement("div"); actions.className = "cpp-image-actions";
            const download = document.createElement("a"); download.href = image.src;
            download.download = `image-${item.seed ?? index}.${format}`; download.textContent = "Download";
            const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "Remove";
            remove.addEventListener("click", () => card.remove()); actions.append(download, remove); card.append(actions);
            if (request.init_image) {
                const source = document.createElement("img"); source.src = request.init_image;
                source.className = "cpp-source-preview"; source.alt = "Source image"; card.append(source);
            }
            const effective = {...request, seed: item.seed ?? request.seed};
            card.cppRequest = effective;
            window.CppOptionalPlugins?.decorate(card, effective);
            preview.prepend(card);
            window.dispatchEvent(new CustomEvent("cpp-generation-result", {detail: {request:effective,result,item,card,image}}));
        }
        byId("initial-text")?.remove();
    }
    async function pollTask(streamUrl, request) {
        while (!stopped) {
            const response = await fetch(streamUrl, {cache:"no-store"});
            if (response.status === 425) {
                setStatus("Waiting for a render worker…");
                await new Promise(resolve => setTimeout(resolve, 500));
                continue;
            }
            if (!response.ok) throw new Error(`Render stream returned HTTP ${response.status}`);
            const events = splitJsonValues(await response.text());
            let completed = false;
            for (const event of events) {
                if (event.status === "failed") throw new Error(event.detail || "Generation failed.");
                if (event.status === "succeeded") {
                    if (!stopped) showImages(event, request);
                    completed = true;
                    if (!stopped) return event;
                } else if (Number.isFinite(event.step) && Number.isFinite(event.total_steps)) {
                    if (progress) { progress.max = event.total_steps; progress.value = event.step; }
                    setStatus(`Rendering: step ${event.step} of ${event.total_steps}`);
                }
            }
            if (completed) return;
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }
    function buildRequest() {
        const checkpoint = checkpointPicker.value || "";
        if (!checkpoint) throw new Error("Select a checkpoint before generating.");
        if (!companionsReady) throw new Error("Wait for the VAE and text-encoder lists to load, or reload if loading failed.");
        const seedInput = integer("seed", -1, -1, 2147483647);
        const seed = seedInput < 0 ? Math.floor(Math.random() * 2147483647) : seedInput;
        const request = {
            prompt: byId("prompt")?.value || "",
            negative_prompt: byId("negative_prompt")?.value || "",
            use_stable_diffusion_model: checkpoint,
            use_vae_model: vaePicker.value || "",
            use_text_encoder_model: textEncoderPicker.value || "",
            width: integer("width", 512, 64, 2048),
            height: integer("height", 512, 64, 2048),
            num_outputs: integer("num_images", 1, 1, 8),
            num_inference_steps: integer("steps", 25, 1, 200),
            guidance_scale: number("guidance_scale", 7.5, 0, 30),
            seed,
            session_id: sessionId,
            output_format: byId("output_format")?.value || "jpeg",
            stream_image_progress: true,
            vram_usage_level: "balanced",
        };
        if (byId("sampler")?.value) request.sampler_name = byId("sampler").value;
        window.CppImageModifiers?.applyToRequest(request);
        window.CppGeneratePlugins?.applyToRequest(request);
        window.CppImageTools?.applyToRequest(request);
        Object.assign(request, window.CppIPAdapter?.requestOptions() || {});
        try {
            const saved = JSON.parse(localStorage.getItem("easy-diffusion-native-device-routing-v1") || "{}");
            request.backend_assignment = Object.entries(saved.image || {})
                .filter(([, value]) => typeof value === "string" && value)
                .map(([key, value]) => `${key}=${value}`).join(",");
        } catch (_) {}
        return request;
    }
    async function runRequest(request, isCancelled) {
        stopped = false;
        if (progress) { progress.max = 1; progress.value = 0; }
        try {
            if (!request.prompt.trim()) throw new Error("Enter a prompt first.");
            if (!["on", "off"].includes(document.documentElement.dataset.kiosk))
                throw new Error("Cannot verify kiosk policy. Reload before generating.");
            preparation = new AbortController();
            await window.CppOptionalPlugins?.prepare(request, preparation.signal);
            preparation = null;
            if (isCancelled()) throw new Error("Generation cancelled.");
            setStatus("Adding image to the queue…");
            const response = await fetch("/render", {
                method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(request),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.detail || `Render request returned HTTP ${response.status}`);
            activeTask = payload.task;
            if (isCancelled()) { await stopActiveTask(); throw new Error("Generation cancelled."); }
            const result = await pollTask(payload.stream || `/image/stream/${activeTask}`, request);
            if (stopped) setStatus("Generation stopped.");
            else setStatus("Generation complete.");
            return result;
        } catch (error) {
            setStatus(error.message || "Generation failed.", true);
            throw error;
        } finally {
            activeTask = null;
            preparation = null;
            if (progress) progress.value = 0;
        }
    }
    async function startGeneration() {
        try {
            const request = buildRequest();
            const requests = window.CppOptionalPlugins?.expandRequests(request) || [request];
            await Promise.all(requests.map(request => queue.enqueue(request)));
        } catch (error) { setStatus(error.message || "Generation failed.", true); }
    }
    async function stopActiveTask() {
        if (!activeTask) return;
        const response = await fetch(`/image/stop?task=${encodeURIComponent(activeTask)}`, {cache:"no-store"});
        if (!response.ok) throw new Error(`Could not stop task (HTTP ${response.status}).`);
    }
    async function cancel() {
        stopped = true;
        queue.cancel();
        preparation?.abort();
        window.dispatchEvent(new Event("cpp-generation-cancel"));
        setStatus("Stopping generation…");
        try { await stopActiveTask(); }
        catch (error) { setStatus(error.message, true); throw error; }
    }
    start?.addEventListener("click", startGeneration);
    stop?.addEventListener("click", () => cancel().catch(() => {}));
    window.CppGeneration = {buildRequest, enqueue: request => queue.enqueue({...request, session_id:sessionId}), cancel,
        get state() { return queue.state; },
        get newestFirst() { return queue.newestFirst; },
        set newestFirst(value) { queue.newestFirst = value; },
    };
    window.dispatchEvent(new Event("cpp-generation-ready"));
    byId("clear-all-previews")?.addEventListener("click", () => {
        preview?.replaceChildren();
    });
    Promise.all([loadCheckpoints(), loadCompanions()]).then(() => window.CppInputPreferences.bindBasic());
})();
