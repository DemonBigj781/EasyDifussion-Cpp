(() => {
    "use strict";
    if (window.CppGeneratePage) return;
    window.CppGeneratePage = true;

    const byId = id => document.getElementById(id);
    const status = byId("generation-queue-status");
    const progress = byId("generation-progress");
    const preview = byId("preview-content");
    const start = byId("makeImage");
    const stop = byId("stopImage");
    let activeTask = null;
    let stopped = false;
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
            select.replaceChildren(new Option("Select a checkpoint", ""));
            for (const model of models) {
                const value = String(model.model || model.name || "");
                if (value) select.add(new Option(String(model.name || value), value));
            }
            try {
                const saved = localStorage.getItem("easy-diffusion-cpp-checkpoint-v1");
                if (saved && Array.from(select.options).some(option => option.value === saved)) select.value = saved;
            } catch (_) {}
            select.addEventListener("change", () => {
                try { localStorage.setItem("easy-diffusion-cpp-checkpoint-v1", select.value); } catch (_) {}
            });
            if (modelStatus) modelStatus.textContent = models.length ? `${models.length} checkpoints available.` : "No checkpoints found.";
        } catch (error) {
            select.replaceChildren(new Option("Could not load checkpoints", ""));
            if (modelStatus) modelStatus.textContent = `Checkpoint list unavailable (${error.name}).`;
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
    function showImages(result) {
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
            preview.prepend(card);
        }
        byId("initial-text")?.remove();
    }
    async function pollTask(streamUrl) {
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
                    showImages(event);
                    completed = true;
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
        const checkpoint = byId("stable_diffusion_model")?.value || "";
        if (!checkpoint) throw new Error("Select a checkpoint before generating.");
        const seedInput = integer("seed", -1, -1, 2147483647);
        const seed = seedInput < 0 ? Math.floor(Math.random() * 2147483647) : seedInput;
        const request = {
            prompt: byId("prompt")?.value || "",
            negative_prompt: byId("negative_prompt")?.value || "",
            use_stable_diffusion_model: checkpoint,
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
        window.CppGeneratePlugins?.applyToRequest(request);
        try {
            const saved = JSON.parse(localStorage.getItem("easy-diffusion-native-device-routing-v1") || "{}");
            request.backend_assignment = Object.entries(saved.image || {})
                .filter(([, value]) => typeof value === "string" && value)
                .map(([key, value]) => `${key}=${value}`).join(",");
        } catch (_) {}
        return request;
    }
    async function startGeneration() {
        if (!start || activeTask) return;
        stopped = false;
        start.disabled = true;
        if (stop) stop.disabled = false;
        if (progress) { progress.max = 1; progress.value = 0; }
        try {
            const request = buildRequest();
            if (!request.prompt.trim()) throw new Error("Enter a prompt first.");
            setStatus("Adding image to the queue…");
            const response = await fetch("/render", {
                method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(request),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.detail || `Render request returned HTTP ${response.status}`);
            activeTask = payload.task;
            await pollTask(payload.stream || `/image/stream/${activeTask}`);
            if (stopped) setStatus("Generation stopped.");
            else setStatus("Generation complete.");
        } catch (error) {
            setStatus(error.message || "Generation failed.", true);
        } finally {
            activeTask = null;
            start.disabled = false;
            if (stop) stop.disabled = true;
            if (progress) progress.value = 0;
        }
    }
    start?.addEventListener("click", startGeneration);
    stop?.addEventListener("click", async () => {
        if (!activeTask) return;
        stopped = true;
        setStatus("Stopping generation…");
        try { await fetch(`/image/stop?task=${encodeURIComponent(activeTask)}`, {cache:"no-store"}); } catch (_) {}
    });
    byId("clear-all-previews")?.addEventListener("click", () => {
        preview?.replaceChildren();
    });
    loadCheckpoints();
})();
