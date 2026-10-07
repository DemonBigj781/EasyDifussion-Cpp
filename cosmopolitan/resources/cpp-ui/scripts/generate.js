(() => {
    "use strict";
    if (window.CppNativeGeneration) return;
    window.CppNativeGeneration = true;
    const byId = id => document.getElementById(id);
    const controls = document.querySelector(".generation-controls");
    if (!controls) return;
    const start = byId("makeImage"), stop = byId("stopImage");
    const status = byId("generation-queue-status"), progress = byId("generation-progress");
    const preview = byId("preview-content"), modelStatus = byId("generation-model-status");
    const editable = ["prompt", "negative_prompt", "seed", "width", "height", "steps", "guidance_scale", "sampler"];
    let ready = false, active = null, loadingModels = false, models = new Set();
    const devices = new Map();

    for (const control of controls.querySelectorAll("input,select,textarea,button")) control.disabled = true;
    byId("show-download-popup").disabled = true;
    byId("show-download-popup").title = "Use the PNG download link beneath each generated image.";
    byId("output_format").value = "png";
    byId("output_format").title = "The native image API returns PNG.";
    byId("num_images").value = "1";
    byId("num_images").title = "This interface runs one image per request.";
    for (const id of ["vae_model", "text_encoder_model"]) {
        byId(id).value = "";
        byId(id).placeholder = "Use a complete checkpoint";
        byId(id).title = "Separate companion models are not supported by this native form.";
    }
    byId("generation-companion-status").textContent = "Use a complete checkpoint with its text encoder and VAE. Separate components, LoRAs, ControlNet, image-to-image and generation plugins are not connected to this form.";
    byId("cpp-modifier-status").textContent = "Image modifier presets are unavailable here. Describe the style in the prompt.";
    byId("cpp-modifier-grid").setAttribute("aria-busy", "false");
    byId("lora_model").textContent = "LoRA selection is unavailable in this native form.";
    for (const id of ["generate-plugin-panels", "cpp-image-modifiers"]) {
        byId(id).setAttribute("aria-disabled", "true");
        byId(id).title = "Not supported by this native generation form.";
    }

    const oldCheckpoint = byId("stable_diffusion_model");
    const checkpoint = document.createElement("select");
    checkpoint.id = oldCheckpoint.id;
    checkpoint.name = oldCheckpoint.name;
    checkpoint.disabled = true;
    checkpoint.add(new Option("Loading checkpoints…", ""));
    oldCheckpoint.replaceWith(checkpoint);
    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.textContent = "Refresh checkpoints";
    refresh.disabled = true;
    modelStatus.after(refresh);
    const backendLabel = document.createElement("label");
    backendLabel.textContent = "Compute device ";
    const backend = document.createElement("select");
    backend.id = "cosmo-generation-backend";
    backend.disabled = true;
    backendLabel.append(backend);
    refresh.after(backendLabel);
    const backendHelp = document.createElement("p");
    backendHelp.textContent = "Software adapters run on the CPU. Hardware entries identify an actual integrated or discrete GPU. Unsupported operations may fall back to GGML CPU; device memory is shown only when measured.";
    backendLabel.after(backendHelp);
    const sampler = byId("sampler");
    sampler.replaceChildren(new Option("Euler", "euler"), new Option("Euler a", "euler_a"), new Option("DPM++ 2M", "dpm++2m"));
    for (const id of ["width", "height"]) {
        const update = () => { byId(`${id}-value`).textContent = `${byId(id).value} px`; };
        byId(id).addEventListener("input", update);
        update();
    }

    function message(text, failed = false) {
        status.textContent = text;
        status.dataset.state = failed ? "error" : "";
    }
    function updateControls() {
        const available = ready && active === null && !loadingModels;
        for (const id of editable) byId(id).disabled = !available;
        checkpoint.disabled = !available || models.size === 0;
        backend.disabled = !available;
        refresh.disabled = !available;
        start.disabled = !available || !models.has(checkpoint.value) || !devices.has(backend.value);
        stop.disabled = !active || !active.sampling || active.stopping;
    }
    async function request(path, body) {
        let response, text;
        try {
            response = await fetch(path, body === undefined ? {cache: "no-store"} : {
                method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
            });
            text = await response.text();
        } catch (error) {
            error.transportFailed = true;
            throw error;
        }
        let value;
        try { value = text ? JSON.parse(text) : {}; } catch (_) { value = {message: text}; }
        if (!response.ok) {
            const error = new Error(value.message || value.detail || `HTTP ${response.status}`);
            error.status = response.status;
            throw error;
        }
        return value;
    }
    function integer(id, min, max) {
        const value = Number(byId(id).value);
        if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`${id}: enter an integer between ${min} and ${max}.`);
        return value;
    }
    async function loadModels(rescan = false) {
        loadingModels = true;
        updateControls();
        try {
            if (rescan) await request("/v1/sdapi/v1/refresh-checkpoints", {});
            const result = await request("/v1/sdapi/v1/checkpoints");
            if (!Array.isArray(result.models) || result.models.some(model => typeof model.name !== "string")) throw new Error("Invalid checkpoint list from the native server");
            const previous = checkpoint.value;
            models = new Set(result.models.map(model => model.name));
            checkpoint.replaceChildren(new Option("Select a complete checkpoint", ""));
            for (const name of models) checkpoint.add(new Option(name, name));
            let saved = "";
            try { saved = localStorage.getItem("cosmo-native-checkpoint") || ""; } catch (_) {}
            if (models.has(previous)) checkpoint.value = previous;
            else if (models.has(saved)) checkpoint.value = saved;
            else if (models.size === 1) checkpoint.value = [...models][0];
            modelStatus.textContent = models.size ? `${models.size} checkpoint(s) indexed. The selected file must contain the components required by its architecture.` : "No checkpoints found. Start sdkit with --ckpt-dir pointing to your model directory, then refresh.";
        } finally { loadingModels = false; updateControls(); }
    }
    function endJob(job) {
        if (active !== job) return;
        clearTimeout(job.timer);
        active = null;
        updateControls();
    }
    async function poll(job) {
        if (active !== job) return;
        try {
            const state = await request("/v1/internal/progress", {id_task: job.id, live_preview: false});
            if (active !== job) return;
            if (state.completed && job.transportFailed) {
                endJob(job);
                message("The generation connection was lost. The server has finished, but its image response was not received. You may submit a new request.", true);
                return;
            }
            if (!state.completed && Number(state.current_step) > 0) {
                job.sampling = true;
                progress.max = Number(state.total_steps) || job.steps;
                progress.value = Number(state.current_step);
                if (!job.stopping && !job.transportFailed) message(`Sampling ${state.current_step} / ${state.total_steps}. The image still needs VAE decoding afterward.`);
            }
            updateControls();
        } catch (error) {
            if (active !== job) return;
            if (error.status !== 404 && !job.transportFailed) message(`Generation is still pending; progress is unavailable (${error.message}).`);
        }
        if (active === job) job.timer = setTimeout(() => poll(job), 750);
    }
    function showImage(data, job) {
        if (typeof data !== "string" || !data.startsWith("iVBORw0KGgo")) throw new Error("The native server did not return a PNG image.");
        const card = document.createElement("figure");
        card.className = "cpp-generated-image";
        const image = document.createElement("img");
        image.src = `data:image/png;base64,${data}`;
        image.alt = `Generated image: ${job.prompt}`;
        const caption = document.createElement("figcaption");
        caption.textContent = `${job.model} · seed ${job.seed} · ${job.backend}`;
        const download = document.createElement("a");
        download.href = image.src;
        download.download = `image-${job.seed}.png`;
        download.textContent = "Download PNG";
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "Remove";
        remove.addEventListener("click", () => card.remove());
        card.append(image, caption, download, remove);
        preview.prepend(card);
        byId("initial-text")?.remove();
    }
    async function generate() {
        if (!ready || active) return;
        let job;
        try {
            const prompt = byId("prompt").value.trim();
            if (!prompt) throw new Error("Enter a prompt first.");
            if (!models.has(checkpoint.value)) throw new Error("Select an indexed complete checkpoint.");
            if (!devices.has(backend.value)) throw new Error("Select an available compute device.");
            const width = integer("width", 64, 2048), height = integer("height", 64, 2048);
            if (width % 64 || height % 64) throw new Error("Width and height must be multiples of 64.");
            const steps = integer("steps", 1, 200);
            const cfg = Number(byId("guidance_scale").value);
            if (!Number.isFinite(cfg) || cfg < 0 || cfg > 30) throw new Error("Guidance must be between 0 and 30.");
            let seed = integer("seed", -1, 2147483647);
            if (seed === -1) seed = crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
            const id = `cosmo-${Date.now()}-${crypto.getRandomValues(new Uint32Array(1))[0]}`;
            job = {id, prompt, model: checkpoint.value, seed, steps, backend: backend.value, sampling: false, stopping: false, transportFailed: false, timer: null};
            active = job;
            updateControls();
            progress.removeAttribute("value");
            message("Loading the checkpoint and preparing generation. Stop becomes available after the first sampling step completes.");
            const body = {
                force_task_id: id, prompt, negative_prompt: byId("negative_prompt").value,
                width, height, steps, cfg_scale: cfg, seed, batch_size: 1,
                sampler_name: sampler.value, scheduler: "discrete", backend: job.backend,
                override_settings: {sd_model_checkpoint: job.model, forge_additional_modules: [], live_previews_enable: false},
            };
            job.timer = setTimeout(() => poll(job), 500);
            const result = await request("/v1/sdapi/v1/txt2img", body);
            if (!Array.isArray(result.images)) throw new Error("Native generation returned no image list.");
            if (!result.images.length) {
                if (!job.stopping) throw new Error("Generation completed without an image.");
                message("Generation stopped.");
            } else {
                for (const data of result.images) showImage(data, job);
                progress.max = 1;
                progress.value = 1;
                message(job.stopping ? "The image completed before the stop request took effect." : "Generation complete. Download the PNG beneath the image.");
            }
            endJob(job);
        } catch (error) {
            if (job && error.transportFailed) {
                job.transportFailed = true;
                message("Connection to the generation request was lost. Waiting for the server to finish before allowing another request; reload only after checking the server.", true);
            } else {
                if (job) endJob(job);
                message(error.message || "Generation failed.", true);
            }
        }
    }
    async function cancel() {
        const job = active;
        if (!job || !job.sampling || job.stopping) return;
        job.stopping = true;
        updateControls();
        message("Stop requested. Waiting for the native generation call to return…");
        try { await request("/v1/sdapi/v1/interrupt", {id_task: job.id}); }
        catch (error) {
            if (active !== job) return;
            job.stopping = false;
            message(`Stop request failed: ${error.message}. The generation request remains active.`, true);
            updateControls();
        }
    }
    start.addEventListener("click", generate);
    stop.addEventListener("click", cancel);
    checkpoint.addEventListener("change", () => {
        try { localStorage.setItem("cosmo-native-checkpoint", checkpoint.value); } catch (_) {}
        updateControls();
    });
    backend.addEventListener("change", () => {
        updateControls();
        const selected = devices.get(backend.value);
        if (selected) message(`Selected ${selected.label}. This request uses ${selected.selector}.`);
    });
    refresh.addEventListener("click", () => loadModels(true).catch(error => message(error.message, true)));
    byId("clear-all-previews").addEventListener("click", () => preview.replaceChildren());
    window.addEventListener("beforeunload", event => { if (active) { event.preventDefault(); event.returnValue = ""; } });
    (async () => {
        try {
            const state = await window.CppKiosk.ready;
            if (state.capabilities.txt2img !== true || state.capabilities.max_concurrent_generations !== 1) throw new Error("Unsupported native generation protocol");
            const [result, config] = await Promise.all([
                request("/v1/sdapi/v1/backend-devices"), request("/v1/sdapi/v1/config"),
            ]);
            if (!Array.isArray(result.devices)) throw new Error("Invalid device list from the native server");
            const compute = config.effective?.compute;
            if (!compute || !["cpu", "webgpu"].includes(compute.backend)) throw new Error("Invalid effective compute configuration");
            for (const device of result.devices) {
                const kind = String(device.backend).toLowerCase();
                if (!["cpu", "webgpu"].includes(kind)) continue;
                if (typeof device.selector !== "string" || !device.selector || devices.has(device.selector))
                    throw new Error("Missing or duplicate native device selector");
                let label;
                if (kind === "cpu") label = "GGML CPU";
                else {
                    if (typeof device.software !== "boolean" || typeof device.provider !== "string")
                        throw new Error("WebGPU device classification is unavailable");
                    const type = device.software ? "Software CPU" : device.type === "integrated-gpu" ? "Integrated GPU" : device.type === "gpu" ? "Discrete GPU" : "Unknown adapter type";
                    label = `${type} · ${device.description || device.selector} · ${device.provider}`;
                    if (device.memory_known === true && Number(device.memory_total) > 0)
                        label += ` · ${(Number(device.memory_total) / 1073741824).toFixed(1)} GiB`;
                }
                label += ` [${device.selector}]`;
                devices.set(device.selector, {...device, kind, label});
                backend.add(new Option(label, device.selector));
            }
            if (!backend.options.length) throw new Error("No supported native image device is available");
            const requested = compute.device || "auto";
            const choices = [...devices.values()].filter(device => device.kind === compute.backend);
            let selected;
            if (requested === "auto") {
                selected = compute.backend === "cpu" ? choices[0] : choices.find(device => device.selector === result.default_webgpu_selector);
            } else {
                const matches = choices.filter(device => device.selector === requested ||
                    (device.stable_id_available === true && device.stable_id === requested));
                if (matches.length === 1) selected = matches[0];
            }
            if (selected) backend.value = selected.selector;
            else {
                const unavailable = new Option(`Configured device unavailable: ${requested}`, "");
                unavailable.disabled = true;
                backend.add(unavailable);
                backend.value = "";
            }
            await loadModels();
            ready = true;
            updateControls();
            message(selected ? "Ready for text-to-image generation using a complete checkpoint. Models load when you choose Generate." :
                "The configured device is unavailable. Choose an available compute device or update Settings; no replacement was selected.", !selected);
        } catch (error) {
            ready = false;
            updateControls();
            message(`${error.message}. Generation is disabled.`, true);
        }
    })();
})();
