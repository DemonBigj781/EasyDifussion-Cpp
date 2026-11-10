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
    let imageInputsSupported = false, maskInputsSupported = false;
    let ready = false, active = null, loadingModels = false, savingDefaults = false, models = new Set();
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
    byId("generation-companion-status").textContent = "Use a complete checkpoint with its text encoder and VAE. Separate components, LoRAs, ControlNet, generation plugins are not connected to this form.";
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
    const saveDefaults = document.createElement("button");
    saveDefaults.id = "cosmo-save-generation-defaults";
    saveDefaults.type = "button";
    saveDefaults.textContent = "Save generation defaults";
    saveDefaults.title = "Save this checkpoint and image recipe for the UI, shell and API. Device startup preferences are saved in Settings.";
    saveDefaults.disabled = true;
    refresh.after(saveDefaults);
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

    // Inputs stay browser-local until Generate. Never save image data in defaults.
    const imagePanel = document.createElement("fieldset");
    imagePanel.id = "cosmo-image-inputs";
    imagePanel.hidden = true;
    const imageLegend = document.createElement("legend");
    imageLegend.textContent = "Image-to-image / uploaded-mask inpainting";
    const imageHelp = document.createElement("p");
    imageHelp.textContent = "Optional PNG or JPEG initial image. Output uses the selected width and height. An optional mask uses white to repaint and black to preserve. Painting and automatic mask editors remain available in the Python UI.";
    imagePanel.append(imageLegend, imageHelp);
    const uploads = {};
    for (const kind of ["initial", "mask"]) {
        const label = document.createElement("label");
        label.textContent = kind === "initial" ? "Initial image " : "Inpainting mask ";
        const input = document.createElement("input");
        input.id = `cosmo-${kind}-file`; input.type = "file"; input.accept = "image/png,image/jpeg";
        const thumbnail = document.createElement("img");
        thumbnail.id = `cosmo-${kind}-preview`; thumbnail.alt = label.textContent.trim();
        thumbnail.hidden = true; thumbnail.style.maxWidth = "240px"; thumbnail.style.maxHeight = "180px";
        const clear = document.createElement("button");
        clear.id = `cosmo-${kind}-clear`; clear.type = "button"; clear.textContent = `Remove ${kind} image`;
        label.append(input); imagePanel.append(label, thumbnail, clear);
        const entry = uploads[kind] = {input, label, thumbnail, clear, data: null, revision: 0, loading: false};
        clear.addEventListener("click", () => { if (!active) clearUpload(kind); });
        input.addEventListener("change", () => loadUpload(kind, input.files?.[0]));
    }
    function numericControl(id, title, value, min, max, step) {
        const label = document.createElement("label"); label.textContent = title + " "; label.id = `${id}-label`;
        const input = document.createElement("input"); input.id = id; input.type = "number";
        input.value = value; input.min = min; input.max = max; input.step = step;
        label.append(input); imagePanel.append(label); return input;
    }
    const strength = numericControl("cosmo-image-strength", "Denoising strength", "0.75", "0", "1", "0.05");
    const maskBlur = numericControl("cosmo-mask-blur", "Mask blur (output pixels)", "4", "0", "64", "1");
    const invertLabel = document.createElement("label"); invertLabel.textContent = "Invert mask ";
    const invertMask = document.createElement("input"); invertMask.id = "cosmo-mask-invert"; invertMask.type = "checkbox";
    invertLabel.append(invertMask); imagePanel.append(invertLabel);
    controls.append(imagePanel);
    function clearUpload(kind) {
        const entry = uploads[kind]; ++entry.revision; entry.loading = false; entry.data = null;
        entry.input.value = ""; entry.thumbnail.removeAttribute("src"); entry.thumbnail.hidden = true;
        if (kind === "initial") clearUpload("mask");
        updateControls();
    }
    async function loadUpload(kind, file) {
        if (active || !imageInputsSupported || (kind === "mask" && !maskInputsSupported)) return;
        clearUpload(kind);
        if (!file) return;
        const entry = uploads[kind], revision = entry.revision;
        entry.loading = true; updateControls();
        try {
            if (!["image/png", "image/jpeg"].includes(file.type) || file.size > 16 * 1024 * 1024)
                throw new Error("Choose a PNG or JPEG no larger than 16 MiB.");
            const data = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = () => reject(new Error("Could not read the selected image."));
                reader.readAsDataURL(file);
            });
            const decoded = new Image();
            await new Promise((resolve, reject) => {
                decoded.onload = resolve; decoded.onerror = () => reject(new Error("The selected file is not a decodable image."));
                decoded.src = data;
            });
            if (!decoded.naturalWidth || !decoded.naturalHeight || decoded.naturalWidth * decoded.naturalHeight > 32 * 1024 * 1024)
                throw new Error("Image dimensions exceed the 32 megapixel input limit.");
            if (entry.revision !== revision) return;
            entry.data = data; entry.thumbnail.src = data; entry.thumbnail.hidden = false;
            message(`${kind === "initial" ? "Initial image" : "Mask"} ready (${decoded.naturalWidth} × ${decoded.naturalHeight}). Upload occurs only when you choose Generate.`);
        } catch (error) {
            if (entry.revision === revision) message(error.message, true);
        } finally {
            if (entry.revision === revision) { entry.loading = false; updateControls(); }
        }
    }

    function message(text, failed = false) {
        status.textContent = text;
        status.dataset.state = failed ? "error" : "";
    }
    function updateControls() {
        const readingImage = Object.values(uploads).some(entry => entry.loading);
        const available = ready && active === null && !loadingModels && !savingDefaults && !readingImage;
        for (const [kind, entry] of Object.entries(uploads)) {
            entry.input.disabled = !imageInputsSupported || !ready || active !== null || savingDefaults || (kind === "mask" && (!maskInputsSupported || !uploads.initial.data));
            entry.clear.disabled = !imageInputsSupported || !ready || active !== null || (!entry.data && !entry.loading);
        }
        strength.disabled = !imageInputsSupported || !available || !uploads.initial.data;
        maskBlur.disabled = invertMask.disabled = !maskInputsSupported || !available || !uploads.mask.data;
        for (const id of editable) byId(id).disabled = !available;
        checkpoint.disabled = !available || models.size === 0;
        backend.disabled = !available;
        refresh.disabled = !available;
        saveDefaults.disabled = !available || !models.has(checkpoint.value);
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
            const [result, options] = await Promise.all([
                request("/v1/sdapi/v1/checkpoints"), request("/v1/sdapi/v1/options"),
            ]);
            if (!Array.isArray(result.models) || result.models.some(model => typeof model.name !== "string")) throw new Error("Invalid checkpoint list from the native server");
            if (typeof options.sd_model_checkpoint !== "string") throw new Error("Invalid saved checkpoint from the native server");
            const previous = checkpoint.value;
            models = new Set(result.models.map(model => model.name));
            checkpoint.replaceChildren(new Option("Select a complete checkpoint", ""));
            for (const name of models) checkpoint.add(new Option(name, name));
            const saved = options.sd_model_checkpoint;
            if (models.has(saved)) checkpoint.value = saved;
            else if (!saved && models.has(previous)) checkpoint.value = previous;
            else if (!saved && models.size === 1) checkpoint.value = [...models][0];
            modelStatus.textContent = models.size ? `${models.size} checkpoint(s) indexed. The selected file must contain the components required by its architecture.` : "No checkpoints found. Start sdkit with --ckpt-dir pointing to your model directory, then refresh.";
            if (saved && !models.has(saved)) modelStatus.textContent = `Saved checkpoint ${saved} is unavailable in the model index. Select an available checkpoint or update its directory in Settings.`;
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
    function imageRecipe() {
        const width = integer("width", 64, 2048), height = integer("height", 64, 2048);
        if (width % 64 || height % 64) throw new Error("Width and height must be multiples of 64.");
        const cfg = Number(byId("guidance_scale").value);
        if (!Number.isFinite(cfg) || cfg < 0 || cfg > 30) throw new Error("Guidance must be between 0 and 30.");
        return {
            prompt: byId("prompt").value.trim(), negative_prompt: byId("negative_prompt").value,
            width, height, steps: integer("steps", 1, 200), cfg_scale: cfg,
            seed: integer("seed", -1, 2147483647), sampler_name: sampler.value, scheduler: "discrete",
        };
    }
    async function persistDefaults() {
        if (!ready || active || savingDefaults || loadingModels) return;
        try {
            if (!models.has(checkpoint.value)) throw new Error("Select an indexed checkpoint before saving defaults.");
            const recipe = imageRecipe();
            savingDefaults = true;
            updateControls();
            const result = await request("/v1/sdapi/v1/settings", {
                options: {sd_model_checkpoint: checkpoint.value}, inference: {image: recipe},
            });
            if (result.saved?.options?.sd_model_checkpoint !== checkpoint.value ||
                !result.saved?.inference?.image) throw new Error("The native server did not confirm saved generation defaults.");
            message("Checkpoint and generation defaults saved for this application. The UI, shell and API use the same configuration file.");
        } catch (error) { message(`Could not save generation defaults: ${error.message}`, true); }
        finally { savingDefaults = false; updateControls(); }
    }
    async function generate() {
        if (!ready || active || Object.values(uploads).some(entry => entry.loading)) return;
        let job;
        try {
            const prompt = byId("prompt").value.trim();
            if (!prompt) throw new Error("Enter a prompt first.");
            if (!models.has(checkpoint.value)) throw new Error("Select an indexed complete checkpoint.");
            if (!devices.has(backend.value)) throw new Error("Select an available compute device.");
            const recipe = imageRecipe();
            const imageInputs = {};
            if (uploads.initial.data) {
                const denoise = Number(strength.value);
                if (!Number.isFinite(denoise) || denoise < 0 || denoise > 1) throw new Error("Denoising strength must be between 0 and 1.");
                imageInputs.init_images = [uploads.initial.data.split(",")[1]];
                imageInputs.denoising_strength = denoise;
                if (uploads.mask.data) {
                    const blur = Number(maskBlur.value);
                    if (!Number.isFinite(blur) || blur < 0 || blur > 64) throw new Error("Mask blur must be between 0 and 64.");
                    imageInputs.mask = uploads.mask.data.split(",")[1];
                    imageInputs.mask_blur = blur;
                    imageInputs.inpainting_mask_invert = invertMask.checked ? 1 : 0;
                }
            }
            let seed = recipe.seed;
            if (seed === -1) seed = crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
            const id = `cosmo-${Date.now()}-${crypto.getRandomValues(new Uint32Array(1))[0]}`;
            job = {id, prompt, model: checkpoint.value, seed, steps: recipe.steps, backend: backend.value, sampling: false, stopping: false, transportFailed: false, timer: null};
            active = job;
            updateControls();
            progress.removeAttribute("value");
            message("Loading the checkpoint and preparing generation. Stop becomes available after the first sampling step completes.");
            const body = {
                ...recipe, ...imageInputs, force_task_id: id, seed, batch_size: 1, backend: job.backend,
                override_settings: {sd_model_checkpoint: job.model, forge_additional_modules: [], live_previews_enable: false},
            };
            job.timer = setTimeout(() => poll(job), 500);
            const result = await request(imageInputs.init_images ? "/v1/sdapi/v1/img2img" : "/v1/sdapi/v1/txt2img", body);
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
    checkpoint.addEventListener("change", updateControls);
    saveDefaults.addEventListener("click", persistDefaults);
    backend.addEventListener("change", () => {
        updateControls();
        const selected = devices.get(backend.value);
        if (selected) message(`Selected ${selected.label}. This request uses ${selected.selector}.`);
    });
    refresh.addEventListener("click", () => loadModels(true).catch(error => message(error.message, true)));
    byId("clear-all-previews").addEventListener("click", () => preview.replaceChildren());
    window.addEventListener("beforeunload", event => { if (active) { event.preventDefault(); event.returnValue = ""; } });
    updateControls();
    (async () => {
        try {
            const state = await window.CppKiosk.ready;
            if (state.capabilities.txt2img !== true || state.capabilities.max_concurrent_generations !== 1) throw new Error("Unsupported native generation protocol");
            imageInputsSupported = state.capabilities.img2img_ui === true;
            maskInputsSupported = imageInputsSupported && state.capabilities.uploaded_mask_inpainting_ui === true;
            uploads.mask.label.hidden = uploads.mask.clear.hidden = !maskInputsSupported;
            byId("cosmo-mask-blur-label").hidden = invertLabel.hidden = !maskInputsSupported;
            if (!maskInputsSupported) imageHelp.textContent = "Optional PNG or JPEG initial image. Output uses the selected width and height. Uploaded-mask inpainting is unavailable on this server.";
            imagePanel.hidden = !imageInputsSupported;
            const [result, config] = await Promise.all([
                request("/v1/sdapi/v1/backend-devices"), request("/v1/sdapi/v1/config"),
            ]);
            if (!Array.isArray(result.devices)) throw new Error("Invalid device list from the native server");
            const compute = config.effective?.compute;
            if (!compute || !["cpu", "webgpu"].includes(compute.backend)) throw new Error("Invalid effective compute configuration");
            const recipe = config.effective?.inference?.image;
            if (!recipe || recipe.scheduler !== "discrete") throw new Error("Invalid image defaults from the native configuration");
            const fields = {prompt: "prompt", negative_prompt: "negative_prompt", width: "width", height: "height",
                steps: "steps", cfg_scale: "guidance_scale", seed: "seed", sampler_name: "sampler"};
            for (const [key, id] of Object.entries(fields)) {
                if (recipe[key] === undefined) throw new Error(`Missing image default: ${key}`);
                byId(id).value = recipe[key];
            }
            imageRecipe();
            for (const id of ["width", "height"]) byId(`${id}-value`).textContent = `${byId(id).value} px`;
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
            message(selected ? "Ready to generate using a complete checkpoint. Optional image inputs appear when supported by this server. Models load when you choose Generate." :
                "The configured device is unavailable. Choose an available compute device or update Settings; no replacement was selected.", !selected);
        } catch (error) {
            ready = false;
            updateControls();
            message(`${error.message}. Generation is disabled.`, true);
        }
    })();
})();
