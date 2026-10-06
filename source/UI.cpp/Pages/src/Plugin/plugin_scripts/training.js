(() => {
    "use strict";
    const el = id => document.getElementById(`training-${id}`);
    if (!el("start") || el("start").dataset.initialized) return;
    el("start").dataset.initialized = "true";
    const storageKey = "easy-diffusion-cpp-training-v1";
    const fields = {
        dataset: "dataset", trigger: "trigger", kind: "kind", architecture: "architecture", backend: "backend",
        model: "model", qwen3: "qwen3", vae: "vae", output_name: "name",
        steps: "steps", resolution: "resolution", batch_size: "batch", learning_rate: "rate",
        text_encoder_learning_rate: "text-encoder-rate", rank: "rank", checkpointing: "checkpointing",
        network_alpha: "alpha", lr_scheduler: "scheduler", lr_warmup_steps: "warmup", lr_scheduler_num_cycles: "cycles",
        blocks_to_swap: "blocks-swap", optimizer_type: "optimizer", precision: "precision",
        save_every: "save-every", seed: "seed", epochs: "epochs", dataset_repeats: "repeats",
    };
    const numeric = new Set(["steps", "resolution", "batch_size", "learning_rate",
        "text_encoder_learning_rate", "rank", "network_alpha", "lr_warmup_steps", "lr_scheduler_num_cycles",
        "blocks_to_swap", "save_every", "seed", "epochs", "dataset_repeats"]);
    el("model").dataset.path = "";
    el("model").classList.add("model-filter");
    el("model").autocomplete = "off";
    const modelPicker = new ModelDropdown(el("model"), null, "Select a checkpoint");
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) {}
    if (!saved || typeof saved !== "object" || Array.isArray(saved)) saved = {};
    let readiness = null, active = true, submitting = false, selectedJob = null, timer;
    let spriteRunning = false;
    const error = exception => { el("error").textContent = exception.message || String(exception); };

    async function api(path, payload) {
        const response = await fetch(`/training${path}`, payload === undefined ? {cache: "no-store"} : {
            method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
        return data;
    }
    function persist() {
        const data = {};
        for (const [key, id] of Object.entries(fields)) data[key] = key === "model" ? modelPicker.value : el(id).value;
        data.cache_text_encoder_outputs = el("cache-text").checked;
        try { localStorage.setItem(storageKey, JSON.stringify(data)); } catch (_) {}
    }
    function controls() {
        const sd15 = el("architecture").value === "sd15";
        const anima = el("architecture").value === "anima";
        el("backend").querySelector('option[value="native"]').disabled = !sd15;
        if (!sd15) el("backend").value = "python";
        const native = el("backend").value === "native";
        if (native || anima) el("kind").value = "lora";
        el("kind").querySelector('option[value="embedding"]').disabled = native || anima;
        el("text-encoder-rate").disabled = !sd15 || el("kind").value !== "lora";
        el("batch").disabled = native;
        el("precision").disabled = native;
        el("precision").closest("label").hidden = native;
        for (const id of ["epochs", "repeats"]) { el(id).disabled = !native; el(id).closest("label").hidden = !native; }
        el("steps").disabled = native && el("epochs").value.trim() !== "";
        el("anima-settings").hidden = !anima;
        el("native-note").hidden = !native;
        el("checkpointing").disabled = native;
        el("alpha").disabled = el("kind").value !== "lora";
        const constant = el("scheduler").value === "constant";
        el("warmup").disabled = constant;
        el("cycles").disabled = constant;
        if (native) {
            el("optimizer").value = "AdamW";
            el("batch").value = "1";
            el("precision").value = "no";
            el("checkpointing").value = "off";
        }
        for (const id of ["qwen3", "vae", "blocks-swap", "optimizer", "cache-text"])
            el(id).disabled = !anima;
        const ready = native ? readiness?.native_cpp?.ready : readiness?.python_runtime?.ready;
        if (readiness) el("readiness").textContent = (native ? readiness.native_cpp?.detail : readiness.python_runtime?.detail) || readiness.detail;
        el("start").disabled = !ready || active || submitting || spriteRunning || !el("dataset").value || !modelPicker.value;
        el("cancel").disabled = !selectedJob || !["queued", "running"].includes(selectedJob.status);
        el("resume").disabled = active || submitting || !selectedJob || selectedJob.spec?.native_cpp ||
            selectedJob.kind !== "lora" || !["failed", "cancelled", "interrupted"].includes(selectedJob.status);
    }
    function request() {
        const data = {};
        for (const [key, id] of Object.entries(fields)) {
            const value = key === "model" ? modelPicker.value : el(id).value;
            if (key === "epochs" && (value.trim() === "" || el(id).disabled)) { data[key] = null; continue; }
            if (key === "dataset_repeats" && el(id).disabled) { data[key] = 1; continue; }
            if (key === "network_alpha" && (value.trim() === "" || el(id).disabled)) {
                data[key] = null;
                continue;
            }
            if (el("scheduler").value === "constant" && ["lr_warmup_steps", "lr_scheduler_num_cycles"].includes(key)) {
                data[key] = key === "lr_warmup_steps" ? 0 : 1;
                continue;
            }
            if (numeric.has(key) && (value.trim() === "" || !Number.isFinite(Number(value))))
                throw new Error(`${key} must be a finite number.`);
            data[key] = numeric.has(key) ? Number(value) : value;
        }
        if (el("text-encoder-rate").disabled) data.text_encoder_learning_rate = 0;
        data.cache_text_encoder_outputs = el("cache-text").checked;
        if (data.learning_rate <= 0 || data.learning_rate > .1 || data.text_encoder_learning_rate < 0 || data.text_encoder_learning_rate > .1)
            throw new Error("Learning rates must be at most 0.1; UNet must be positive and text encoder non-negative.");
        if (!Number.isInteger(data.steps) || data.steps < 1 || data.steps > 100000)
            throw new Error("Training steps must be an integer from 1 to 100000.");
        if (data.network_alpha !== null && (data.network_alpha <= 0 || data.network_alpha > 128))
            throw new Error("LoRA alpha must be positive and at most 128.");
        if (data.epochs !== null && (!Number.isInteger(data.epochs) || data.epochs < 1 || data.epochs > 100000))
            throw new Error("Epochs must be an integer from 1 to 100000.");
        if (!Number.isInteger(data.dataset_repeats) || data.dataset_repeats < 1 || data.dataset_repeats > 100000)
            throw new Error("Dataset repeats must be an integer from 1 to 100000.");
        if (!Number.isInteger(data.lr_warmup_steps) || data.lr_warmup_steps < 0 || (!data.epochs && data.lr_warmup_steps >= data.steps) ||
            !Number.isInteger(data.lr_scheduler_num_cycles) || data.lr_scheduler_num_cycles < 1 ||
            (!data.epochs && data.lr_scheduler_num_cycles > data.steps - data.lr_warmup_steps))
            throw new Error("Scheduler requires integer warmup below training steps and positive cycles no greater than post-warmup steps.");
        return data;
    }
    function fillOptions(id, options) {
        if (id === "model") {
            const previous = modelPicker.value || saved.model;
            modelPicker.inputModels = buildTree(options.map(item => ({model: item.value})));
            modelPicker.populateModels();
            if (options.some(item => item.value === previous)) modelPicker.value = previous;
            else if (options.length === 1) modelPicker.value = options[0].value;
            return;
        }
        const previous = el(id).value || saved[Object.keys(fields).find(key => fields[key] === id)];
        el(id).replaceChildren(new Option("Select…", ""), ...options.map(item => new Option(item.name, item.value)));
        if (options.some(item => item.value === previous)) el(id).value = previous;
        else if (options.length === 1) el(id).value = options[0].value;
    }
    async function jobs() {
        clearTimeout(timer);
        const data = await api("/jobs");
        active = data.jobs.some(job => ["queued", "running", "cancelling"].includes(job.status));
        const previous = el("job").value;
        el("job").replaceChildren(...data.jobs.map(job => new Option(`${job.kind}: ${job.status} — ${job.id.slice(0, 8)}`, job.id)));
        if (data.jobs.some(job => job.id === previous)) el("job").value = previous;
        selectedJob = el("job").value ? await api(`/jobs/${encodeURIComponent(el("job").value)}`) : null;
        if (selectedJob) {
            el("job-status").textContent = `${selectedJob.status} · ${selectedJob.progress?.step || 0} / ${selectedJob.progress?.total || selectedJob.spec?.steps || "?"} steps`;
            el("progress").max = selectedJob.progress?.total || 1;
            el("progress").value = selectedJob.progress?.step || 0;
            el("log").textContent = `${selectedJob.status}\n` +
                (selectedJob.log || []).map(event => event.message || event.error || JSON.stringify(event)).join("\n");
        } else {
            el("job-status").textContent = "No job selected.";
            el("progress").max = 1;
            el("progress").value = 0;
            el("log").textContent = "Select a job to view its log.";
        }
        controls();
        if (active) timer = setTimeout(() => jobs().catch(error), 1500);
    }
    async function refresh() {
        el("error").textContent = "";
        const loads = [
            api("/datasets").then(data => fillOptions("dataset", data.datasets.map(name => ({name, value: name})))),
            api("/models").then(data => fillOptions("model", data.models.map(item => ({name: item.name, value: item.path})))),
            api("/assets").then(data => {
                fillOptions("qwen3", data.text_encoders.map(item => ({name: item.name, value: item.path})));
                fillOptions("vae", data.vaes.map(item => ({name: item.name, value: item.path})));
            }),
            jobs(),
            api("/readiness").then(data => { readiness = data; el("readiness").textContent = data.detail; }),
            api("/spritegpt").then(data => {
                spriteRunning = Boolean(data.running);
                el("spritegpt-status").textContent = data.detail || (spriteRunning ? "Running" : "Stopped");
            }),
        ];
        await Promise.all(loads.map(load => load.catch(error).finally(controls)));
    }
    for (const [key, id] of Object.entries(fields)) {
        if (saved[key] !== undefined && el(id).tagName !== "SELECT") el(id).value = saved[key];
        if (saved[key] !== undefined && [...(el(id).options || [])].some(option => option.value === saved[key])) el(id).value = saved[key];
        el(id).addEventListener("change", () => { controls(); persist(); });
    }
    for (const id of ["rate", "text-encoder-rate"]) { el(id).step = "any"; el(id).min = "0"; el(id).max = "0.1"; }
    el("alpha").step = "any"; el("alpha").min = "0.00000001"; el("alpha").max = "128";
    el("warmup").min = "0"; el("cycles").min = "1";
    for (const id of ["epochs", "repeats"]) { el(id).min = "1"; el(id).max = "100000"; el(id).step = "1"; }
    if (typeof saved.cache_text_encoder_outputs === "boolean") el("cache-text").checked = saved.cache_text_encoder_outputs;
    el("cache-text").addEventListener("change", persist);
    el("refresh").addEventListener("click", () => refresh().catch(error));
    el("job").addEventListener("change", () => jobs().catch(error));
    el("start").addEventListener("click", async () => {
        if (submitting || el("start").disabled) return;
        submitting = true; controls(); el("error").textContent = "";
        try { const data = request(); persist(); await api("/jobs", data); await jobs(); }
        catch (exception) { error(exception); }
        finally { submitting = false; controls(); }
    });
    for (const action of ["cancel", "resume"]) el(action).addEventListener("click", async () => {
        try { await api(`/jobs/${encodeURIComponent(el("job").value)}/${action}`, {}); await jobs(); }
        catch (exception) { error(exception); }
    });
    el("spritegpt-start").addEventListener("click", () => api("/spritegpt/start", {}).then(refresh).catch(error));
    el("spritegpt-stop").addEventListener("click", () => api("/spritegpt/stop", {}).then(refresh).catch(error));
    el("spritegpt-open").addEventListener("click", async () => {
        try { const data = await api("/spritegpt"); if (data.url) window.open(data.url, "_blank", "noopener"); }
        catch (exception) { error(exception); }
    });
    controls();
    refresh().catch(error);
})();
