;(function () {
    "use strict"
    if (document.getElementById("tab-training")) return
    createTab({
        id: "training", label: "Training", icon: "fa-graduation-cap",
        content: window.loadRequiredPluginHTML("/plugins/core/training_plugin/training.tab.plugin.html"),
        css: `
            #training .training-panel { max-width: 1000px; margin: auto; text-align: left; }
            #training fieldset { margin: 20px 0; padding: 16px; border: 1px solid var(--background-color4); }
            #training .training-grid { display: grid; grid-template-columns: minmax(150px, 1fr) 2fr; gap: 12px; align-items: center; }
            #training input, #training select, #training textarea { width: 100%; box-sizing: border-box; }
            #training button { margin: 4px; }
            #training code, #training p { overflow-wrap: anywhere; }
            #training-log { height: 250px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; background: var(--background-color1); padding: 12px; }
            #training-error { color: #e56a6a; }
            #training-progress { width: 100%; }
            @media (max-width: 650px) { #training .training-grid { grid-template-columns: 1fr; } }
        `,
        onOpen: () => { refresh().catch(showError) },
    })
    const el = (id) => document.getElementById(`training-${id}`)
    let ready = false
    let active = false
    let items = []
    let scannedDataset = ""
    let selectedJob = ""
    let timer
    let refreshing = false

    function showError(error) { el("error").textContent = error.message || String(error) }
    async function api(path, payload) {
        const response = await fetch(`/training${path}`, payload === undefined ? { cache: "no-store" } : {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
        })
        const data = await response.json()
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data))
        return data
    }
    function controls() {
        el("start").disabled = !ready || active
        el("autotag").disabled = active
        el("save-caption").disabled = active || !items.length
    }
    function action(id, callback) {
        el(id).addEventListener("click", async () => {
            el("error").textContent = ""
            try { await callback() } catch (error) { showError(error) }
        })
    }
    async function refresh() {
        if (refreshing) return
        refreshing = true
        el("readiness").textContent = "Checking training runtime…"
        try {
            const results = await Promise.allSettled([
                api("/readiness"), api("/models"), refreshJobs(),
            ])
            if (results[0].status === "fulfilled") {
                const state = results[0].value
                ready = state.ready
                el("readiness").textContent = state.detail + (state.gpu ? ` — ${state.gpu}` : "")
                el("dataset-root").textContent = state.dataset_root
            } else {
                ready = false
                el("readiness").textContent = "Runtime check failed."
                showError(results[0].reason)
            }
            if (results[1].status === "fulfilled") {
                const current = el("model").value
                el("model").replaceChildren(new Option("Choose checkpoint", ""))
                results[1].value.models.forEach((model) => el("model").add(new Option(model.name, model.path)))
                el("model").value = current
            } else showError(results[1].reason)
            if (results[2].status === "rejected") showError(results[2].reason)
        } finally { refreshing = false; controls() }
    }
    async function scan() {
        const dataset = el("dataset").value.trim()
        const data = await api("/dataset/scan", { dataset })
        items = data.items
        scannedDataset = dataset
        el("image").replaceChildren()
        items.forEach((item, index) => el("image").add(new Option(item.image, String(index))))
        el("dataset-status").textContent = `${data.count} images; ${items.filter((item) => item.caption).length} non-empty captions.`
        selectImage()
        controls()
    }
    function selectImage() {
        el("caption").value = items[Number(el("image").value)]?.caption || ""
    }
    function displayJob(job) {
        const isActive = ["queued", "running", "cancelling"].includes(job.status)
        el("cancel").disabled = !isActive || job.status === "cancelling"
        el("resume").disabled = active || job.kind !== "lora" || !["failed", "cancelled", "interrupted"].includes(job.status)
        el("job-status").textContent = `${job.status}${job.output ? ` — Saved: ${job.output}. Refresh your model selectors to use it.` : ""}${job.error ? ` — ${job.error}` : ""}`
        el("progress").max = job.progress?.total || 1
        el("progress").value = job.progress?.step || 0
        el("log").textContent = job.log.map((event) => event.message || event.error || JSON.stringify(event)).join("\n")
        el("log").scrollTop = el("log").scrollHeight
    }
    async function refreshJobs() {
        clearTimeout(timer)
        const data = await api("/jobs")
        active = data.jobs.some((job) => ["queued", "running", "cancelling"].includes(job.status))
        el("job").replaceChildren()
        data.jobs.forEach((job) => el("job").add(new Option(`${job.kind}: ${job.status} — ${job.id.slice(0, 8)}`, job.id)))
        if (!data.jobs.some((job) => job.id === selectedJob)) selectedJob = data.jobs[0]?.id || ""
        el("job").value = selectedJob
        if (selectedJob) displayJob(await api(`/jobs/${selectedJob}`))
        controls()
        if (active) timer = setTimeout(() => refreshJobs().catch((error) => {
            showError(error)
            timer = setTimeout(() => refreshJobs().catch(showError), 5000)
        }), 1500)
    }
    async function acceptJob(job) {
        selectedJob = job.id
        active = true
        controls()
        await refreshJobs()
    }
    action("refresh", refresh)
    action("scan", scan)
    action("save-caption", async () => {
        const item = items[Number(el("image").value)]
        if (!item) throw new Error("Scan a dataset first")
        const result = await api("/dataset/caption", {
            dataset: scannedDataset, image: item.image, caption: el("caption").value, previous: item.caption,
        })
        item.caption = result.caption
        el("dataset-status").textContent = `Saved caption for ${item.image}`
    })
    action("autotag", async () => {
        await acceptJob(await api("/autotag", {
            dataset: el("dataset").value.trim(), trigger: el("trigger").value.trim(),
            tagger: el("tagger").value.trim(), threshold: Number(el("threshold").value),
            character_threshold: Number(el("character-threshold").value), exclude_tags: el("exclude").value,
        }))
    })
    action("cancel", async () => { await api(`/jobs/${selectedJob}/cancel`, {}); await refreshJobs() })
    action("resume", async () => acceptJob(await api(`/jobs/${selectedJob}/resume`, {})))
    el("image").addEventListener("change", selectImage)
    el("job").addEventListener("change", () => {
        selectedJob = el("job").value
        refreshJobs().catch(showError)
    })
    el("architecture").addEventListener("change", () => {
        el("resolution").value = el("architecture").value === "sdxl" ? "1024" : "512"
    })
    el("kind").addEventListener("change", () => {
        const embedding = el("kind").value === "embedding"
        el("rank").disabled = embedding
        el("vectors").disabled = !embedding
        el("init-word").disabled = !embedding
        el("rate").value = embedding ? "0.0005" : "0.0001"
    })
    el("form").addEventListener("submit", async (event) => {
        event.preventDefault()
        el("error").textContent = ""
        el("start").disabled = true
        try {
            await acceptJob(await api("/jobs", {
                dataset: el("dataset").value.trim(), kind: el("kind").value,
                architecture: el("architecture").value, model: el("model").value,
                output_name: el("name").value, trigger: el("trigger").value.trim(),
                init_word: el("init-word").value, precision: el("precision").value,
                steps: Number(el("steps").value), resolution: Number(el("resolution").value),
                batch_size: Number(el("batch").value), rank: Number(el("rank").value),
                vectors: Number(el("vectors").value), learning_rate: Number(el("rate").value),
                save_every: Number(el("save-every").value), seed: Number(el("seed").value),
            }))
        } catch (error) { showError(error) } finally { controls() }
    })
    controls()
})()
