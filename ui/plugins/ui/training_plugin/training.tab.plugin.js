;(function () {
    "use strict"
    const trainingTabExists = Boolean(document.getElementById("tab-training"))
    if (!trainingTabExists) createTab({
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
            #training .training-gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin: 12px 0; }
            #training .training-gallery-card { border: 1px solid var(--background-color4); padding: 8px; min-width: 0; }
            #training .training-gallery-card img, #training .training-gallery-card video { width: 100%; max-height: 180px; object-fit: contain; background: #111; }
            #training .training-gallery-card textarea { min-height: 50px; }
            @media (max-width: 650px) { #training .training-grid { grid-template-columns: 1fr; } }
        `,
        onOpen: () => { refresh().catch(showError) },
    })
    if (!document.getElementById("tab-grabber-gallery")) createTab({
        id: "grabber-gallery", label: "Grabber Gallery", icon: "fa-globe",
        content: `<div class="training-panel"><h2>Grabber Gallery</h2><p id="grabber-gallery-status" role="status" aria-live="polite">Loading websites…</p><fieldset><legend>Search images</legend><div class="training-grid"><label for="training-scrape-query">Search tags</label><div><input id="training-scrape-query" list="training-scrape-tag-suggestions" placeholder="e.g. blue_hair" maxlength="500" autocomplete="off"><datalist id="training-scrape-tag-suggestions"></datalist></div><label for="training-scrape-dataset">Gallery dataset</label><input id="training-scrape-dataset" placeholder="grabber" autocomplete="off"><label for="training-scrape-sources">Websites</label><select id="training-scrape-sources" multiple size="8"></select><label>Website selection</label><div><button id="grabber-gallery-sites-all" type="button" class="tertiaryButton">Select all</button><button id="grabber-gallery-sites-none" type="button" class="tertiaryButton">Clear selection</button></div><label for="training-scrape-limit">Maximum results per site</label><input id="training-scrape-limit" type="number" min="1" max="300" value="80"><label for="training-scrape-threshold">Near duplicate cutoff</label><input id="training-scrape-threshold" type="number" min="0.50" max="1" step="0.01" value="0.95"></div><p id="training-scrape-source-note">Loading available websites automatically…</p><button id="training-scrape-search" type="button" class="primaryButton">Search</button><small>Leave Gallery dataset blank to use “grabber”.</small></fieldset><fieldset class="grabber-trigger-tags"><label for="grabber-gallery-trigger-tags">Trigger tags</label><input id="grabber-gallery-trigger-tags" placeholder="Tags to add to images" autocomplete="off"><small>Use the tag button on an image card to add these tags.</small></fieldset><h3>Search results</h3><p><button id="grabber-gallery-select-all" type="button" class="tertiaryButton">Select all results</button><button id="grabber-gallery-download" type="button" class="tertiaryButton">Download selected</button></p><progress id="grabber-gallery-progress" hidden></progress><div id="grabber-gallery-results" class="training-gallery-grid"></div><fieldset><legend>Scrape attempts</legend><select id="training-scrape-attempts"></select><button id="training-scrape-refresh-attempts" type="button" class="tertiaryButton">Refresh attempts</button><button id="training-scrape-load-attempt" type="button" class="tertiaryButton">Open attempt</button><p id="training-scrape-status" role="status">Search results and downloads are grouped by attempt.</p></fieldset><div class="training-gallery-tabs"><button id="training-scrape-before-tab" type="button" class="tertiaryButton">Before filtering</button><button id="training-scrape-filtered-tab" type="button" class="tertiaryButton">Filtered and tagged</button><button id="grabber-gallery-refresh" type="button" class="tertiaryButton">Refresh gallery</button></div><div id="training-scrape-before" class="training-gallery-grid" hidden></div><div id="training-scrape-filtered" class="training-gallery-grid" hidden></div><p><button id="training-scrape-delete" type="button" class="tertiaryButton">Delete selected</button><button id="grabber-gallery-tag" type="button" class="tertiaryButton">Auto-tag gallery</button><button id="training-scrape-tag" type="button" class="tertiaryButton">Re-run tagging</button><button id="training-scrape-archive" type="button" class="tertiaryButton">Archive selected to dataset</button></p><small>Tagger settings are in Training. Edit each image’s tags in the gallery before archiving.</small></div>`,
        css: `#grabber-gallery .training-panel { max-width:1200px;margin:auto;text-align:left } #grabber-gallery .training-grid { display:grid;grid-template-columns:minmax(150px,1fr) 2fr;gap:10px;align-items:center } #grabber-gallery input,#grabber-gallery select,#grabber-gallery textarea { width:100%;box-sizing:border-box } #grabber-gallery button { margin:4px } #grabber-gallery-results,#grabber-gallery .training-gallery-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin:12px 0 } #grabber-gallery .training-gallery-card { border:1px solid var(--background-color4);padding:10px;min-width:0;display:flex;flex-direction:column;gap:8px } #grabber-gallery .training-gallery-card img { width:100%;height:220px;object-fit:contain;background:#111 } #grabber-gallery .training-gallery-card textarea { min-height:64px } #grabber-gallery .grabber-trigger-tags { display:flex;flex-direction:column;gap:8px } #training-scrape-before,#training-scrape-filtered { display:inline-grid;width:calc(50% - 12px);vertical-align:top;align-content:start } #training-scrape-before::before,#training-scrape-filtered::before { grid-column:1/-1;font-weight:bold;margin:4px 0 } #training-scrape-before::before { content:'Before filter' } #training-scrape-filtered::before { content:'Filtered and tagged' } #training-scrape-before-tab,#training-scrape-filtered-tab { display:none } @media(max-width:700px){#grabber-gallery .training-grid{grid-template-columns:1fr}#training-scrape-before,#training-scrape-filtered{display:grid;width:100%}}`,
    })
    const trainingTabElement = document.getElementById("tab-training")
    if (trainingTabElement?.dataset.trainingUiInitialized === "true") return
    if (trainingTabElement) trainingTabElement.dataset.trainingUiInitialized = "true"
    const el = (id) => document.getElementById(`training-${id}`)
    const galleryEl = (id) => document.getElementById(`grabber-gallery-${id}`)
    const beforeGallery = el("scrape-before")
    const filteredGallery = el("scrape-filtered")
    const motionGallery = document.createElement("div")
    motionGallery.id = "grabber-gallery-frames"
    motionGallery.className = "training-gallery-grid"
    const sourceTabs = document.createElement("div")
    sourceTabs.className = "training-gallery-tabs"
    const stillsTab = document.createElement("button")
    stillsTab.type = "button"
    stillsTab.className = "tertiaryButton"
    stillsTab.textContent = "Still images"
    const framesTab = document.createElement("button")
    framesTab.type = "button"
    framesTab.className = "tertiaryButton"
    framesTab.textContent = "GIF/video frames"
    sourceTabs.append(stillsTab, framesTab)
    const selectedGallery = document.createElement("div")
    selectedGallery.id = "grabber-gallery-selected"
    selectedGallery.className = "training-gallery-grid"
    const parentGallery = beforeGallery.parentNode
    const galleryColumns = document.createElement("div")
    galleryColumns.className = "grabber-gallery-columns"
    Object.assign(galleryColumns.style, {
        display: "none", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "16px", width: "100%",
    })
    beforeGallery.hidden = true
    beforeGallery.style.display = "none"
    filteredGallery.style.display = "grid"
    filteredGallery.style.width = "100%"
    selectedGallery.style.display = "grid"
    selectedGallery.style.width = "100%"
    parentGallery.insertBefore(galleryColumns, beforeGallery)
    parentGallery.insertBefore(beforeGallery, galleryColumns.nextSibling)
    const filteredPane = document.createElement("section")
    const filteredHeading = document.createElement("h3")
    filteredHeading.textContent = "Processed images"
    filteredPane.append(filteredHeading, sourceTabs, filteredGallery, motionGallery)
    const selectedPane = document.createElement("section")
    const selectedHeading = document.createElement("h3")
    selectedHeading.textContent = "Selected for dataset"
    selectedPane.append(selectedHeading, selectedGallery)
    galleryColumns.append(filteredPane, selectedPane)
    const triggerFieldset = document.querySelector("#grabber-gallery .grabber-trigger-tags")
    triggerFieldset.querySelector("label").textContent = "LoRA trigger tag"
    triggerFieldset.querySelector("input").placeholder = "Optional trigger tag for this dataset"
    triggerFieldset.querySelector("small").remove()
    const taggerSettings = document.createElement("fieldset")
    taggerSettings.innerHTML = `<legend>Auto-tag settings</legend><div class="training-grid"><label for="training-tagger">WD14 model</label><input id="training-tagger" value="wd-v1-4-moat-tagger-v2"><label for="training-threshold">Tag threshold</label><input id="training-threshold" type="number" min="0" max="1" step="0.01" value="0.35"><label for="training-character-threshold">Character threshold</label><input id="training-character-threshold" type="number" min="0" max="1" step="0.01" value="0.85"><label for="training-exclude">Exclude tags (comma separated)</label><input id="training-exclude"></div>`
    const galleryActions = el("scrape-archive").parentNode
    galleryActions.parentNode.insertBefore(taggerSettings, galleryActions)
    galleryActions.parentNode.insertBefore(triggerFieldset, galleryActions)
    el("scrape-archive").textContent = "Submit selected to training dataset"
    el("scrape-search").textContent = "Search and download"
    let ready = false
    const modelPicker = new ModelDropdown(el("model"), null, "Select a checkpoint")
    let nativeReady = false
    let active = false
    let selectedJob = ""
    let timer
    let refreshing = false
    let spriteRunning = false
    let scrapeAttempt = ""
    let scrapeResults = []
    let scrapeGallery = null
    let scrapeDataset = ""
    let archivedDatasetView = false
    let scrapeTagJobId = ""
    let selectedScrapeNames = new Set()
    let activeSourceGallery = "filtered"
    let scrapeSuggestTimer
    let scrapeSuggestSequence = 0

    function showError(error) { el("error").textContent = error.message || String(error) }
    function scrapeDatasetName() { return el("scrape-dataset").value.trim() || scrapeDataset || "grabber" }
    async function api(path, payload) {
        const response = await fetch(`/training${path}`, payload === undefined ? { cache: "no-store" } : {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
        })
        const data = await response.json()
        if (!response.ok) {
            if (Array.isArray(data.detail)) {
                const issue = data.detail[0]
                const field = issue?.loc?.at(-1)
                const labels = { dataset: "Dataset folder", query: "Search terms", sources: "Websites", limit: "Maximum results" }
                throw new Error(`${labels[field] || field || "Request"}: ${issue?.msg || "invalid value"}`)
            }
            throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data))
        }
        return data
    }
    const datasetSuggestions = document.createElement("datalist")
    datasetSuggestions.id = "training-dataset-suggestions"
    el("scrape-dataset").setAttribute("list", datasetSuggestions.id)
    el("scrape-dataset").parentNode.append(datasetSuggestions)
    async function loadDatasetSuggestions(preferred = el("dataset").value) {
        const data = await api("/datasets")
        datasetSuggestions.replaceChildren(...data.datasets.map((name) => new Option(name, name)))
        const dataset = el("dataset")
        const placeholder = data.datasets.length
            ? new Option("Choose a training dataset", "")
            : new Option("No prepared datasets — submit from Grabber Gallery", "")
        dataset.replaceChildren(placeholder, ...data.datasets.map((name) => new Option(name, name)))
        dataset.value = data.datasets.includes(preferred) ? preferred : (data.datasets[0] || "")
        el("dataset-status").textContent = data.datasets.length
            ? `${data.datasets.length} prepared dataset${data.datasets.length === 1 ? "" : "s"} loaded.`
            : "No prepared datasets found. Submit tagged images from Grabber Gallery first."
    }
    function loadAssetOptions(select, assets, placeholder, preferredPattern) {
        const current = select.value
        select.replaceChildren(new Option(placeholder, ""), ...assets.map((asset) => new Option(asset.name, asset.path)))
        if (assets.some((asset) => asset.path === current)) {
            select.value = current
            return
        }
        const preferred = assets.find((asset) => preferredPattern.test(asset.name) && asset.name.endsWith("(folder)"))
            || assets.find((asset) => preferredPattern.test(asset.name))
            || assets[0]
        select.value = preferred?.path || ""
    }
    function controls() {
        const isAnima = el("architecture").value === "anima"
        const isSd15 = el("architecture").value === "sd15"
        el("backend").querySelector('option[value="native"]').disabled = !isSd15
        if (!isSd15) el("backend").value = "python"
        const nativeSd15 = el("backend").value === "native"
        if ((isAnima || nativeSd15) && el("kind").value === "embedding") el("kind").value = "lora"
        if (nativeSd15) el("optimizer").value = "AdamW"
        el("text-encoder-rate").disabled = !isSd15 || el("kind").value !== "lora"
        document.querySelectorAll(".native-training-option").forEach(field => { field.hidden = !nativeSd15; if (field.matches("input")) field.disabled = !nativeSd15 })
        el("steps").disabled = nativeSd15 && el("epochs").value.trim() !== ""
        el("alpha").disabled = el("kind").value !== "lora"
        const constant = el("scheduler").value === "constant"
        el("warmup").disabled = constant
        el("cycles").disabled = constant
        el("start").disabled = !(nativeSd15 ? nativeReady : ready) || active || spriteRunning || !el("dataset").value
        document.querySelectorAll(".scrape-image-tag").forEach((button) => { button.disabled = active || spriteRunning })
        document.querySelectorAll(".anima-training-option").forEach((field) => {
            field.hidden = !isAnima
            const control = field.matches("input, select, textarea") ? field : null
            if (control) control.disabled = !isAnima
        })
        el("kind").querySelector('option[value="embedding"]').disabled = isAnima || nativeSd15
        el("qwen3").required = isAnima
        el("vae").required = isAnima
        if (nativeSd15) el("batch").value = "1"
        el("batch").disabled = nativeSd15
        el("precision").disabled = nativeSd15
    }
    function action(id, callback) {
        el(id).addEventListener("click", async () => {
            el("error").textContent = ""
            try { await callback() } catch (error) {
                showError(error)
                if (id.startsWith("scrape-")) galleryEl("status").textContent = error.message || String(error)
            }
        })
    }
    async function refresh() {
        if (refreshing) return
        refreshing = true
        el("readiness").textContent = "Checking training runtime…"
        loadDatasetSuggestions().catch(showError)
        try {
            const results = await Promise.allSettled([
                api("/readiness"), api("/models"), refreshJobs(), api("/assets"),
            ])
            await refreshSpriteGPT()
            if (results[0].status === "fulfilled") {
                const state = results[0].value
                ready = Boolean(state.python_runtime?.ready)
                nativeReady = Boolean(state.native_cpp?.ready)
                el("readiness").textContent = state.detail + (state.gpu ? ` — ${state.gpu}` : "")
                el("dataset-root").textContent = state.dataset_root
            } else {
                ready = false
                nativeReady = false
                el("readiness").textContent = "Runtime check failed."
                showError(results[0].reason)
            }
            if (results[1].status === "fulfilled") {
                const current = modelPicker.value
                modelPicker.inputModels = buildTree(results[1].value.models.map(model => ({model: model.path})))
                modelPicker.populateModels()
                modelPicker.value = current || ""
            } else showError(results[1].reason)
            if (results[2].status === "rejected") showError(results[2].reason)
            if (results[3].status === "fulfilled") {
                loadAssetOptions(el("qwen3"), results[3].value.text_encoders, "No configured text encoders found", /qwen.?3/i)
                loadAssetOptions(el("vae"), results[3].value.vaes, "No configured VAE files found", /qwen.?image/i)
            } else showError(results[3].reason)
        } finally { refreshing = false; controls() }
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
        if (scrapeTagJobId && !active) {
            const tagJob = data.jobs.find((job) => job.id === scrapeTagJobId)
            if (tagJob) {
                const detail = await api(`/jobs/${scrapeTagJobId}`)
                await loadScrapeGallery()
                galleryEl("status").textContent = detail.status === "completed"
                    ? "Image auto-tagged. Review its tags below the preview."
                    : `Image auto-tagging ${detail.status}${detail.error ? `: ${detail.error}` : ""}`
                scrapeTagJobId = ""
            }
        }
        controls()
        if (active) timer = setTimeout(() => refreshJobs().catch((error) => {
            showError(error)
            timer = setTimeout(() => refreshJobs().catch(showError), 5000)
        }), 1500)
    }
    async function refreshSpriteGPT() {
        const state = await api("/spritegpt")
        spriteRunning = state.running
        el("spritegpt-status").textContent = state.detail + (state.running ? ` — ${state.url}` : "")
        el("spritegpt-open").disabled = !state.running
        el("spritegpt-start").disabled = state.running || !state.available
        el("spritegpt-stop").disabled = !state.running
        controls()
    }
    action("spritegpt-start", async () => {
        el("spritegpt-status").textContent = "Starting SpriteGPT runtime…"
        const state = await api("/spritegpt/start", {})
        spriteRunning = state.running
        el("spritegpt-status").textContent = state.detail + (state.url ? ` — ${state.url}` : "")
        if (!state.running) throw new Error("SpriteGPT did not start. Check training/logs/spritegpt-runtime.log")
        controls()
    })
    action("spritegpt-stop", async () => {
        const state = await api("/spritegpt/stop", {})
        spriteRunning = state.running
        el("spritegpt-status").textContent = state.detail
        controls()
    })
    action("spritegpt-open", async () => {
        const state = await api("/spritegpt")
        if (!state.running || !state.url) throw new Error(state.detail)
        window.open(state.url, "_blank", "noopener")
    })
    function selectedFrom(container) {
        return [...container.querySelectorAll('input[type="checkbox"]:checked')].map((item) => item.value)
    }
    function selectedScrapePayload() {
        const names = []
        const frame_names = []
        for (const key of selectedScrapeNames) {
            const separator = key.indexOf("::")
            const kind = key.slice(0, separator)
            const name = key.slice(separator + 2)
            ;(kind === "frames" ? frame_names : names).push(name)
        }
        return { names, frame_names }
    }
    function showSourceGallery(kind) {
        activeSourceGallery = kind === "frames" ? "frames" : "filtered"
        filteredGallery.hidden = activeSourceGallery !== "filtered"
        filteredGallery.style.display = activeSourceGallery === "filtered" ? "grid" : "none"
        motionGallery.hidden = activeSourceGallery !== "frames"
        motionGallery.style.display = activeSourceGallery === "frames" ? "grid" : "none"
    }
    function showScrapeTab(tab) {
        const comparing = tab !== "results"
        galleryEl("results").hidden = comparing
        galleryEl("results").style.display = comparing ? "none" : "grid"
        galleryColumns.style.display = comparing ? "grid" : "none"
        el("scrape-before").hidden = true
        showSourceGallery(activeSourceGallery)
        if (!comparing) {
            filteredGallery.hidden = true
            motionGallery.hidden = true
        }
        selectedGallery.hidden = !comparing
        selectedGallery.style.display = comparing ? "grid" : "none"
    }
    function renderScrapeResults() {
        const root = galleryEl("results")
        root.replaceChildren()
        galleryEl("status").textContent = `${scrapeResults.length} preview results · attempt ${scrapeAttempt}`
        showScrapeTab("results")
        galleryEl("select-all").hidden = true
        galleryEl("download").hidden = true
        if (!scrapeResults.length) {
            root.textContent = "No previewable search results were returned. Try different terms or websites."
            return
        }
        for (const result of scrapeResults) {
            const card = document.createElement("div")
            card.className = "training-gallery-card"
            const image = document.createElement("img")
            image.src = result.preview_url || result.url
            image.alt = result.name || "Search result"
            image.loading = "lazy"
            image.referrerPolicy = "no-referrer"
            image.addEventListener("error", () => {
                const unavailable = document.createElement("span")
                unavailable.className = "grabber-preview-unavailable"
                unavailable.textContent = "Preview unavailable from this site"
                image.replaceWith(unavailable)
            }, { once: true })
            const text = document.createElement("span")
            text.textContent = `${result.source} · ${result.name || result.md5 || result.key}`
            card.append(image, text)
            if (result.page_url) {
                const link = document.createElement("a")
                link.href = result.page_url
                link.target = "_blank"
                link.rel = "noreferrer"
                link.textContent = "Open source page"
                card.append(link)
            }
            root.append(card)
        }
    }
    function renderGallery() {
        if (!scrapeGallery) return
        el("scrape-status").textContent = archivedDatasetView
            ? `${scrapeGallery.filtered.length} images in ${scrapeGallery.attempt}`
            : `Attempt ${scrapeGallery.attempt} · ${scrapeGallery.sources.join(", ")} · near-duplicate cutoff ${scrapeGallery.threshold}`
        ;["scrape-delete", "scrape-tag", "scrape-archive"].forEach((id) => { el(id).hidden = archivedDatasetView })
        galleryEl("tag").hidden = archivedDatasetView
        const before = el("scrape-before")
        before.replaceChildren()
        for (const item of scrapeGallery.before) {
            const card = document.createElement("div")
            card.className = "training-gallery-card"
            const ext = item.preview.toLowerCase()
            let media
            if (/\.(mp4|webm|mov|mkv|avi)(\?|$)/.test(ext)) {
                media = document.createElement("video")
                media.controls = true
                media.src = item.preview
            } else {
                media = document.createElement("img")
                media.src = item.preview
                media.alt = item.name || "Original scraped file"
                media.loading = "lazy"
            }
            const text = document.createElement("span")
            text.textContent = `${item.source} · ${item.name || item.key}`
            card.append(media, text)
            if (item.page_url) {
                const link = document.createElement("a")
                link.href = item.page_url
                link.target = "_blank"
                link.rel = "noreferrer"
                link.textContent = "Open source page"
                card.append(link)
            }
            before.append(card)
        }
        const filtered = el("scrape-filtered")
        filtered.replaceChildren()
        motionGallery.replaceChildren()
        selectedGallery.replaceChildren()
        const galleryItems = [...scrapeGallery.filtered, ...(scrapeGallery.frames || [])]
        const availableNames = new Set(galleryItems.map((item) => `${item.kind || "filtered"}::${item.name}`))
        selectedScrapeNames = new Set([...selectedScrapeNames].filter((name) => availableNames.has(name)))
        for (const item of galleryItems) {
            const kind = item.kind || "filtered"
            const selectionKey = `${kind}::${item.name}`
            const isSelected = selectedScrapeNames.has(selectionKey)
            const sourceGallery = kind === "frames" ? motionGallery : filtered
            const card = document.createElement("div")
            card.className = "training-gallery-card"
            const applyTags = document.createElement("button")
            applyTags.type = "button"
            applyTags.className = "tertiaryButton"
            applyTags.textContent = "Auto-tag"
            applyTags.hidden = archivedDatasetView
            applyTags.classList.add("scrape-image-tag")
            applyTags.addEventListener("click", async () => {
                try {
                    if (active) throw new Error("Wait for the current training job to finish")
                    const job = await api("/scrape/tag-image", { dataset: scrapeDatasetName(), attempt: scrapeAttempt,
                        name: item.name, kind, tagger: el("tagger").value.trim(),
                        threshold: Number(el("threshold").value),
                        character_threshold: Number(el("character-threshold").value), exclude_tags: el("exclude").value })
                    scrapeTagJobId = job.id
                    await acceptJob(job)
                    galleryEl("status").textContent = `Auto-tagging ${item.name}…`
                } catch (error) { showError(error) }
            })
            const checkbox = document.createElement("input")
            checkbox.type = "checkbox"
            checkbox.value = selectionKey
            checkbox.style.width = "auto"
            checkbox.checked = isSelected
            checkbox.setAttribute("aria-label", `Select ${item.name} for dataset`)
            checkbox.addEventListener("change", () => {
                if (checkbox.checked) selectedScrapeNames.add(selectionKey)
                else selectedScrapeNames.delete(selectionKey)
                ;(checkbox.checked ? selectedGallery : sourceGallery).append(card)
            })
            const image = document.createElement("img")
            image.src = item.preview
            image.alt = item.name
            image.loading = "lazy"
            const name = document.createElement("small")
            name.textContent = item.name
            const caption = document.createElement("textarea")
            caption.value = item.caption
            caption.setAttribute("aria-label", `Tags or caption for ${item.name}`)
            caption.readOnly = archivedDatasetView
            caption.addEventListener("change", async () => {
                try {
                    await api("/scrape/caption", { dataset: scrapeDatasetName(), attempt: scrapeAttempt,
                        name: item.name, caption: caption.value, kind })
                    el("scrape-status").textContent = `Saved tags for ${item.name}`
                } catch (error) { showError(error) }
            })
            const selectLabel = document.createElement("label")
            selectLabel.append(checkbox, document.createTextNode(" Select for dataset"))
            selectLabel.hidden = archivedDatasetView
            card.append(applyTags, image, name, caption, selectLabel)
            ;(isSelected ? selectedGallery : sourceGallery).append(card)
        }
        showSourceGallery(activeSourceGallery)
    }
    async function loadScrapeGallery() {
        if (!scrapeAttempt) return
        scrapeGallery = await api("/scrape/gallery", { dataset: scrapeDatasetName(), attempt: scrapeAttempt })
        renderGallery()
    }
    async function loadScrapeSites() {
        const data = await api("/scrape/sources")
        el("scrape-sources").size = Math.max(6, Math.min(20, data.sources.length))
        el("scrape-sources").replaceChildren(...data.sources.map((site) => new Option(site, site)))
        el("scrape-source-note").textContent = `${data.sources.length} websites ready. Select sites to limit the search; clear selection to search all.`
        galleryEl("status").textContent = `${data.sources.length} websites loaded. Ready to search.`
    }
    el("scrape-query").addEventListener("input", () => {
        clearTimeout(scrapeSuggestTimer)
        const query = el("scrape-query").value
        const token = query.split(/[\s,]+/).pop().trim()
        const sequence = ++scrapeSuggestSequence
        const list = el("scrape-tag-suggestions")
        if (token.length < 2) {
            list.replaceChildren()
            return
        }
        scrapeSuggestTimer = setTimeout(async () => {
            try {
                const sources = [...el("scrape-sources").selectedOptions].map((option) => option.value)
                const data = await api("/scrape/tag-suggestions", { query: token, sources })
                if (sequence !== scrapeSuggestSequence) return
                const prefix = query.slice(0, query.length - token.length)
                list.replaceChildren(...data.suggestions.map((tag) => {
                    const option = document.createElement("option")
                    option.value = `${prefix}${tag}`
                    return option
                }))
            } catch (_) {
                if (sequence === scrapeSuggestSequence) list.replaceChildren()
            }
        }, 500)
    })
    async function refreshScrapeAttempts() {
        const data = await api("/scrape/attempts", { dataset: scrapeDatasetName() })
        const archived = await api("/scrape/archived-datasets")
        const select = el("scrape-attempts")
        const old = archivedDatasetView ? `dataset:${scrapeDatasetName()}` : scrapeAttempt
        select.replaceChildren()
        const attemptsGroup = document.createElement("optgroup")
        attemptsGroup.label = "Scrape attempts"
        data.attempts.forEach((item) => attemptsGroup.append(new Option(
            `${item.attempt} · ${item.status} · ${item.images} images · ${item.query}`, item.attempt)))
        select.append(attemptsGroup)
        const datasetsGroup = document.createElement("optgroup")
        datasetsGroup.label = "Archived datasets"
        archived.datasets.forEach((item) => datasetsGroup.append(new Option(
            `${item.dataset} · ${item.images} images`, `dataset:${item.dataset}`)))
        select.append(datasetsGroup)
        if (old && [...select.options].some((option) => option.value === old)) select.value = old
    }
    action("scrape-refresh-attempts", refreshScrapeAttempts)
    action("scrape-load-attempt", async () => {
        if (!el("scrape-attempts").options.length) await refreshScrapeAttempts()
        const nextAttempt = el("scrape-attempts").value
        if (nextAttempt.startsWith("dataset:")) {
            const dataset = nextAttempt.slice("dataset:".length)
            if (!dataset) throw new Error("Choose an archived dataset")
            archivedDatasetView = true
            scrapeDataset = dataset
            el("scrape-dataset").value = dataset
            scrapeAttempt = ""
            selectedScrapeNames.clear()
            scrapeGallery = await api("/dataset/gallery", { dataset })
            renderGallery()
            showScrapeTab("filtered")
            return
        }
        archivedDatasetView = false
        if (scrapeAttempt !== nextAttempt) selectedScrapeNames.clear()
        scrapeAttempt = nextAttempt
        if (!scrapeAttempt) throw new Error("Choose a saved scrape attempt")
        await loadScrapeGallery()
        showScrapeTab("filtered")
    })
    action("scrape-search", async () => {
        galleryEl("status").textContent = "Searching Grabber…"
        const sources = [...el("scrape-sources").selectedOptions].map((option) => option.value)
        try {
            const dataset = el("scrape-dataset").value.trim() || "grabber"
            el("scrape-dataset").value = dataset
            const query = el("scrape-query").value.trim()
            if (!query) throw new Error("Enter search terms")
            scrapeDataset = dataset
            const data = await api("/scrape/search", { dataset,
                query, sources, limit: Number(el("scrape-limit").value) })
            scrapeAttempt = data.attempt
            archivedDatasetView = false
            ;["scrape-delete", "scrape-tag", "scrape-archive"].forEach((id) => { el(id).hidden = false })
            galleryEl("tag").hidden = false
            selectedScrapeNames.clear()
            scrapeResults = data.results
            renderScrapeResults()
            const rejected = Math.max(0, Number(data.reported_count || 0) - data.count)
            scrapeGallery = null
            await loadDatasetSuggestions()
            if (data.results.length) {
                await downloadScrapeResults(data.results.map((result) => result.key))
                galleryEl("status").textContent = `${data.count} search results · ${rejected} unusable URL${rejected === 1 ? "" : "s"} · downloaded and processed automatically · attempt ${scrapeAttempt}`
            } else {
                galleryEl("status").textContent = `No usable previews in ${data.reported_count} search results. Try different terms or sites.`
            }
            await refreshScrapeAttempts()
            document.getElementById("tab-grabber-gallery").click()
        } catch (error) {
            galleryEl("status").textContent = `Search failed: ${error.message || String(error)}`
            document.getElementById("tab-grabber-gallery").click()
            throw error
        }
    })
    galleryEl("select-all").addEventListener("click", () => {
        const boxes = [...galleryEl("results").querySelectorAll('input[type="checkbox"]')]
        const select = boxes.some((item) => !item.checked)
        boxes.forEach((item) => { item.checked = select })
    })
    galleryEl("sites-all").addEventListener("click", () => {
        [...el("scrape-sources").options].forEach((option) => { option.selected = true })
    })
    galleryEl("sites-none").addEventListener("click", () => {
        [...el("scrape-sources").options].forEach((option) => { option.selected = false })
    })
    galleryEl("refresh").addEventListener("click", async () => {
        try {
            if (!scrapeAttempt) await refreshScrapeAttempts()
            if (el("scrape-attempts").value) {
                scrapeAttempt = el("scrape-attempts").value
                await loadScrapeGallery()
                showScrapeTab("filtered")
            } else galleryEl("status").textContent = "No saved scrape attempts in this gallery yet."
        } catch (error) { galleryEl("status").textContent = `Gallery refresh failed: ${error.message || String(error)}` }
    })
    galleryEl("tag").addEventListener("click", () => el("scrape-tag").click())
    async function downloadScrapeResults(keys) {
        const button = galleryEl("download")
        try {
            if (!scrapeAttempt) throw new Error("Search first to create a scrape attempt")
            if (!keys.length) throw new Error("No search results to download")
            button.disabled = true
            button.textContent = "Downloading…"
            galleryEl("progress").hidden = false
            showScrapeTab("results")
            galleryEl("status").textContent = `Downloading all ${keys.length} search results; large images and videos can take a while.`
            const result = await api("/scrape/download", { dataset: scrapeDatasetName(),
                attempt: scrapeAttempt, keys, threshold: Number(el("scrape-threshold").value) })
            const saved = result.results.reduce((count, item) => count + (item.files?.length || 0), 0)
            const duplicate = result.results.filter((item) => ["duplicate", "near_duplicate"].includes(item.status)).length
            const failed = result.results.filter((item) => item.status === "failed").length
            const limit = result.results.some((item) => item.status === "limit_reached")
            const failureDetails = result.results.filter((item) => item.status === "failed").slice(0, 3)
                .map((item) => item.message).filter(Boolean).join("; ")
            galleryEl("status").textContent = `Download finished: ${saved} PNG${saved === 1 ? "" : "s"} saved, ${duplicate} duplicate result${duplicate === 1 ? "" : "s"} skipped, ${failed} failed${limit ? "; gallery image limit reached" : ""}.${failureDetails ? ` Errors: ${failureDetails}` : ""}`
            el("scrape-status").textContent = `${saved} PNGs saved · ${duplicate} duplicates skipped · ${failed} failed · cutoff ${result.threshold}`
            await loadScrapeGallery()
            showScrapeTab("filtered")
        } catch (error) {
            galleryEl("status").textContent = `Download failed: ${error.message || String(error)}`
            throw error
        } finally {
            button.disabled = false
            button.textContent = "Download selected"
            galleryEl("progress").hidden = true
        }
    }
    galleryEl("download").addEventListener("click", async () => {
        try {
            const keys = selectedFrom(galleryEl("results"))
            await downloadScrapeResults(keys)
        } catch (error) { showError(error) }
    })
    action("scrape-before-tab", async () => showScrapeTab("before"))
    action("scrape-filtered-tab", async () => showScrapeTab("filtered"))
    stillsTab.addEventListener("click", () => showSourceGallery("filtered"))
    framesTab.addEventListener("click", () => showSourceGallery("frames"))
    action("scrape-delete", async () => {
        const selection = selectedScrapePayload()
        if (!selection.names.length && !selection.frame_names.length) throw new Error("Move images or frames into Selected before deleting")
        const data = await api("/scrape/delete", { dataset: scrapeDatasetName(), attempt: scrapeAttempt, ...selection })
        selectedScrapeNames.clear()
        scrapeGallery = data
        renderGallery()
    })
    action("scrape-tag", async () => {
        if (!scrapeAttempt) throw new Error("Choose a scrape attempt first")
        galleryEl("status").textContent = "Submitting gallery for auto-tagging… Configure the tagger in Grabber Gallery."
        const job = await api("/scrape/tag", { dataset: scrapeDatasetName(), attempt: scrapeAttempt,
            replace_existing: true, tagger: el("tagger").value.trim(), threshold: Number(el("threshold").value),
            character_threshold: Number(el("character-threshold").value), exclude_tags: el("exclude").value })
        await acceptJob(job)
        el("scrape-status").textContent = "Gallery submitted for tagging; review tags before archiving."
        galleryEl("status").textContent = "Auto-tagging started. Return to the gallery and refresh when the job finishes."
    })
    action("scrape-archive", async () => {
        if (active) throw new Error("Wait for gallery tagging to finish before archiving")
        const targetDataset = el("dataset").value.trim()
        if (!targetDataset) throw new Error("Choose a training dataset first")
        const selection = selectedScrapePayload()
        if (!selection.names.length && !selection.frame_names.length) throw new Error("Select images or frames to submit to the training dataset")
        el("trigger").value = galleryEl("trigger-tags").value.trim()
        const data = await api("/scrape/archive", { dataset: scrapeDatasetName(), target_dataset: targetDataset,
            attempt: scrapeAttempt, ...selection })
        selectedScrapeNames.clear()
        names.forEach((name) => selectedScrapeNames.delete(name))
        await loadDatasetSuggestions(targetDataset)
        el("scrape-status").textContent = `${data.archived.length} images and captions submitted to ${targetDataset}. It is ready in Training → Dataset.`
        await refreshScrapeAttempts()
        await loadScrapeGallery()
    })
    async function acceptJob(job) {
        selectedJob = job.id
        active = true
        controls()
        await refreshJobs()
    }
    action("refresh", refresh)
    action("cancel", async () => { await api(`/jobs/${selectedJob}/cancel`, {}); await refreshJobs() })
    action("resume", async () => acceptJob(await api(`/jobs/${selectedJob}/resume`, {})))
    el("job").addEventListener("change", () => {
        selectedJob = el("job").value
        refreshJobs().catch(showError)
    })
    el("dataset").addEventListener("change", controls)
    el("scheduler").addEventListener("change", controls)
    el("backend").addEventListener("change", controls)
    el("epochs").addEventListener("input", controls)
    el("architecture").addEventListener("change", () => {
        el("resolution").value = el("architecture").value === "sdxl" ? "1024" : "512"
        controls()
    })
    el("kind").addEventListener("change", () => {
        const embedding = el("kind").value === "embedding"
        el("rank").disabled = embedding
        el("vectors").disabled = !embedding
        el("init-word").disabled = !embedding
        el("rate").value = embedding ? "0.0005" : "0.00005"
        controls()
    })
    el("form").addEventListener("submit", async (event) => {
        event.preventDefault()
        el("error").textContent = ""
        el("start").disabled = true
        try {
            await acceptJob(await api("/jobs", {
                dataset: el("dataset").value.trim(), kind: el("kind").value,
                architecture: el("architecture").value, model: modelPicker.value, backend: el("backend").value,
                qwen3: el("qwen3").value.trim(), vae: el("vae").value.trim(),
                output_name: el("name").value, trigger: el("trigger").value.trim(),
                init_word: el("init-word").value, precision: el("precision").value,
                steps: Number(el("steps").value), resolution: Number(el("resolution").value),
                epochs: el("epochs").disabled || el("epochs").value.trim() === "" ? null : Number(el("epochs").value),
                dataset_repeats: el("repeats").disabled ? 1 : Number(el("repeats").value),
                batch_size: Number(el("batch").value), rank: Number(el("rank").value),
                network_alpha: el("alpha").disabled || el("alpha").value.trim() === "" ? null : Number(el("alpha").value),
                lr_scheduler: el("scheduler").value,
                lr_warmup_steps: el("warmup").disabled ? 0 : Number(el("warmup").value),
                lr_scheduler_num_cycles: el("cycles").disabled ? 1 : Number(el("cycles").value),
                vectors: Number(el("vectors").value), learning_rate: Number(el("rate").value),
                text_encoder_learning_rate: el("text-encoder-rate").disabled ? 0 : Number(el("text-encoder-rate").value),
                save_every: Number(el("save-every").value), seed: Number(el("seed").value),
                checkpointing: el("checkpointing").value,
                blocks_to_swap: Number(el("blocks-swap").value),
                optimizer_type: el("optimizer").value,
                cache_text_encoder_outputs: el("cache-text").checked,
            }))
        } catch (error) { showError(error) } finally { controls() }
    })
    controls()
    loadScrapeSites().catch((error) => {
        galleryEl("status").textContent = `Could not load Grabber websites: ${error.message || String(error)}`
        el("scrape-source-note").textContent = "Website list failed to load. Refresh the page to retry."
    })
    loadDatasetSuggestions().catch((error) => {
        galleryEl("status").textContent = `Could not load dataset names: ${error.message || String(error)}`
    })
})()
