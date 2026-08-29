// Dedicated asymmetric outpainting for generated images.

;(function () {
    "use strict"
    if (window.__asymmetricOutpaintPluginLoaded) return
    window.__asymmetricOutpaintPluginLoaded = true

    const ID = "sdkit3-outpaint"
    const STATE_KEY = "easy-diffusion-outpaint-v1"
    const MAX_EDGE = 4096

    function clamp(value, minimum, maximum, fallback) {
        const number = Number(value)
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback
    }

    function round8(value) {
        return Math.max(0, Math.round(clamp(value, 0, MAX_EDGE, 0) / 8) * 8)
    }

    function notify(message, isError = false) {
        if (typeof showToast === "function") showToast(message, 5000, isError)
        else if (isError) console.error(message)
        else console.log(message)
    }

    function readState() {
        try { return JSON.parse(localStorage.getItem(STATE_KEY) || "{}") }
        catch (_) { return {} }
    }

    function dataPath(input) {
        return input?.dataset?.path || input?.value || ""
    }

    const dialog = document.createElement("dialog")
    dialog.id = `${ID}-dialog`
    dialog.innerHTML = `
        <form method="dialog" class="outpaint-dialog-card">
            <button class="outpaint-close tertiaryButton" value="cancel" title="Close" aria-label="Close">×</button>
            <h2>Outpaint</h2>
            <p class="outpaint-subtitle">Expand any combination of edges. The source image stays pixel-aligned.</p>
            <div class="outpaint-layout">
                <section>
                    <canvas id="${ID}-preview" width="480" height="320"></canvas>
                    <div id="${ID}-size" class="outpaint-size"></div>
                    <div class="outpaint-edge-grid">
                        <span></span><label>Top <input id="${ID}-top" type="number" min="0" max="${MAX_EDGE}" step="8"></label><span></span>
                        <label>Left <input id="${ID}-left" type="number" min="0" max="${MAX_EDGE}" step="8"></label>
                        <button id="${ID}-equal" type="button" class="tertiaryButton" title="Set every edge to the largest entered value"><i class="fa-solid fa-arrows"></i> Equal</button>
                        <label>Right <input id="${ID}-right" type="number" min="0" max="${MAX_EDGE}" step="8"></label>
                        <span></span><label>Bottom <input id="${ID}-bottom" type="number" min="0" max="${MAX_EDGE}" step="8"></label><span></span>
                    </div>
                    <div class="outpaint-row">
                        <label>Edge overlap <input id="${ID}-overlap" type="number" min="0" max="512" step="8"></label>
                        <label>Fill
                            <select id="${ID}-fill"><option value="mirror">Mirror edges</option><option value="stretch">Stretch edges</option><option value="noise">Noise</option></select>
                        </label>
                    </div>
                </section>
                <section class="outpaint-settings">
                    <label>Prompt<textarea id="${ID}-prompt"></textarea></label>
                    <label>Negative prompt<textarea id="${ID}-negative"></textarea></label>
                    <label>Model<input id="${ID}-model" type="text" spellcheck="false" autocomplete="off" class="model-filter model-selector"></label>
                    <label>VAE<input id="${ID}-vae" type="text" spellcheck="false" autocomplete="off" class="model-filter model-selector"></label>
                    <label>Sampler<span id="${ID}-sampler-slot"></span></label>
                    <label>Scheduler<span id="${ID}-scheduler-slot"></span></label>
                    <div class="outpaint-row">
                        <label>Steps <input id="${ID}-steps" type="number" min="1" max="500" step="1"></label>
                        <label>CFG <input id="${ID}-cfg" type="number" min="0" max="50" step="0.1"></label>
                        <label>Strength <input id="${ID}-strength" type="number" min="0.01" max="1" step="0.01"></label>
                    </div>
                    <div class="outpaint-row">
                        <label>Images <input id="${ID}-total" type="number" min="1" max="100" step="1"></label>
                        <label>Parallel <input id="${ID}-parallel" type="number" min="1" max="16" step="1"></label>
                        <label>Seed <input id="${ID}-seed" type="number" min="0" max="4294967295" step="1"></label>
                    </div>
                </section>
            </div>
            <div class="outpaint-actions">
                <button value="cancel" class="secondaryButton">Cancel</button>
                <button id="${ID}-apply" value="default" class="primaryButton"><i class="fa-solid fa-expand"></i> Generate outpaint</button>
            </div>
        </form>`
    document.body.appendChild(dialog)

    const style = document.createElement("style")
    style.textContent = `
        #${ID}-dialog { width:min(980px,96vw); max-height:94vh; padding:0; border:0; border-radius:8px; color:var(--text-color); background:var(--background-color2); }
        #${ID}-dialog::backdrop { background:rgba(0,0,0,.68); }
        .outpaint-dialog-card { padding:20px; position:relative; overflow:auto; max-height:calc(94vh - 40px); }
        .outpaint-dialog-card h2 { margin:0 32px 2px 0; }
        .outpaint-subtitle { margin:0 0 14px; opacity:.8; }
        .outpaint-close { position:absolute; right:12px; top:10px; font-size:24px; }
        .outpaint-layout { display:grid; grid-template-columns:minmax(330px,1.05fr) minmax(300px,.95fr); gap:20px; }
        #${ID}-preview { display:block; width:100%; height:auto; max-height:46vh; background:#171717; border:1px solid var(--background-color4); touch-action:pan-y pinch-zoom; }
        .outpaint-size { text-align:center; margin:6px 0; font-variant-numeric:tabular-nums; }
        .outpaint-edge-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:7px; align-items:center; text-align:center; }
        .outpaint-edge-grid label { display:flex; flex-direction:column; gap:3px; }
        .outpaint-edge-grid input { width:100%; box-sizing:border-box; text-align:center; }
        .outpaint-row { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }
        .outpaint-row > label { flex:1; min-width:105px; }
        .outpaint-row input,.outpaint-row select { width:100%; box-sizing:border-box; }
        .outpaint-settings { display:flex; flex-direction:column; gap:9px; }
        .outpaint-settings > label { display:flex; flex-direction:column; gap:3px; }
        .outpaint-settings textarea { min-height:60px; resize:vertical; }
        .outpaint-settings select,.outpaint-settings input { width:100%; box-sizing:border-box; }
        .outpaint-actions { display:flex; justify-content:flex-end; gap:10px; margin-top:18px; }
        @media (max-width:760px) {
            .outpaint-layout { grid-template-columns:1fr; }
            .outpaint-dialog-card { padding:14px; }
            #${ID}-preview { max-height:36vh; }
        }`
    document.head.appendChild(style)

    const byId = (suffix) => document.getElementById(`${ID}-${suffix}`)
    const preview = byId("preview")
    const previewContext = preview.getContext("2d")
    const edgeNames = ["left", "right", "top", "bottom"]
    const edgeFields = Object.fromEntries(edgeNames.map((name) => [name, byId(name)]))
    let sourceImage = null
    let sourceRequest = null

    const sampler = document.getElementById("sampler_name")?.cloneNode(true)
    const scheduler = document.getElementById("scheduler_name")?.cloneNode(true)
    if (sampler) { sampler.id = `${ID}-sampler`; byId("sampler-slot").appendChild(sampler) }
    if (scheduler) { scheduler.id = `${ID}-scheduler`; byId("scheduler-slot").appendChild(scheduler) }
    const model = typeof ModelDropdown === "function" ? new ModelDropdown(byId("model"), "stable-diffusion") : null
    const vae = typeof ModelDropdown === "function" ? new ModelDropdown(byId("vae"), "vae", "None") : null

    function expansions() {
        return Object.fromEntries(edgeNames.map((name) => [name, round8(edgeFields[name].value)]))
    }

    function drawPreview() {
        if (!sourceImage) return
        const edges = expansions()
        const outputWidth = sourceImage.naturalWidth + edges.left + edges.right
        const outputHeight = sourceImage.naturalHeight + edges.top + edges.bottom
        byId("size").textContent = `${sourceImage.naturalWidth} × ${sourceImage.naturalHeight} → ${outputWidth} × ${outputHeight}`
        previewContext.clearRect(0, 0, preview.width, preview.height)
        const scale = Math.min(preview.width / outputWidth, preview.height / outputHeight)
        const ox = (preview.width - outputWidth * scale) / 2
        const oy = (preview.height - outputHeight * scale) / 2
        previewContext.fillStyle = "#444"
        previewContext.fillRect(ox, oy, outputWidth * scale, outputHeight * scale)
        previewContext.fillStyle = "rgba(103,58,183,.48)"
        previewContext.fillRect(ox, oy, outputWidth * scale, edges.top * scale)
        previewContext.fillRect(ox, oy + (edges.top + sourceImage.naturalHeight) * scale, outputWidth * scale, edges.bottom * scale)
        previewContext.fillRect(ox, oy + edges.top * scale, edges.left * scale, sourceImage.naturalHeight * scale)
        previewContext.fillRect(ox + (edges.left + sourceImage.naturalWidth) * scale, oy + edges.top * scale, edges.right * scale, sourceImage.naturalHeight * scale)
        previewContext.drawImage(sourceImage, ox + edges.left * scale, oy + edges.top * scale, sourceImage.naturalWidth * scale, sourceImage.naturalHeight * scale)
        previewContext.strokeStyle = "#fff"
        previewContext.strokeRect(ox + edges.left * scale, oy + edges.top * scale, sourceImage.naturalWidth * scale, sourceImage.naturalHeight * scale)
    }

    function mirrorHorizontal(ctx, image, x, y, width, height, fromLeft) {
        if (width <= 0) return
        const sample = Math.max(1, Math.min(width, image.naturalWidth))
        ctx.save()
        if (fromLeft) {
            ctx.translate(x + width, 0)
            ctx.scale(-1, 1)
            ctx.drawImage(image, 0, 0, sample, image.naturalHeight, 0, y, width, height)
        } else {
            ctx.translate(x, 0)
            ctx.scale(-1, 1)
            ctx.drawImage(image, image.naturalWidth - sample, 0, sample, image.naturalHeight, -width, y, width, height)
        }
        ctx.restore()
    }

    function stretchHorizontal(ctx, image, x, y, width, height, fromLeft) {
        if (width <= 0) return
        const sx = fromLeft ? 0 : image.naturalWidth - 1
        ctx.drawImage(image, sx, 0, 1, image.naturalHeight, x, y, width, height)
    }

    function applyNoise(canvas, edges) {
        const ctx = canvas.getContext("2d", { willReadFrequently: true })
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const pixels = imageData.data
        const x0 = edges.left
        const x1 = edges.left + sourceImage.naturalWidth
        const y0 = edges.top
        const y1 = edges.top + sourceImage.naturalHeight
        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                if (x >= x0 && x < x1 && y >= y0 && y < y1) continue
                const offset = (y * canvas.width + x) * 4
                const value = 64 + Math.floor(Math.random() * 128)
                pixels[offset] = value
                pixels[offset + 1] = value
                pixels[offset + 2] = value
                pixels[offset + 3] = 255
            }
        }
        ctx.putImageData(imageData, 0, 0)
    }

    function buildCanvases(image, edges, fillMode, overlap) {
        const canvas = document.createElement("canvas")
        canvas.width = image.naturalWidth + edges.left + edges.right
        canvas.height = image.naturalHeight + edges.top + edges.bottom
        const ctx = canvas.getContext("2d")
        ctx.fillStyle = "#777"
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        if (fillMode !== "noise") {
            const fillHorizontal = fillMode === "stretch" ? stretchHorizontal : mirrorHorizontal
            fillHorizontal(ctx, image, 0, edges.top, edges.left, image.naturalHeight, true)
            fillHorizontal(ctx, image, edges.left + image.naturalWidth, edges.top, edges.right, image.naturalHeight, false)
        }
        ctx.drawImage(image, edges.left, edges.top)

        if (fillMode !== "noise" && (edges.top > 0 || edges.bottom > 0)) {
            const middle = document.createElement("canvas")
            middle.width = canvas.width
            middle.height = canvas.height
            middle.getContext("2d").drawImage(canvas, 0, 0)
            if (edges.top > 0) {
                if (fillMode === "stretch") ctx.drawImage(middle, 0, edges.top, canvas.width, 1, 0, 0, canvas.width, edges.top)
                else {
                    ctx.save(); ctx.translate(0, edges.top); ctx.scale(1, -1)
                    ctx.drawImage(middle, 0, edges.top, canvas.width, Math.min(edges.top, image.naturalHeight), 0, 0, canvas.width, edges.top)
                    ctx.restore()
                }
            }
            if (edges.bottom > 0) {
                if (fillMode === "stretch") ctx.drawImage(middle, 0, edges.top + image.naturalHeight - 1, canvas.width, 1, 0, edges.top + image.naturalHeight, canvas.width, edges.bottom)
                else {
                    ctx.save(); ctx.translate(0, edges.top + image.naturalHeight); ctx.scale(1, -1)
                    ctx.drawImage(middle, 0, edges.top + image.naturalHeight - Math.min(edges.bottom, image.naturalHeight), canvas.width, Math.min(edges.bottom, image.naturalHeight), 0, -edges.bottom, canvas.width, edges.bottom)
                    ctx.restore()
                }
            }
        }
        if (fillMode === "noise") applyNoise(canvas, edges)

        const mask = document.createElement("canvas")
        mask.width = canvas.width
        mask.height = canvas.height
        const maskCtx = mask.getContext("2d")
        maskCtx.fillStyle = "black"
        maskCtx.fillRect(0, 0, mask.width, mask.height)
        maskCtx.fillStyle = "white"
        if (edges.top) maskCtx.fillRect(0, 0, mask.width, edges.top)
        if (edges.bottom) maskCtx.fillRect(0, edges.top + image.naturalHeight, mask.width, edges.bottom)
        if (edges.left) maskCtx.fillRect(0, edges.top, edges.left, image.naturalHeight)
        if (edges.right) maskCtx.fillRect(edges.left + image.naturalWidth, edges.top, edges.right, image.naturalHeight)

        const feather = Math.max(0, Math.min(round8(overlap), Math.floor(Math.min(image.naturalWidth, image.naturalHeight) / 2)))
        if (feather > 0) {
            maskCtx.globalCompositeOperation = "lighten"
            const gradients = []
            if (edges.left) gradients.push([edges.left, 0, edges.left + feather, 0, edges.left, edges.top, feather, image.naturalHeight])
            if (edges.right) gradients.push([edges.left + image.naturalWidth, 0, edges.left + image.naturalWidth - feather, 0, edges.left + image.naturalWidth - feather, edges.top, feather, image.naturalHeight])
            if (edges.top) gradients.push([0, edges.top, 0, edges.top + feather, edges.left, edges.top, image.naturalWidth, feather])
            if (edges.bottom) gradients.push([0, edges.top + image.naturalHeight, 0, edges.top + image.naturalHeight - feather, edges.left, edges.top + image.naturalHeight - feather, image.naturalWidth, feather])
            for (const [x0, y0, x1, y1, x, y, width, height] of gradients) {
                const gradient = maskCtx.createLinearGradient(x0, y0, x1, y1)
                gradient.addColorStop(0, "white")
                gradient.addColorStop(1, "black")
                maskCtx.fillStyle = gradient
                maskCtx.fillRect(x, y, width, height)
            }
            maskCtx.globalCompositeOperation = "source-over"
        }
        return { image: canvas.toDataURL("image/png"), mask: mask.toDataURL("image/png") }
    }

    function removeLegacyEditors() {
        if (!Array.isArray(PLUGINS?.IMAGE_INFO_BUTTONS)) return
        for (const group of PLUGINS.IMAGE_INFO_BUTTONS) {
            if (!Array.isArray(group)) continue
            for (let index = group.length - 1; index >= 0; index--) {
                const item = group[index]
                if (item?.text === "Move" || String(item?.html || "").includes("outpaint-label")) group.splice(index, 1)
            }
        }
        document.getElementById("outpaintit-settings")?.classList.add("displayNone")
    }

    async function openOutpaint(request, image) {
        if (!image?.naturalWidth || !image?.naturalHeight) {
            notify("The source image is not available for outpainting.", true)
            return
        }
        // Snapshot the source request immediately. Later UI changes cannot race
        // the asynchronous dialog/task submission.
        sourceRequest = JSON.parse(JSON.stringify(request || {}))
        sourceImage = image
        const state = readState()
        for (const name of edgeNames) edgeFields[name].value = round8(state[name] ?? 128)
        byId("overlap").value = round8(state.overlap ?? 32)
        byId("fill").value = state.fill || "mirror"
        byId("prompt").value = sourceRequest.prompt || ""
        byId("negative").value = sourceRequest.negative_prompt || ""
        if (model) model.value = sourceRequest.use_stable_diffusion_model || dataPath(document.getElementById("stable_diffusion_model"))
        if (vae) vae.value = sourceRequest.use_vae_model || dataPath(document.getElementById("vae_model"))
        if (sampler) sampler.value = sourceRequest.sampler_name || document.getElementById("sampler_name")?.value || "euler_a"
        if (scheduler) scheduler.value = sourceRequest.scheduler_name || document.getElementById("scheduler_name")?.value || "simple"
        byId("steps").value = sourceRequest.num_inference_steps || 25
        byId("cfg").value = sourceRequest.guidance_scale ?? 7.5
        byId("strength").value = state.strength ?? sourceRequest.prompt_strength ?? 0.9
        byId("total").value = state.total ?? 1
        byId("parallel").value = state.parallel ?? 1
        byId("seed").value = Math.floor(Math.random() * 4294967296)
        drawPreview()
        dialog.showModal()
    }

    edgeNames.forEach((name) => edgeFields[name].addEventListener("input", drawPreview))
    byId("equal").addEventListener("click", () => {
        const value = Math.max(...Object.values(expansions())) || 128
        edgeNames.forEach((name) => { edgeFields[name].value = value })
        drawPreview()
    })
    dialog.addEventListener("click", (event) => {
        if (event.target === dialog) dialog.close("cancel")
    })
    byId("apply").addEventListener("click", (event) => {
        event.preventDefault()
        if (!sourceImage || !sourceRequest) return

        // Read and freeze every option before doing canvas work or queuing.
        const edges = expansions()
        if (!Object.values(edges).some(Boolean)) {
            notify("Enter an expansion for at least one edge.", true)
            return
        }
        const settings = Object.freeze({
            ...edges,
            overlap: round8(byId("overlap").value),
            fill: byId("fill").value,
            prompt: byId("prompt").value,
            negativePrompt: byId("negative").value,
            diffusionModel: model?.value || sourceRequest.use_stable_diffusion_model,
            vaeModel: vae?.value || sourceRequest.use_vae_model,
            samplerName: sampler?.value || sourceRequest.sampler_name,
            schedulerName: scheduler?.value || sourceRequest.scheduler_name,
            steps: Math.round(clamp(byId("steps").value, 1, 500, 25)),
            cfg: clamp(byId("cfg").value, 0, 50, 7.5),
            strength: clamp(byId("strength").value, 0.01, 1, 0.9),
            total: Math.round(clamp(byId("total").value, 1, 100, 1)),
            parallel: Math.round(clamp(byId("parallel").value, 1, 16, 1)),
            seed: Math.round(clamp(byId("seed").value, 0, 4294967295, 0)),
        })
        localStorage.setItem(STATE_KEY, JSON.stringify(settings))

        try {
            const rendered = buildCanvases(sourceImage, settings, settings.fill, settings.overlap)
            const parallel = Math.min(settings.parallel, settings.total)
            const newTask = modifyCurrentRequest(sourceRequest, {
                init_image: rendered.image,
                mask: rendered.mask,
                width: sourceImage.naturalWidth + settings.left + settings.right,
                height: sourceImage.naturalHeight + settings.top + settings.bottom,
                prompt: settings.prompt,
                negative_prompt: settings.negativePrompt,
                prompt_strength: settings.strength,
                num_inference_steps: settings.steps,
                guidance_scale: settings.cfg,
                sampler_name: settings.samplerName,
                scheduler_name: settings.schedulerName,
                use_stable_diffusion_model: settings.diffusionModel,
                use_vae_model: settings.vaeModel,
                num_outputs: parallel,
                seed: settings.seed,
                preserve_init_image_color_profile: true,
            })
            newTask.numOutputsTotal = settings.total
            newTask.batchCount = Math.ceil(settings.total / parallel)
            newTask.seed = settings.seed

            // Spatial conditioning for the old image has the wrong dimensions
            // after expansion. Leave non-spatial generation settings intact.
            delete newTask.reqBody.use_controlnet_model
            delete newTask.reqBody.control_image
            delete newTask.reqBody.control_filter_to_apply
            delete newTask.reqBody.control_net_lllite_model
            delete newTask.reqBody.control_net_lllite_image
            dialog.close("generate")
            createTask(newTask)
        } catch (error) {
            console.error(error)
            notify(`Unable to prepare outpaint: ${error.message}`, true)
        }
    })

    PLUGINS.IMAGE_INFO_BUTTONS.push([
        { type: "label", class: ["imgInfoLabel", "imgSeedLabel"], text: "Edit" },
        { text: "Outpaint", filter: () => true, on_click: openOutpaint },
    ])

    // Core and user plugins may load in either order.
    removeLegacyEditors()
    setTimeout(removeLegacyEditors, 0)
    setTimeout(removeLegacyEditors, 500)
    setTimeout(removeLegacyEditors, 2000)
})()
