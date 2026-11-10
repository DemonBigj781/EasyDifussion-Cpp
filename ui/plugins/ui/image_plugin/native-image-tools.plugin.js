// Native, RAM-bounded helpers for the initial-image/inpaint workflow.
;(function () {
    "use strict"

    if (window.__nativeImageToolsLoaded) return
    window.__nativeImageToolsLoaded = true

    const buttons = document.getElementById("init_image_buttons")
    const preview = document.getElementById("init_image_preview")
    if (!buttons || !preview || typeof imageInpainter === "undefined") return

    const holder = document.createElement("div")
    holder.id = "native-image-tools"
    holder.innerHTML = `
        <button id="native-remove-background" class="button" type="button" title="Remove the background locally with the bundled U²-Net ONNX model and replace the initial image">
            <i class="fa-solid fa-person-rays"></i> Remove background
        </button>
        <button id="native-text-mask" class="button" type="button" title="Create an editable inpaint mask for text removal">
            <i class="fa-solid fa-font"></i> Text remove mask
        </button>
        <button id="native-object-mask" class="button" type="button" title="Detect objects with the native C++ YOLO sidecar and load them as an editable inpaint mask">
            <i class="fa-solid fa-vector-square"></i> Object detect mask
        </button>
        <button id="native-face-mask" class="button" type="button" title="Detect faces with the native C++ YOLO sidecar and mask them">
            <i class="fa-regular fa-face-smile"></i> Faces
        </button>
        <button id="native-preview-mask" class="button" type="button" title="Preview the current inpaint mask over the initial image without changing the image">
            <i class="fa-regular fa-eye"></i> Preview mask
        </button>
        <button id="native-remove-object" class="button" type="button" title="Process the currently painted/loaded inpaint mask with the Deep Object Removal model">
            <i class="fa-solid fa-eraser"></i> Remove masked object
        </button>
        <small id="native-image-tools-status" aria-live="polite"></small>`
    buttons.appendChild(holder)

    const style = document.createElement("style")
    style.textContent = `
        #native-image-tools { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
        #native-image-tools .button { border: 0; font: inherit; }
        #native-image-tools-status { flex-basis: 100%; min-height: 1.25em; opacity: .8; }
        #native-mask-preview { position: relative; display: none; flex-basis: 100%; max-width: min(420px, 100%); border: 1px solid rgba(255,255,255,.25); border-radius: 6px; overflow: hidden; }
        #native-mask-preview.is-visible { display: block; }
        #native-mask-preview img { display: block; width: 100%; height: auto; }
    `
    document.head.appendChild(style)

    const previewPanel = document.createElement("div")
    previewPanel.id = "native-mask-preview"
    previewPanel.innerHTML = '<img alt="Inpaint mask preview">'
    holder.appendChild(previewPanel)

    const status = document.getElementById("native-image-tools-status")
    const controls = Array.from(holder.querySelectorAll("button"))

    function ensureImage() {
        if (!preview.src || !preview.naturalWidth || !preview.naturalHeight) {
            throw new Error("Choose an initial image first.")
        }
    }

    function imagePayload() {
        const canvas = document.createElement("canvas")
        canvas.width = preview.naturalWidth
        canvas.height = preview.naturalHeight
        canvas.getContext("2d").drawImage(preview, 0, 0)
        return canvas.toDataURL("image/png")
    }

    async function post(url, payload) {
        controls.forEach((button) => button.disabled = true)
        try {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            })
            const data = await response.json().catch(() => ({}))
            if (!response.ok) throw new Error(data.detail || `Tool failed (HTTP ${response.status})`)
            return data
        } finally {
            controls.forEach((button) => button.disabled = false)
        }
    }

    function loadMask(dataUrl, message) {
        imageInpainter.setImg(dataUrl)
        maskSetting.checked = true
        maskSetting.dispatchEvent(new Event("change"))
        status.textContent = message
        previewMask(dataUrl)
        imageInpainter.show()
    }

    function loadDataImage(dataUrl) {
        return new Promise((resolve, reject) => {
            const image = new Image()
            image.onload = () => resolve(image)
            image.onerror = () => reject(new Error("Could not preview the current mask or color guide."))
            image.src = dataUrl
        })
    }

    async function previewMask(maskDataUrl) {
        ensureImage()
        const mask = maskDataUrl || (maskSetting.checked ? imageInpainter.getImg() : null)
        const guidance = window.inpaintColorGuidanceImage || null
        if (!mask && !guidance) throw new Error("Create, paint, or save an inpaint mask or color guide first.")
        const canvas = document.createElement("canvas")
        canvas.width = preview.naturalWidth
        canvas.height = preview.naturalHeight
        const ctx = canvas.getContext("2d")
        ctx.drawImage(preview, 0, 0)
        if (mask) {
            const overlay = await loadDataImage(mask)
            ctx.save()
            ctx.globalAlpha = 0.45
            ctx.globalCompositeOperation = "source-over"
            ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height)
            ctx.restore()
        }
        if (guidance) {
            const colorGuide = await loadDataImage(guidance)
            ctx.save()
            ctx.globalAlpha = 0.7
            ctx.globalCompositeOperation = "source-over"
            ctx.drawImage(colorGuide, 0, 0, canvas.width, canvas.height)
            ctx.restore()
        }
        previewPanel.querySelector("img").src = canvas.toDataURL("image/png")
        previewPanel.classList.add("is-visible")
        status.textContent = guidance
            ? "Previewing the current inpaint mask and color guide. Edit them, then generate or use Remove masked object when ready."
            : "Previewing the current inpaint mask. Edit it, then use Remove masked object when ready."
    }

    async function removeBackground() {
        ensureImage()
        status.textContent = "Removing background…"
        const result = await post("/image-tools/remove-background", {
            image: imagePayload(),
            model: "u2net",
            alpha_matting: false,
        })
        if (!result.image) throw new Error("Background remover returned no image.")
        preview.src = result.image
        status.textContent = `Background removed at ${result.width} × ${result.height}.`
    }

    async function removeObject() {
        ensureImage()
        if (!maskSetting.checked) throw new Error("Paint, create, or save an inpaint mask first.")
        status.textContent = "Removing masked object…"
        const result = await post("/image-tools/remove-object", {
            image: imagePayload(),
            mask: imageInpainter.getImg(),
            feather: 4,
        })
        if (!result.image) throw new Error("Object remover returned no image.")
        preview.src = result.image
        maskSetting.checked = false
        maskSetting.dispatchEvent(new Event("change"))
        status.textContent = `Removed the masked object at ${result.width} × ${result.height}.`
    }

    async function textMask() {
        ensureImage()
        status.textContent = "Finding text for removal…"
        const result = await post("/image-tools/text-mask", {
            image: imagePayload(),
            sensitivity: 0.68,
            padding: 8,
        })
        if (!result.mask) throw new Error("Text mask returned no mask.")
        loadMask(result.mask, `Loaded ${result.regions} likely text-removal region${result.regions === 1 ? "" : "s"}. Preview/edit the mask, then use Remove masked object.`)
    }

    async function detectionMask(model) {
        ensureImage()
        status.textContent = model === "face" ? "Finding faces…" : "Finding objects…"
        const result = await post("/native-vision/detect", {
            image: imagePayload(),
            model,
            confidence: 0.25,
            iou: 0.45,
        })
        const canvas = document.createElement("canvas")
        canvas.width = imageInpainter.width
        canvas.height = imageInpainter.height
        const ctx = canvas.getContext("2d")
        const sx = canvas.width / result.width
        const sy = canvas.height / result.height
        ctx.fillStyle = "white"
        for (const detection of result.detections) {
            const pad = Math.max(4, Math.round(Math.min(detection.width * sx, detection.height * sy) * .04))
            ctx.fillRect(
                Math.max(0, detection.x * sx - pad),
                Math.max(0, detection.y * sy - pad),
                Math.min(canvas.width, detection.width * sx + 2 * pad),
                Math.min(canvas.height, detection.height * sy + 2 * pad),
            )
        }
        loadMask(canvas.toDataURL("image/png"), `Loaded ${result.detections.length} ${model === "face" ? "face" : "object"} mask${result.detections.length === 1 ? "" : "s"}.`)
    }

    function run(action) {
        Promise.resolve().then(action).catch((error) => {
            status.textContent = error.message
            console.error("Native image tool failed", error)
        })
    }

    document.getElementById("native-remove-background").addEventListener("click", () => run(removeBackground))
    document.getElementById("native-text-mask").addEventListener("click", () => run(textMask))
    document.getElementById("native-object-mask").addEventListener("click", () => run(() => detectionMask("objects")))
    document.getElementById("native-face-mask").addEventListener("click", () => run(() => detectionMask("face")))
    document.getElementById("native-preview-mask").addEventListener("click", () => run(previewMask))
    document.getElementById("native-remove-object").addEventListener("click", () => run(removeObject))
    document.addEventListener("inpaint-color-guidance-change", () => {
        if (previewPanel.classList.contains("is-visible")) run(previewMask)
    })
})()
