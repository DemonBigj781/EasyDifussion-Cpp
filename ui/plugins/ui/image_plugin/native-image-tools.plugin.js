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
        <button id="native-text-mask" class="button" type="button" title="Find likely text and load it as an editable inpaint mask">
            <i class="fa-solid fa-font"></i> Text mask
        </button>
        <button id="native-object-mask" class="button" type="button" title="Detect objects with the native C++ YOLO sidecar and mask them">
            <i class="fa-solid fa-vector-square"></i> Objects
        </button>
        <button id="native-face-mask" class="button" type="button" title="Detect faces with the native C++ YOLO sidecar and mask them">
            <i class="fa-regular fa-face-smile"></i> Faces
        </button>
        <button id="native-remove-object" class="button" type="button" title="Remove the currently painted inpaint mask with the Deep Object Removal model">
            <i class="fa-solid fa-eraser"></i> Remove object
        </button>
        <details id="native-sam" class="native-sam">
            <summary>SAM Segmentation</summary>
            <div class="native-sam-grid">
                <label for="native-sam-image">Source image</label>
                <input id="native-sam-image" type="file" accept="image/*">
                <label for="native-sam-model">SAM model</label>
                <input id="native-sam-model" class="model-filter" data-path="" type="text" autocomplete="off" placeholder="Search SAM models…">
                <label for="native-sam-prompt-type">Prompt</label>
                <select id="native-sam-prompt-type"><option value="point">Point</option><option value="box">Box</option></select>
                <label for="native-sam-point-x">Point X</label><input id="native-sam-point-x" type="number" min="0" step="0.5" value="0">
                <label for="native-sam-point-y">Point Y</label><input id="native-sam-point-y" type="number" min="0" step="0.5" value="0">
                <label for="native-sam-box">Box x1,y1,x2,y2</label><input id="native-sam-box" type="text" value="0,0,1,1" placeholder="x1,y1,x2,y2">
            </div>
            <button id="native-sam-generate" class="button" type="button">Generate mask</button>
            <img id="native-sam-preview" class="native-sam-preview" alt="Generated SAM mask" hidden>
        </details>
        <small id="native-image-tools-status" aria-live="polite"></small>`
    buttons.appendChild(holder)

    const style = document.createElement("style")
    style.textContent = `
        #native-image-tools { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
        #native-image-tools .button { border: 0; font: inherit; }
        #native-image-tools-status { flex-basis: 100%; min-height: 1.25em; opacity: .8; }
        .native-sam { flex-basis: 100%; margin-top: 8px; }
        .native-sam-grid { display: grid; grid-template-columns: minmax(110px, auto) minmax(0, 1fr); gap: 6px 8px; margin: 8px 0; }
        .native-sam-preview { max-width: 240px; max-height: 180px; display: block; margin-top: 8px; }
        @media (max-width: 700px) { .native-sam-grid { grid-template-columns: 1fr; } }
    `
    document.head.appendChild(style)

    const status = document.getElementById("native-image-tools-status")
    const controls = Array.from(holder.querySelectorAll("button"))
    const samModel = typeof ModelDropdown === "function"
        ? new ModelDropdown(document.getElementById("native-sam-model"), "sam", "None")
        : null

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

    let samSource = ""
    document.getElementById("native-sam-image").addEventListener("change", () => {
        const file = document.getElementById("native-sam-image").files?.[0]
        if (!file) return
        const reader = new FileReader()
        reader.addEventListener("load", () => { samSource = String(reader.result || "") })
        reader.readAsDataURL(file)
    })

    async function segmentWithSam() {
        const source = samSource || imagePayload()
        if (!source) throw new Error("Choose an image first.")
        if (!samModel?.value) throw new Error("Choose a SAM model first.")
        const promptType = document.getElementById("native-sam-prompt-type").value
        const payload = {image: source, model: samModel.value, prompt_type: promptType, multimask: false}
        if (promptType === "point") {
            payload.point = {x: Number(document.getElementById("native-sam-point-x").value), y: Number(document.getElementById("native-sam-point-y").value)}
        } else {
            const values = document.getElementById("native-sam-box").value.split(",").map(Number)
            if (values.length !== 4) throw new Error("Box prompt needs x1,y1,x2,y2.")
            payload.box = {x1: values[0], y1: values[1], x2: values[2], y2: values[3]}
        }
        const result = await post("/sam/segment", payload)
        if (!result.mask) throw new Error("SAM returned no mask.")
        const maskPreview = document.getElementById("native-sam-preview")
        maskPreview.src = result.mask
        maskPreview.hidden = false
        status.textContent = `SAM mask generated at ${result.width} × ${result.height}.`
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
        imageInpainter.show()
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
        if (!maskSetting.checked) throw new Error("Paint and save an inpaint mask first.")
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
        status.textContent = "Finding text…"
        const result = await post("/image-tools/text-mask", {
            image: imagePayload(),
            sensitivity: 0.62,
            padding: 4,
        })
        loadMask(result.mask, `Loaded ${result.regions} likely text region${result.regions === 1 ? "" : "s"}. Edit the mask, then Save.`)
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
    document.getElementById("native-remove-object").addEventListener("click", () => run(removeObject))
    document.getElementById("native-sam-generate").addEventListener("click", () => run(segmentWithSam))
})()
