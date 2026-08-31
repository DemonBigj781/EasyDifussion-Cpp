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
        <button id="native-text-mask" class="button" type="button" title="Find likely text and load it as an editable inpaint mask">
            <i class="fa-solid fa-font"></i> Text mask
        </button>
        <button id="native-object-mask" class="button" type="button" title="Detect objects with the native C++ YOLO sidecar and mask them">
            <i class="fa-solid fa-vector-square"></i> Objects
        </button>
        <button id="native-face-mask" class="button" type="button" title="Detect faces with the native C++ YOLO sidecar and mask them">
            <i class="fa-regular fa-face-smile"></i> Faces
        </button>
        <small id="native-image-tools-status" aria-live="polite"></small>`
    buttons.appendChild(holder)

    const style = document.createElement("style")
    style.textContent = `
        #native-image-tools { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
        #native-image-tools .button { border: 0; font: inherit; }
        #native-image-tools-status { flex-basis: 100%; min-height: 1.25em; opacity: .8; }
    `
    document.head.appendChild(style)

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
        imageInpainter.show()
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

    document.getElementById("native-text-mask").addEventListener("click", () => run(textMask))
    document.getElementById("native-object-mask").addEventListener("click", () => run(() => detectionMask("objects")))
    document.getElementById("native-face-mask").addEventListener("click", () => run(() => detectionMask("face")))
})()
