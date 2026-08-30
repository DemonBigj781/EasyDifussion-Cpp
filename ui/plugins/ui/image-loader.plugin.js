// Image Loader plugin for Easy Diffusion
// Owns both image-input UIs while preserving Easy Diffusion's native IDs.
// v2.0.0, last updated: 8/22/2026

;(function () {
    "use strict"

    function ensureInitialImageUI() {
        const existing = document.getElementById("initial-image-settings")
        if (existing) return existing

        const editorSettings = document.getElementById("editor-settings")
        if (!editorSettings?.parentNode) {
            throw new Error("Initial and Reference Images plugin: Image Settings panel was not found")
        }

        const panel = document.createElement("div")
        panel.id = "initial-image-settings"
        panel.className = "settings-box panel-box"
        panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/image-loader.plugin.html")

        editorSettings.parentNode.insertBefore(panel, editorSettings)
        return panel
    }

    window.ensureInitialImageUI = ensureInitialImageUI
    ensureInitialImageUI()

    // This required plugin is bootstrapped before main.js. Install behavior
    // only once if the script is requested again during development.
    if (window.__initialAndReferenceImagesPluginLoaded) return
    window.__initialAndReferenceImagesPluginLoaded = true

    const getRefContainer = () => document.getElementById("editor-inputs-ref-images")

    // The complete Reference Images loader is part of this panel. Preserve the
    // native model check, then keep the loader available for sdkit3 models.
    function patchCheckReferenceImageField() {
        if (typeof window.checkReferenceImageField !== "function") {
            setTimeout(patchCheckReferenceImageField, 200)
            return
        }
        const original = window.checkReferenceImageField
        window.checkReferenceImageField = function () {
            original.apply(this, arguments)
            getRefContainer()?.classList.remove("displayNone")
        }
        getRefContainer()?.classList.remove("displayNone")
    }

    patchCheckReferenceImageField()

    function notify(message, isError) {
        if (typeof showToast === "function") showToast(message, 5000, Boolean(isError))
        else if (isError) console.error(message)
        else console.log(message)
    }

    function decodePngTextChunks(buffer) {
        const bytes = new Uint8Array(buffer)
        const signature = [137, 80, 78, 71, 13, 10, 26, 10]
        if (bytes.length < signature.length || signature.some((byte, index) => bytes[index] !== byte)) {
            throw new Error("The selected file is not a valid PNG image")
        }

        const view = new DataView(buffer)
        const latin1 = new TextDecoder("latin1")
        const utf8 = new TextDecoder("utf-8")
        const metadata = {}
        let offset = 8

        while (offset + 12 <= bytes.length) {
            const length = view.getUint32(offset)
            const dataStart = offset + 8
            const dataEnd = dataStart + length
            if (dataEnd + 4 > bytes.length) throw new Error("The PNG metadata is truncated")
            const type = latin1.decode(bytes.subarray(offset + 4, offset + 8))
            const data = bytes.subarray(dataStart, dataEnd)

            if (type === "tEXt") {
                const separator = data.indexOf(0)
                if (separator > 0) {
                    const key = latin1.decode(data.subarray(0, separator))
                    metadata[key] = latin1.decode(data.subarray(separator + 1))
                }
            } else if (type === "iTXt") {
                const separator = data.indexOf(0)
                if (separator > 0 && separator + 2 < data.length && data[separator + 1] === 0) {
                    // Easy Diffusion writes tEXt today, but accept uncompressed
                    // international text chunks produced by other PNG tools.
                    let textStart = separator + 3
                    for (let skippedFields = 0; skippedFields < 2; skippedFields += 1) {
                        const next = data.indexOf(0, textStart)
                        if (next < 0) {
                            textStart = data.length
                            break
                        }
                        textStart = next + 1
                    }
                    const key = latin1.decode(data.subarray(0, separator))
                    metadata[key] = utf8.decode(data.subarray(textStart))
                }
            }

            offset = dataEnd + 4
            if (type === "IEND") break
        }
        return metadata
    }

    function decodeExifUserComment(value) {
        const bytes = value instanceof Uint8Array ? value : Uint8Array.from(value || [])
        if (bytes.length < 8) throw new Error("The image has no usable EXIF UserComment")
        const prefix = String.fromCharCode(...bytes.subarray(0, 8))
        const body = bytes.subarray(8)
        if (prefix === "UNICODE\0") return new TextDecoder("utf-16be").decode(body).replace(/^\uFEFF/, "")
        if (prefix === "ASCII\0\0\0") return new TextDecoder("utf-8").decode(body)
        if (prefix === "JIS\0\0\0\0\0") return new TextDecoder("shift-jis").decode(body)
        return new TextDecoder("utf-8").decode(bytes).replace(/^\0+/, "")
    }

    function parseEmbeddedList(value) {
        if (Array.isArray(value)) return value
        if (typeof value !== "string") return value
        try {
            const parsed = JSON.parse(value)
            if (Array.isArray(parsed)) return parsed
        } catch (_) {
            // Pillow stores Python's printable list form in PNG tEXt chunks.
        }
        if (!/^\s*\[.*\]\s*$/.test(value)) return value
        const matches = value.match(/'(?:\\.|[^'])*'|"(?:\\.|[^"])*"/g) || []
        return matches.map((item) => item.slice(1, -1).replace(/\\(['"\\])/g, "$1"))
    }

    function normalizePngMetadata(metadata) {
        const result = {}
        Object.entries(metadata).forEach(([key, originalValue]) => {
            let value = originalValue
            if (value === "None") value = null
            else if (/^\s*\[.*\]\s*$/.test(value)) value = parseEmbeddedList(value)
            else if (typeof TASK_MAPPING !== "undefined" && TASK_MAPPING[key]?.parse) {
                value = TASK_MAPPING[key].parse(value)
            }
            result[key] = value
        })
        return result
    }

    async function readSetupMetadata(file) {
        const buffer = await file.arrayBuffer()
        if (file.type === "image/png" || file.name.toLowerCase().endsWith(".png")) {
            return normalizePngMetadata(decodePngTextChunks(buffer))
        }
        if (typeof ExifReader === "undefined") throw new Error("The EXIF reader is not available")
        const tags = await ExifReader.load(buffer)
        const userComment = tags?.UserComment
        if (!userComment) throw new Error("No Easy Diffusion setup was found in the image EXIF data")
        const text = typeof userComment.description === "string" && userComment.description.startsWith("{")
            ? userComment.description
            : decodeExifUserComment(userComment.value)
        return JSON.parse(text)
    }

    async function loadSetupFromImage(file, askBeforeRestore = true) {
        if (typeof restoreTaskToUI !== "function") throw new Error("Easy Diffusion's settings UI is not ready")
        const reqBody = await readSetupMetadata(file)
        const recognizedKeys = Object.keys(reqBody || {}).filter((key) =>
            typeof TASK_MAPPING !== "undefined" && Object.prototype.hasOwnProperty.call(TASK_MAPPING, key)
        )
        if (!reqBody || typeof reqBody !== "object" || recognizedKeys.length === 0) {
            throw new Error("No Easy Diffusion generation setup was found in this image")
        }
        if (askBeforeRestore && !window.confirm(`Restore ${recognizedKeys.length} generation settings from ${file.name}?`)) {
            return false
        }

        const seed = Number.parseInt(reqBody.seed, 10)
        const numOutputs = Number.parseInt(reqBody.num_outputs, 10)
        const task = {
            reqBody,
            numOutputsTotal: Number.isFinite(numOutputs) && numOutputs > 0 ? numOutputs : 1,
        }
        if (Number.isFinite(seed)) task.seed = seed
        restoreTaskToUI(task, typeof TASK_REQ_NO_EXPORT === "undefined" ? [] : TASK_REQ_NO_EXPORT)
        notify(`Loaded generation setup from ${file.name}.`)
        return true
    }

    window.loadSetupFromImage = loadSetupFromImage

    function importAsCompletedQueueImage(dataUrl, file) {
        if (typeof getCurrentUserRequest !== "function" || typeof createTask !== "function" ||
            typeof onRenderTaskCompleted !== "function") {
            throw new Error("Easy Diffusion's queue UI is not ready")
        }

        const loadedImage = new Image()
        loadedImage.addEventListener("load", () => {
            const task = getCurrentUserRequest()
            const prompt = document.getElementById("prompt")?.value?.trim() || ""
            task.reqBody.prompt = prompt
            task.reqBody.original_prompt = prompt
            task.reqBody.width = loadedImage.naturalWidth
            task.reqBody.height = loadedImage.naturalHeight
            task.reqBody.imported_image_name = file.name
            task.reqBody.imported_image = true
            ;[
                "init_image", "mask", "strict_mask_border",
                "control_image", "control_filter_to_apply", "control_image_preprocessed",
                "ref_images",
            ].forEach((key) => delete task.reqBody[key])
            if (file.type === "image/png") task.reqBody.output_format = "png"
            else if (file.type === "image/webp") task.reqBody.output_format = "webp"
            else if (file.type === "image/jpeg") task.reqBody.output_format = "jpeg"

            task.numOutputsTotal = 1
            task.batchCount = 1
            task.batchesDone = 1
            task.startTime = Date.now()
            createTask(task)

            const outputContainer = document.createElement("div")
            outputContainer.className = "img-batch"
            task.outputContainer.insertBefore(outputContainer, task.outputContainer.firstChild)
            onRenderTaskCompleted(task, task.reqBody, null, outputContainer, {
                status: "succeeded",
                output: [{ data: dataUrl, seed: task.seed }],
            })
            task.outputMsg.innerText = `Loaded ${file.name} for post-image operations`
            notify(`Loaded ${file.name} as a completed queue image.`)
        }, { once: true })
        loadedImage.addEventListener("error", () => notify(`Could not decode ${file.name}.`, true), { once: true })
        loadedImage.src = dataUrl
    }

    const queueImageInput = document.getElementById("queue_image_input")
    queueImageInput?.addEventListener("change", () => {
        const file = queueImageInput.files?.[0]
        queueImageInput.value = ""
        if (!file?.type.startsWith("image/")) return
        const reader = new FileReader()
        reader.addEventListener("load", () => {
            try { importAsCompletedQueueImage(reader.result, file) }
            catch (error) { notify(error.message, true) }
        }, { once: true })
        reader.addEventListener("error", () => notify(`Could not read ${file.name}.`, true), { once: true })
        reader.readAsDataURL(file)
    })

    const setupImageInput = document.getElementById("setup_image_input")
    setupImageInput?.addEventListener("change", async () => {
        const file = setupImageInput.files?.[0]
        setupImageInput.value = ""
        if (!file) return
        try {
            await loadSetupFromImage(file)
        } catch (error) {
            notify(error?.message || `Could not read setup metadata from ${file.name}.`, true)
        }
    })
})()
