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
})()
