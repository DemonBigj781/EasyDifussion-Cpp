/*
 * Make Very Similar Images for Easy Diffusion C++.
 * Adapted from AvidGameFan's v1.3.3 plugin:
 * https://github.com/AvidGameFan/ed-plugins/blob/master/makeverysimilarimages.plugin.js
 */

;(function () {
    "use strict"
    if (window.__easyDiffusionMakeVerySimilarLoaded) return
    window.__easyDiffusionMakeVerySimilarLoaded = true

    const ID = "make-very-similar"
    const STORAGE_KEY = "MakeVerySimilar_Plugin_Settings"
    const defaults = {
        highQuality: false,
        enhanceImage: false,
        addNoise: false,
        preserve: false,
        useChangedPrompt: false,
    }
    const editor = document.getElementById("editor-settings")
    if (!editor?.parentNode || !Array.isArray(PLUGINS?.IMAGE_INFO_BUTTONS)) {
        console.error("Make Very Similar: required Easy Diffusion UI APIs were not found")
        return
    }

    function readSettings() {
        try { return { ...defaults, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") } }
        catch (_) { return { ...defaults } }
    }
    let settings = readSettings()

    const panel = document.createElement("div")
    panel.id = `${ID}-settings`
    panel.className = "settings-box panel-box"
    panel.innerHTML = `
        <h4 class="collapsible">Make Very Similar
            <i id="${ID}-reset" class="fa-solid fa-arrow-rotate-left section-button"><span class="simple-tooltip top-left">Reset settings</span></i>
        </h4>
        <div class="collapsible-content settings-panel-entries">
            <label><input id="${ID}-quality" type="checkbox"> More steps / higher quality</label>
            <label><input id="${ID}-enhance" type="checkbox"> Sharpen and slightly increase contrast</label>
            <label><input id="${ID}-noise" type="checkbox"> Add fine texture noise</label>
            <label><input id="${ID}-preserve" type="checkbox"> Preserve the source more closely</label>
            <label><input id="${ID}-prompt" type="checkbox"> Use the prompt currently in the editor</label>
            <small>Use the <b>Very Similar</b> button below a finished image. Each click creates one img2img variation and omits ControlNet, masks, and prior upscale operations.</small>
        </div>`
    editor.after(panel)
    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const fields = {
        highQuality: document.getElementById(`${ID}-quality`),
        enhanceImage: document.getElementById(`${ID}-enhance`),
        addNoise: document.getElementById(`${ID}-noise`),
        preserve: document.getElementById(`${ID}-preserve`),
        useChangedPrompt: document.getElementById(`${ID}-prompt`),
    }

    function renderSettings() {
        Object.entries(fields).forEach(([key, field]) => { field.checked = Boolean(settings[key]) })
    }
    function saveSettings() {
        Object.entries(fields).forEach(([key, field]) => { settings[key] = field.checked })
        localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
    }
    Object.values(fields).forEach((field) => field.addEventListener("change", saveSettings))
    document.getElementById(`${ID}-reset`).addEventListener("click", (event) => {
        event.stopPropagation()
        settings = { ...defaults }
        renderSettings()
        saveSettings()
    })
    renderSettings()

    const allModelText = (modelName, loras) => [
        String(modelName || ""),
        ...(Array.isArray(loras) ? loras : loras ? [loras] : []),
    ].join(" ")

    function isTurbo(modelName, loras) {
        return /turbo|lightning|hyper|flash|schnell|\blcm\b/i.test(allModelText(modelName, loras))
    }
    function isLightning(modelName, loras) {
        return /lightning|hyper|schnell/i.test(allModelText(modelName, loras))
    }
    function isFluxLike(modelName) {
        const name = String(modelName || "")
        if (name === stableDiffusionModelField?.value
            && ((typeof isFluxModel === "function" && isFluxModel())
                || (typeof isChromaModel === "function" && isChromaModel()))) return true
        return /flux|lyhanime_kor|chroma|sd3|qwen|z[_ -]?image|klein/i.test(name)
    }
    function isXl(modelName) {
        const name = String(modelName || "")
        if (name === stableDiffusionModelField?.value && typeof modelsDB === "object") {
            const tags = modelsDB?.["stable-diffusion"]?.[name]?.tags || []
            if (tags.some((tag) => String(tag).startsWith("sd_xl"))) return true
        }
        return /xl|playground|disneyrealcartoonmix|mobius|zovya/i.test(name) || isFluxLike(name)
    }

    function stepsToUse(defaultSteps, flux, turbo, xl, lightning) {
        let steps = Math.max(1, Number.parseInt(defaultSteps, 10) || 20)
        if (flux) {
            if (turbo) steps = settings.highQuality
                ? Math.min(lightning ? Math.max(8, steps + 2) : Math.max(8, steps + 3), 8)
                : Math.min(lightning ? Math.max(6, steps + 2) : Math.max(7, steps + 3), 7)
            else steps = Math.min(steps + 5, settings.highQuality ? 12 : 15)
        } else if (xl) {
            if (turbo) steps = settings.highQuality
                ? Math.min(lightning ? Math.max(8, steps + 3) : Math.max(10, steps + 3), 10)
                : Math.min(lightning ? Math.max(6, steps + 2) : Math.max(7, steps + 3), 8)
            else steps = Math.min(steps + 5, settings.highQuality ? 20 : 18)
        } else if (turbo) {
            steps = settings.highQuality
                ? Math.min(Math.max(10, steps + 3), 12)
                : Math.min(lightning ? Math.max(6, steps + 2) : Math.max(7, steps + 3), 10)
        } else {
            steps = Math.min(steps + 5, settings.highQuality ? 30 : 20)
        }
        if (settings.preserve) steps *= 2.5
        return Math.max(1, Math.floor(steps))
    }

    function sharpen(context, width, height, amount) {
        const source = context.getImageData(0, 0, width, height)
        const output = context.createImageData(width, height)
        const weights = [0, -1, 0, -1, 5, -1, 0, -1, 0]
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const destination = (y * width + x) * 4
                let red = 0, green = 0, blue = 0
                for (let kernelY = 0; kernelY < 3; kernelY++) {
                    for (let kernelX = 0; kernelX < 3; kernelX++) {
                        const sampleY = Math.min(height - 1, Math.max(0, y + kernelY - 1))
                        const sampleX = Math.min(width - 1, Math.max(0, x + kernelX - 1))
                        const sourceOffset = (sampleY * width + sampleX) * 4
                        const weight = weights[kernelY * 3 + kernelX]
                        red += source.data[sourceOffset] * weight
                        green += source.data[sourceOffset + 1] * weight
                        blue += source.data[sourceOffset + 2] * weight
                    }
                }
                output.data[destination] = red * amount + source.data[destination] * (1 - amount)
                output.data[destination + 1] = green * amount + source.data[destination + 1] * (1 - amount)
                output.data[destination + 2] = blue * amount + source.data[destination + 2] * (1 - amount)
                output.data[destination + 3] = source.data[destination + 3]
            }
        }
        context.putImageData(output, 0, 0)
    }

    function adjustPixels(imageData) {
        const pixels = imageData.data
        const contrast = 0.8 * 2.55
        const factor = (255 + contrast) / (255.01 - contrast)
        for (let offset = 0; offset < pixels.length; offset += 4) {
            if (settings.enhanceImage) {
                pixels[offset] = factor * (pixels[offset] - 128) + 128
                pixels[offset + 1] = factor * (pixels[offset + 1] - 128) + 128
                pixels[offset + 2] = factor * (pixels[offset + 2] - 128) + 128
            }
            if (settings.addNoise && Math.random() >= 0.7) {
                for (let channel = 0; channel < 3; channel++) {
                    let gaussian = 0
                    for (let sample = 0; sample < 6; sample++) gaussian += Math.random()
                    const noise = Math.floor((gaussian / 6) * 128) - 64
                    pixels[offset + channel] = Math.max(0, Math.min(255, pixels[offset + channel] + noise))
                }
            }
        }
        return imageData
    }

    function enhancedImage(image, flux) {
        if (!settings.enhanceImage && !settings.addNoise) return image.currentSrc || image.src
        const canvas = document.createElement("canvas")
        canvas.width = image.naturalWidth
        canvas.height = image.naturalHeight
        const context = canvas.getContext("2d", { willReadFrequently: true })
        if (!context || !canvas.width || !canvas.height) throw new Error("The source image is not ready.")
        context.drawImage(image, 0, 0)
        if (settings.enhanceImage) sharpen(context, canvas.width, canvas.height, flux ? 0.11 : 0.33)
        context.putImageData(adjustPixels(context.getImageData(0, 0, canvas.width, canvas.height)), 0, 0)
        return canvas.toDataURL("image/png")
    }

    function onMakeVerySimilarClick(originalRequest, image) {
        try {
            const modelName = originalRequest.use_stable_diffusion_model || ""
            const turbo = isTurbo(modelName, originalRequest.use_lora_model)
            const lightning = isLightning(modelName, originalRequest.use_lora_model)
            const flux = isFluxLike(modelName)
            const xl = isXl(modelName)
            let promptStrength = (settings.preserve ? 0.3 : 0.7) - (flux ? (/klein/i.test(modelName) ? 0.1 : 0.05) : 0)
            if (originalRequest.scheduler_name === "beta") promptStrength -= 0.04

            const requestChanges = {
                num_outputs: 1,
                num_inference_steps: stepsToUse(originalRequest.num_inference_steps, flux, turbo, xl, lightning),
                prompt_strength: Math.max(0.05, Math.round(promptStrength * 100) / 100),
                init_image: enhancedImage(image, flux),
                seed: Math.floor(Math.random() * (2 ** 32 - 1)),
            }
            if (settings.useChangedPrompt) {
                requestChanges.prompt = document.getElementById("prompt")?.value.trim() || ""
                requestChanges.original_prompt = requestChanges.prompt
                requestChanges.hidden_positive_prompt = document.getElementById("hidden_positive_prompt")?.value.trim() || ""
                requestChanges.hidden_negative_prompt = document.getElementById("hidden_negative_prompt")?.value.trim() || ""
            }

            const task = modifyCurrentRequest(originalRequest, requestChanges)
            task.numOutputsTotal = 1
            task.batchCount = 1
            ;[
                "use_controlnet_model", "control_filter_to_apply", "control_image", "control_alpha",
                "controlnet_union_type", "control_image_preprocessed", "use_upscale", "mask",
            ].forEach((key) => delete task.reqBody[key])
            createTask(task)
        } catch (error) {
            if (typeof showToast === "function") showToast(`Very Similar failed: ${error.message}`, 6000, true)
            else console.error(error)
        }
    }

    PLUGINS.IMAGE_INFO_BUTTONS.push({
        text: "Very Similar",
        on_click: onMakeVerySimilarClick,
        filter: () => true,
    })
})()
