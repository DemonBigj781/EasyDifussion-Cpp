/**
 * LoRA Shuttle Controls, ported to the Easy Diffusion 4.x multi-LoRA API.
 * Original plugin by The Stig.
 */
(function () {
    "use strict"
    if (window.__easyDiffusionLoraShuttleLoaded) return
    window.__easyDiffusionLoraShuttleLoaded = true

    const STEP = 0.1
    const clamp = (value) => Math.max(0, Math.min(1, Number(value) || 0))
    const rounded = (value) => Number(clamp(value).toFixed(2))

    function requestWeights(request) {
        const weights = Array.isArray(request.lora_alpha) ? request.lora_alpha : [request.lora_alpha]
        return weights.filter((value) => value !== undefined && value !== null).map((value) => rounded(value))
    }

    function queueWithWeights(originalRequest, weights) {
        if (!originalRequest.use_lora_model || weights.length === 0) return
        const newTask = modifyCurrentRequest(originalRequest, {
            lora_alpha: Array.isArray(originalRequest.lora_alpha) ? weights : weights[0],
            num_outputs: 1,
        })
        newTask.numOutputsTotal = 1
        newTask.batchCount = 1
        createTask(newTask)
    }

    function setAll(value) {
        return function (originalRequest) {
            queueWithWeights(originalRequest, requestWeights(originalRequest).map(() => rounded(value)))
        }
    }

    function shiftAll(delta) {
        return function (originalRequest) {
            queueWithWeights(originalRequest, requestWeights(originalRequest).map((value) => rounded(value + delta)))
        }
    }

    function queueGrid(originalRequest) {
        const weights = requestWeights(originalRequest)
        if (weights.length === 0) return
        for (let value = 0; value <= 1.0001; value += STEP) {
            queueWithWeights(originalRequest, weights.map(() => rounded(value)))
        }
    }

    const hasLora = (request) => Boolean(request.use_lora_model) && requestWeights(request).length > 0
    PLUGINS.IMAGE_INFO_BUTTONS.push([
        { html: '<span class="imageadjust-label">LoRA:</span>', type: "label", filter: hasLora },
        { html: '<i class="fa-solid fa-fast-backward"></i>', on_click: setAll(0), filter: hasLora },
        { html: '<i class="fa-solid fa-step-backward"></i>', on_click: shiftAll(-STEP), filter: hasLora },
        { html: '<i class="fa-solid fa-compress"></i>', on_click: setAll(0.5), filter: hasLora },
        { html: '<i class="fa-solid fa-step-forward"></i>', on_click: shiftAll(STEP), filter: hasLora },
        { html: '<i class="fa-solid fa-fast-forward"></i>', on_click: setAll(1), filter: hasLora },
        { html: '<i class="fa-solid fa-border-none"></i>', on_click: queueGrid, filter: hasLora },
    ])
})()
