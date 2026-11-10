(() => {
    "use strict"

    async function requestJson(url, options = {}) {
        const response = await fetch(url, { cache: "no-store", ...options })
        let payload = {}
        try {
            payload = await response.json()
        } catch (_) {}
        if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`)
        return payload
    }

    function change(field, value) {
        field.value = String(value)
        field.dispatchEvent(new Event("input", { bubbles: true }))
        field.dispatchEvent(new Event("change", { bubbles: true }))
    }

    function modelName(path) {
        return String(path || "").split(/[\\/]/).filter(Boolean).at(-1) || "unconfigured model"
    }

    function initMain() {
        const panel = document.getElementById("local-llm-prompt-panel")
        if (!panel) return
        const mainPrompt = document.getElementById("prompt")
        const source = document.getElementById("local-llm-source-prompt")
        const instruction = document.getElementById("local-llm-instruction")
        const output = document.getElementById("local-llm-output")
        const status = document.getElementById("local-llm-main-status")
        const processButton = document.getElementById("local-llm-process")
        const applyButton = document.getElementById("local-llm-apply")
        const loadButton = document.getElementById("local-llm-load-prompt")
        if (!mainPrompt || !source || !instruction || !output || !status || !processButton || !applyButton || !loadButton) return

        const loadCurrent = () => change(source, mainPrompt.value)
        loadButton.addEventListener("click", loadCurrent)
        if (panel.matches("details")) {
            panel.addEventListener("toggle", () => {
                if (panel.open && !source.value.trim()) loadCurrent()
            })
        }
        loadCurrent()

        requestJson("/local-llm/health").then((health) => {
            status.textContent = health.ready
                ? `Ready: ${modelName(health.model_path)} on ${String(health.device).toUpperCase()}. The model loads on first use.`
                : `Not ready: ${health.detail}`
            processButton.disabled = !health.ready
        }).catch((error) => {
            status.textContent = `Local LLM status unavailable: ${error.message}`
            processButton.disabled = true
        })

        processButton.addEventListener("click", async () => {
            if (!source.value.trim()) loadCurrent()
            processButton.disabled = true
            applyButton.disabled = true
            status.textContent = "Processing prompt locally…"
            try {
                const result = await requestJson("/local-llm/process", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt: source.value, instruction: instruction.value }),
                })
                output.value = result.processed_prompt
                applyButton.disabled = !output.value.trim()
                status.textContent = `Processed with ${modelName(result.model_path)} on ${String(result.device).toUpperCase()}.`
            } catch (error) {
                status.textContent = `Prompt processing failed: ${error.message}`
            } finally {
                processButton.disabled = false
            }
        })

        applyButton.addEventListener("click", () => {
            if (!output.value.trim()) return
            change(mainPrompt, output.value)
            change(source, output.value)
            status.textContent = "The processed text replaced the main prompt."
        })
    }

    function initSettings() {
        const root = document.getElementById("local-llm-settings")
        if (!root) return
        const status = document.getElementById("local-llm-settings-status")
        const model = document.getElementById("local-llm-model-path")
        const models = document.getElementById("local-llm-model-options")
        const device = document.getElementById("local-llm-device")
        const gpuLayers = document.getElementById("local-llm-gpu-layers")
        const saveButton = document.getElementById("local-llm-save-settings")
        const fields = {
            context_length: document.getElementById("local-llm-context-length"),
            max_tokens: document.getElementById("local-llm-max-tokens"),
            temperature: document.getElementById("local-llm-temperature"),
            top_p: document.getElementById("local-llm-top-p"),
            top_k: document.getElementById("local-llm-top-k"),
            repeat_penalty: document.getElementById("local-llm-repeat-penalty"),
        }
        if (!status || !model || !models || !device || !gpuLayers || !saveButton || Object.values(fields).some((field) => !field)) return
        let lastGpuLayers = 999

        function applyDevice() {
            const cpu = device.value === "cpu"
            if (cpu) {
                if (Number(gpuLayers.value) > 0) lastGpuLayers = Number(gpuLayers.value)
                gpuLayers.value = "0"
            } else if (Number(gpuLayers.value) === 0) {
                gpuLayers.value = String(lastGpuLayers)
            }
            gpuLayers.disabled = cpu
        }

        function populate(settings) {
            model.value = settings.model_path || ""
            device.value = settings.device === "cuda" ? "cuda" : "cpu"
            gpuLayers.value = String(settings.gpu_layers ?? 0)
            if (Number(settings.gpu_layers) > 0) lastGpuLayers = Number(settings.gpu_layers)
            for (const [key, field] of Object.entries(fields)) field.value = String(settings[key])
            models.replaceChildren()
            for (const item of settings.models || []) {
                const option = document.createElement("option")
                option.value = item.path
                option.label = `${item.relative_path} (${(item.size / (1024 ** 3)).toFixed(2)} GiB)`
                models.append(option)
            }
            applyDevice()
            const cuda = settings.cuda_available ? "CUDA detected" : "CUDA unavailable"
            status.textContent = settings.valid
                ? `Ready to save or use. ${cuda}; ${settings.models?.length || 0} text GGUF model(s) found.`
                : `Configuration needs attention: ${settings.error || settings.backend?.detail || "unknown error"}`
        }

        device.addEventListener("change", applyDevice)
        requestJson("/local-llm/settings").then(populate).catch((error) => {
            status.textContent = `Could not load local LLM settings: ${error.message}`
        })

        saveButton.addEventListener("click", async () => {
            const payload = {
                model_path: model.value.trim(),
                device: device.value,
                gpu_layers: Number(gpuLayers.value),
            }
            for (const [key, field] of Object.entries(fields)) payload[key] = Number(field.value)
            status.textContent = "Saving local LLM settings…"
            try {
                populate(await requestJson("/local-llm/settings", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }))
                status.textContent = "Local LLM settings saved. A loaded model will restart on the next prompt request."
            } catch (error) {
                status.textContent = `Local LLM settings were not saved: ${error.message}`
            }
        })
    }

    initMain()
    initSettings()
})()
