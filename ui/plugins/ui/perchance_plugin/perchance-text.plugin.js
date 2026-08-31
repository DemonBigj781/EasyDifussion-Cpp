(function () {
    "use strict"

    const core = window.PerchancePluginCore
    if (!core || document.getElementById(`${core.ID_PREFIX}-text-panel`)) return

    const ENDING = " State the prompt as is."
    const PRESETS = {
        "general-image-expander": "Expand, optimize, and improve the following image prompt. Return only the finished prompt, with a clear subject, setting, composition, lighting, color, mood, camera perspective, and useful visual details. Preserve the original intent, remove repetition and contradictions, and do not add explanations." + ENDING,
        "danbooru-sdxl": "Expand, optimize, and improve the following image prompt for an SDXL-based model using the Danbooru tag system. Return only one comma-separated list of concise Danbooru-style tags, ordered from the main subject to pose, environment, composition, lighting, and style. Avoid prose and explanations." + ENDING,
        "natural-language-sdxl": "Expand, optimize, and improve the following image prompt for an SDXL-based model using fluent natural language. Return only the final polished prompt as coherent descriptive prose. Preserve the original intent and avoid explanations, repetition, and contradictions." + ENDING,
        "cinematic-scene": "Rewrite and expand the following image prompt as a cinematic scene. Return only the final prompt. Strengthen shot type, camera angle, lens feel, subject placement, depth, lighting, atmosphere, color grading, and mood." + ENDING,
        "character-design": "Expand the following image prompt into a production-ready character design prompt. Return only the final prompt. Add coherent appearance, build, face, hair, expression, pose, clothing, materials, accessories, palette, setting, and lighting details." + ENDING,
        "negative-prompt": "Create a concise optimized negative prompt for an SDXL-based image model. Return only a comma-separated list of relevant unwanted qualities, anatomy errors, composition problems, artifacts, and conflicting elements." + ENDING,
    }

    function attach() {
        const host = core.ensureMainTab()
        if (!host) return false
        const panel = document.createElement("section")
        panel.id = `${core.ID_PREFIX}-text-panel`
        panel.className = "panel-box"
        panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/perchance_plugin/perchance-text.plugin.html")
        host.appendChild(panel)

        const settings = core.loadSettings()
        const fields = {
            prompt: core.element("text-prompt"),
            preset: core.element("text-preset"),
            start: core.element("start-with"),
            stops: core.element("stops"),
            filters: core.element("filters"),
            timeout: core.element("timeout"),
        }
        fields.prompt.value = settings.textPrompt || ""
        fields.preset.value = settings.textPreset || ""
        fields.start.value = settings.startWith || ""
        fields.stops.value = settings.stops || ""
        fields.filters.value = settings.filterStrings || ""
        fields.timeout.value = settings.timeout || ""

        function save() {
            core.saveSettings({
                textPrompt: fields.prompt.value,
                textPreset: fields.preset.value,
                startWith: fields.start.value,
                stops: fields.stops.value,
                filterStrings: fields.filters.value,
                timeout: fields.timeout.value,
            })
        }
        Object.values(fields).forEach((field) => {
            field.addEventListener("input", save)
            field.addEventListener("change", save)
        })
        panel.querySelectorAll("textarea").forEach((textarea) => {
            const resize = () => {
                textarea.style.height = "auto"
                textarea.style.height = `${Math.max(textarea.scrollHeight, 48)}px`
            }
            textarea.addEventListener("input", resize)
            resize()
        })

        core.element("apply-text-preset").addEventListener("click", () => {
            const starter = PRESETS[fields.preset.value]
            if (!starter) {
                core.setStatus("Choose a prompt starter preset first.")
                return
            }
            fields.start.value = starter
            fields.start.closest("details").open = true
            save()
            core.setStatus("Prompt starter inserted.")
        })
        core.element("import-text-prompt").addEventListener("click", () => {
            fields.prompt.value = document.getElementById("prompt")?.value || ""
            save()
            core.setStatus("Imported the Easy Diffusion prompt.")
        })
        core.element("export-text-prompt").addEventListener("click", () => {
            const target = document.getElementById("prompt")
            if (target) {
                target.value = fields.prompt.value
                core.dispatchInput(target)
            }
            core.setStatus("Exported the text prompt to Easy Diffusion.")
        })

        const output = core.element("text-output")
        const result = core.element("text-result")
        const errorBox = core.element("text-error")
        core.element("apply-text").addEventListener("click", () => {
            const target = document.getElementById("prompt")
            if (target) {
                target.value = output.value
                core.dispatchInput(target)
            }
        })
        core.element("apply-negative-text").addEventListener("click", () => {
            const target = document.getElementById("negative_prompt")
            if (target) {
                target.value = output.value
                core.dispatchInput(target)
            }
        })
        core.element("copy-text").addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(output.value)
                core.setStatus("Generated text copied.")
            } catch (error) {
                core.setStatus(`Copy failed: ${error.message}`)
            }
        })

        const generate = core.element("text-button")
        generate.addEventListener("click", async () => {
            const prompt = fields.prompt.value.trim()
            if (!prompt) {
                core.setStatus("Enter a text prompt first.")
                fields.prompt.focus()
                return
            }
            generate.disabled = true
            save()
            core.setStatus("Generating text with Perchance…")
            const starter = fields.start.value.trim()
            const payload = {
                prompt: starter ? `${starter}\n\n${prompt}` : prompt,
                start_with: "",
                stop: fields.stops.value.split(/\r?\n/).map((value) => value.trim()).filter(Boolean),
            }
            if (fields.timeout.value.trim()) payload.timeout_ms = fields.timeout.value.trim()
            try {
                const data = await core.requestJson("/perchance/text", payload)
                let text = String(data.text || "")
                for (const filter of fields.filters.value.split(/\r?\n/).map((value) => value.trim()).filter(Boolean)) {
                    const candidate = text.split(filter).join("")
                    if (candidate.trim()) text = candidate
                }
                if (!text.trim()) throw new Error("Perchance returned an empty text response.")
                output.value = text
                errorBox.style.display = "none"
                output.style.display = "block"
                result.style.display = "block"
                core.setStatus("Text generated.")
            } catch (error) {
                errorBox.textContent = `Text generation failed: ${error.message}`
                errorBox.style.display = "block"
                result.style.display = "block"
                core.setStatus(errorBox.textContent)
            } finally {
                generate.disabled = false
            }
        })
        core.initializePanel(panel)
        return true
    }

    if (!attach()) {
        let attempts = 0
        const timer = setInterval(() => {
            attempts += 1
            if (attach() || attempts >= 60) clearInterval(timer)
        }, 500)
    }
})()
