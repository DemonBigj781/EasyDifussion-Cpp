(function () {
    "use strict"

    const core = window.PerchancePluginCore
    if (!core || document.getElementById(`${core.ID_PREFIX}-image-panel`)) return

    function attach() {
        const host = core.ensureMainTab()
        if (!host) return false
        const settings = core.loadSettings()
        const panel = document.createElement("section")
        panel.id = `${core.ID_PREFIX}-image-panel`
        panel.className = "panel-box"
        panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/perchance_plugin/perchance-image.plugin.html")
        host.appendChild(panel)

        const prompt = core.element("image-prompt")
        const negative = core.element("negative-prompt")
        const shape = core.element("shape")
        const seed = core.element("seed")
        const guidance = core.element("guidance-scale")
        const generate = core.element("image-button")
        prompt.value = settings.imagePrompt || ""
        negative.value = settings.negativePrompt || ""
        shape.value = settings.shape || "square"
        seed.value = settings.seed || "-1"
        guidance.value = settings.guidanceScale || "7"

        function save() {
            core.saveSettings({
                imagePrompt: prompt.value,
                negativePrompt: negative.value,
                shape: shape.value,
                seed: seed.value,
                guidanceScale: guidance.value,
            })
        }
        ;[prompt, negative, shape, seed, guidance].forEach((field) => {
            field.addEventListener("input", save)
            field.addEventListener("change", save)
        })
        core.element("import-image-prompt").addEventListener("click", () => {
            prompt.value = document.getElementById("prompt")?.value || ""
            negative.value = document.getElementById("negative_prompt")?.value || ""
            save()
            core.setStatus("Imported positive and negative prompts from Easy Diffusion.")
        })
        core.element("export-image-prompt").addEventListener("click", () => {
            const easyPrompt = document.getElementById("prompt")
            const easyNegative = document.getElementById("negative_prompt")
            if (easyPrompt) {
                easyPrompt.value = prompt.value
                core.dispatchInput(easyPrompt)
            }
            if (easyNegative) {
                easyNegative.value = negative.value
                core.dispatchInput(easyNegative)
            }
            core.setStatus("Exported prompts to Easy Diffusion.")
        })
        generate.addEventListener("click", async () => {
            if (!prompt.value.trim()) {
                core.setStatus("Enter an image prompt first.")
                prompt.focus()
                return
            }
            generate.disabled = true
            save()
            core.setStatus("Generating image with Perchance…")
            try {
                const data = await core.requestJson("/perchance/image", {
                    prompt: prompt.value.trim(),
                    negative_prompt: negative.value,
                    shape: shape.value,
                    seed: seed.value,
                    guidance_scale: guidance.value,
                })
                const image = core.element("image")
                image.src = `${data.url}?t=${Date.now()}`
                const link = core.element("image-link")
                link.href = data.url
                link.textContent = data.path
                core.element("image-result").style.display = "block"
                core.setStatus("Image generated and saved to Easy Diffusion outputs.")
            } catch (error) {
                core.setStatus(`Image generation failed: ${error.message}`)
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
