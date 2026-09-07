(function () {
    "use strict"

    const core = window.PerchancePluginCore
    if (!core || document.getElementById("tab-perchance-gallery")) return

    async function loadGalleryImage(image, card, entry) {
        const preview = typeof entry.preview_data_url === "string" ? entry.preview_data_url : ""
        if (!preview.startsWith("data:image/jpeg;base64,")) {
            throw new Error(entry.preview_error || "The server did not return an embedded gallery preview.")
        }

        if (typeof image.decode === "function") {
            image.src = preview
            await image.decode()
        } else {
            await new Promise((resolve, reject) => {
                image.addEventListener("load", resolve, { once: true })
                image.addEventListener("error", () => reject(new Error("The embedded preview could not be decoded.")), { once: true })
                image.src = preview
            })
        }
        if (!image.naturalWidth || !image.naturalHeight) {
            throw new Error("The embedded preview decoded without dimensions.")
        }
        image.style.visibility = "visible"
        card.removeAttribute("aria-busy")
    }

    function attach() {
        const host = core.ensureTab("perchance-gallery", "Perchance Gallery", "fa-globe")
        if (!host) return false
        host.innerHTML = window.loadRequiredPluginHTML("/plugins/core/perchance_plugin/perchance-gallery.tab.plugin.html")
        const settings = core.loadSettings()
        const fields = {
            preset: core.element("gallery-preset"),
            presetName: core.element("gallery-preset-name"),
            id: core.element("gallery-id"),
            channel: core.element("gallery-channel"),
            limit: core.element("gallery-limit"),
            sort: core.element("gallery-sort"),
            range: core.element("gallery-time-range"),
            download: core.element("gallery-download"),
            visible: core.element("gallery-visible"),
        }
        let presets = Array.isArray(settings.galleryPresets)
            ? settings.galleryPresets.filter((preset) => (
                preset &&
                typeof preset.id === "string" &&
                typeof preset.name === "string" &&
                typeof preset.galleryId === "string" &&
                typeof preset.channel === "string"
            )).map((preset) => ({
                id: preset.id,
                name: preset.name.slice(0, 80),
                galleryId: preset.galleryId,
                channel: preset.channel,
            }))
            : []
        fields.id.value = settings.galleryId || ""
        fields.channel.value = settings.galleryChannel || "ai-text-to-image-generator"
        fields.limit.value = settings.galleryLimit || "20"
        fields.sort.value = settings.gallerySort || "recent"
        fields.range.value = settings.galleryTimeRange || ""
        fields.download.checked = Boolean(settings.galleryDownload)
        fields.visible.checked = Boolean(settings.galleryVisible)

        const deletePresetButton = core.element("gallery-delete-preset-button")

        function selectedPreset() {
            return presets.find((preset) => preset.id === fields.preset.value)
        }

        function renderPresets(selectedId = "") {
            const customOption = document.createElement("option")
            customOption.value = ""
            customOption.textContent = "Custom settings"
            fields.preset.replaceChildren(customOption)
            presets.forEach((preset) => {
                const option = document.createElement("option")
                option.value = preset.id
                option.textContent = preset.name
                fields.preset.appendChild(option)
            })
            fields.preset.value = presets.some((preset) => preset.id === selectedId) ? selectedId : ""
            deletePresetButton.disabled = !fields.preset.value
        }

        renderPresets(settings.galleryPresetId || "")
        const initialPreset = selectedPreset()
        fields.presetName.value = initialPreset?.name || settings.galleryPresetName || ""
        if (initialPreset) {
            fields.id.value = initialPreset.galleryId
            fields.channel.value = initialPreset.channel
        }

        function save() {
            core.saveSettings({
                galleryPresetId: fields.preset.value,
                galleryPresetName: fields.presetName.value,
                galleryPresets: presets,
                galleryId: fields.id.value,
                galleryChannel: fields.channel.value,
                galleryLimit: fields.limit.value,
                gallerySort: fields.sort.value,
                galleryTimeRange: fields.range.value,
                galleryDownload: fields.download.checked,
                galleryVisible: fields.visible.checked,
            })
        }

        function detachChangedPreset() {
            const preset = selectedPreset()
            if (preset && (
                preset.galleryId !== fields.id.value ||
                preset.channel !== fields.channel.value
            )) {
                fields.preset.value = ""
                fields.presetName.value = ""
                deletePresetButton.disabled = true
            }
            save()
        }

        ;[fields.id, fields.channel].forEach((field) => {
            field.addEventListener("input", detachChangedPreset)
            field.addEventListener("change", detachChangedPreset)
        })
        ;[fields.presetName, fields.limit, fields.sort, fields.range, fields.download, fields.visible].forEach((field) => {
            field.addEventListener("input", save)
            field.addEventListener("change", save)
        })

        fields.preset.addEventListener("change", () => {
            const preset = selectedPreset()
            if (preset) {
                fields.presetName.value = preset.name
                fields.id.value = preset.galleryId
                fields.channel.value = preset.channel
            } else {
                fields.presetName.value = ""
            }
            deletePresetButton.disabled = !preset
            save()
        })

        deletePresetButton.addEventListener("click", () => {
            const preset = selectedPreset()
            if (!preset) return
            if (!window.confirm(`Delete gallery preset "${preset.name}"?`)) return
            presets = presets.filter((candidate) => candidate.id !== preset.id)
            fields.presetName.value = ""
            renderPresets()
            save()
            core.setStatus(`Gallery preset "${preset.name}" deleted.`)
        })

        function payload() {
            save()
            return {
                gallery_id: fields.id.value.trim(),
                channel: fields.channel.value.trim(),
                content_filter: "none",
                limit: fields.limit.value,
                sort: fields.sort.value,
                time_range: fields.range.value.trim(),
                download: fields.download.checked,
                visible: fields.visible.checked,
            }
        }

        async function render(entries) {
            const results = core.element("gallery-results")
            results.replaceChildren()
            const imageLoads = []
            entries.forEach((entry) => {
                const card = document.createElement("article")
                card.style.cssText = "padding:8px;border:1px solid var(--border-color,rgba(127,127,127,.35));border-radius:6px;min-width:0;"
                if (entry.local_url || entry.preview_data_url || entry.preview_error) {
                    const image = document.createElement("img")
                    image.alt = "Perchance gallery image"
                    image.loading = "eager"
                    image.decoding = "async"
                    image.style.cssText = "display:block;width:100%;height:220px;object-fit:contain;border-radius:4px;background:rgba(0,0,0,.12);visibility:hidden;"
                    card.setAttribute("aria-busy", "true")
                    card.appendChild(image)
                    imageLoads.push(loadGalleryImage(image, card, entry).then(
                        () => null,
                        (error) => {
                            image.style.visibility = "visible"
                            image.alt = "Perchance gallery preview could not be loaded"
                            card.style.outline = "1px solid var(--accent-color, #c66)"
                            card.removeAttribute("aria-busy")
                            return error instanceof Error ? error : new Error(String(error))
                        },
                    ))
                }
                const prompt = document.createElement("div")
                prompt.textContent = entry.prompt || "(No prompt returned)"
                prompt.style.cssText = "font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;margin-top:7px;max-height:120px;overflow:auto;"
                card.appendChild(prompt)
                const actions = document.createElement("div")
                actions.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;"
                const usePrompt = document.createElement("button")
                usePrompt.className = "secondaryButton"
                usePrompt.textContent = "Use Prompt"
                usePrompt.addEventListener("click", () => {
                    const target = document.getElementById("prompt")
                    if (target) {
                        target.value = entry.prompt || ""
                        core.dispatchInput(target)
                    }
                    core.setStatus("Gallery prompt copied to Easy Diffusion.")
                })
                actions.appendChild(usePrompt)
                if (entry.imageId) {
                    const selectId = document.createElement("button")
                    selectId.className = "secondaryButton"
                    selectId.textContent = "Select ID"
                    selectId.addEventListener("click", () => {
                        fields.id.value = entry.imageId
                        detachChangedPreset()
                    })
                    actions.appendChild(selectId)
                }
                card.appendChild(actions)
                results.appendChild(card)
            })
            if (!entries.length) results.textContent = "No gallery entries returned."
            return (await Promise.all(imageLoads)).filter(Boolean)
        }

        async function withBusy(message, callback) {
            const buttons = host.querySelectorAll("button[data-perchance-action]")
            buttons.forEach((button) => { button.disabled = true })
            core.setStatus(message)
            try {
                await callback()
            } catch (error) {
                core.setStatus(error.message)
            } finally {
                buttons.forEach((button) => { button.disabled = false })
            }
        }
        core.element("gallery-save-button").addEventListener("click", async (event) => {
            const name = fields.presetName.value.trim()
            const channel = fields.channel.value.trim()
            if (!name) {
                core.setStatus("Enter a name for this gallery preset.")
                fields.presetName.focus()
                return
            }
            if (!channel) {
                core.setStatus("Enter a generator channel before saving the preset.")
                fields.channel.focus()
                return
            }

            let index = presets.findIndex((preset) => preset.id === fields.preset.value)
            if (index < 0) {
                index = presets.findIndex((preset) => preset.name.toLocaleLowerCase() === name.toLocaleLowerCase())
            }
            const presetId = index >= 0
                ? presets[index].id
                : `gallery-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
            const preset = {
                id: presetId,
                name,
                galleryId: fields.id.value.trim(),
                channel,
            }
            if (index >= 0) presets[index] = preset
            else presets.push(preset)
            fields.presetName.value = name
            renderPresets(presetId)
            save()

            const button = event.currentTarget
            button.disabled = true
            core.setStatus(`Gallery preset "${name}" saved in this browser.`)
            try {
                const data = await core.requestJson("/perchance/settings", {
                    gallery_id: preset.galleryId,
                    channel: preset.channel,
                })
                fields.id.value = data.gallery_id || ""
                fields.channel.value = data.channel || "ai-text-to-image-generator"
                const savedIndex = presets.findIndex((candidate) => candidate.id === presetId)
                presets[savedIndex] = {
                    ...preset,
                    galleryId: fields.id.value,
                    channel: fields.channel.value,
                }
                save()
                core.setStatus(`Gallery preset "${name}" saved and selected.`)
            } catch (error) {
                core.setStatus(`Gallery preset "${name}" was saved in this browser. Server settings were not updated: ${error.message}`)
            } finally {
                button.disabled = false
            }
        })
        core.element("gallery-list-button").addEventListener("click", () => withBusy("Loading Perchance gallery…", async () => {
            const data = await core.requestJson("/perchance/gallery/list", payload())
            const failures = await render(Array.isArray(data.entries) ? data.entries : [])
            core.setStatus(failures.length
                ? `Gallery loaded, but ${failures.length} embedded preview${failures.length === 1 ? "" : "s"} failed: ${failures[0].message}.`
                : "Gallery loaded.")
        }))
        core.element("gallery-get-button").addEventListener("click", () => withBusy("Loading Perchance gallery image…", async () => {
            const request = payload()
            if (!request.gallery_id) throw new Error("Enter a gallery image ID or supported URL.")
            const failures = await render([await core.requestJson("/perchance/gallery/get", request)])
            core.setStatus(failures.length
                ? `Gallery entry loaded, but its embedded preview failed: ${failures[0].message}.`
                : "Gallery image loaded.")
        }))
        core.initializePanel(host)
        core.requestJson("/perchance/status").then((data) => {
            if (data.settings && !selectedPreset()) {
                fields.id.value = data.settings.gallery_id || fields.id.value
                fields.channel.value = data.settings.channel || fields.channel.value
                save()
            }
        }).catch(() => {})
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
