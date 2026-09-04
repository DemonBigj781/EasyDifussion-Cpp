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
        const host = core.ensureTab("perchance-gallery", "Perchance Gallery", "fa-images")
        if (!host) return false
        host.innerHTML = window.loadRequiredPluginHTML("/plugins/core/perchance_plugin/perchance-gallery.tab.plugin.html")
        const settings = core.loadSettings()
        const fields = {
            id: core.element("gallery-id"),
            channel: core.element("gallery-channel"),
            limit: core.element("gallery-limit"),
            sort: core.element("gallery-sort"),
            range: core.element("gallery-time-range"),
            download: core.element("gallery-download"),
            visible: core.element("gallery-visible"),
        }
        fields.id.value = settings.galleryId || ""
        fields.channel.value = settings.galleryChannel || "ai-text-to-image-generator"
        fields.limit.value = settings.galleryLimit || "20"
        fields.sort.value = settings.gallerySort || "recent"
        fields.range.value = settings.galleryTimeRange || ""
        fields.download.checked = Boolean(settings.galleryDownload)
        fields.visible.checked = Boolean(settings.galleryVisible)

        function save() {
            core.saveSettings({
                galleryId: fields.id.value,
                galleryChannel: fields.channel.value,
                galleryLimit: fields.limit.value,
                gallerySort: fields.sort.value,
                galleryTimeRange: fields.range.value,
                galleryDownload: fields.download.checked,
                galleryVisible: fields.visible.checked,
            })
        }
        Object.values(fields).forEach((field) => {
            field.addEventListener("input", save)
            field.addEventListener("change", save)
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
                        save()
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
        core.element("gallery-save-button").addEventListener("click", () => withBusy("Saving Perchance gallery settings…", async () => {
            const data = await core.requestJson("/perchance/settings", {
                gallery_id: fields.id.value.trim(),
                channel: fields.channel.value.trim(),
            })
            fields.id.value = data.gallery_id || ""
            fields.channel.value = data.channel || "ai-text-to-image-generator"
            save()
            core.setStatus("Perchance gallery ID and channel saved.")
        }))
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
            if (data.settings) {
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
