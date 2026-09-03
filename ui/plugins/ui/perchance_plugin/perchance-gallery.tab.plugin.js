(function () {
    "use strict"

    const core = window.PerchancePluginCore
    if (!core || document.getElementById("tab-perchance-gallery")) return

    function setRating(value) {
        const select = core.element("gallery-content-filter")
        const rating = String(value || "g").trim() || "g"
        select.querySelectorAll("option[data-custom-rating]").forEach((option) => option.remove())
        if (!Array.from(select.options).some((option) => option.value === rating)) {
            const option = document.createElement("option")
            option.value = rating
            option.textContent = `${rating} (custom)`
            option.dataset.customRating = "true"
            select.appendChild(option)
        }
        select.value = rating
    }

    function attach() {
        const host = core.ensureTab("perchance-gallery", "Perchance Gallery", "fa-images")
        if (!host) return false
        host.innerHTML = window.loadRequiredPluginHTML("/plugins/core/perchance_plugin/perchance-gallery.tab.plugin.html")
        const settings = core.loadSettings()
        const fields = {
            id: core.element("gallery-id"),
            channel: core.element("gallery-channel"),
            rating: core.element("gallery-content-filter"),
            limit: core.element("gallery-limit"),
            cursor: core.element("gallery-cursor"),
            sort: core.element("gallery-sort"),
            range: core.element("gallery-time-range"),
            download: core.element("gallery-download"),
            visible: core.element("gallery-visible"),
        }
        fields.id.value = settings.galleryId || ""
        fields.channel.value = settings.galleryChannel || "ai-text-to-image-generator"
        setRating(settings.galleryContentFilter || "g")
        fields.limit.value = settings.galleryLimit || "20"
        fields.cursor.value = settings.galleryCursor || ""
        fields.sort.value = settings.gallerySort || "recent"
        fields.range.value = settings.galleryTimeRange || ""
        fields.download.checked = Boolean(settings.galleryDownload)
        fields.visible.checked = Boolean(settings.galleryVisible)

        function save() {
            core.saveSettings({
                galleryId: fields.id.value,
                galleryChannel: fields.channel.value,
                galleryContentFilter: fields.rating.value,
                galleryLimit: fields.limit.value,
                galleryCursor: fields.cursor.value,
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
                content_filter: fields.rating.value,
                limit: fields.limit.value,
                cursor: fields.cursor.value.trim(),
                sort: fields.sort.value,
                time_range: fields.range.value.trim(),
                download: fields.download.checked,
                visible: fields.visible.checked,
            }
        }

        function render(entries) {
            const results = core.element("gallery-results")
            results.replaceChildren()
            entries.forEach((entry) => {
                const card = document.createElement("article")
                card.style.cssText = "padding:8px;border:1px solid var(--border-color,rgba(127,127,127,.35));border-radius:6px;min-width:0;"
                const imageUrl = entry.local_url || entry.imageUrl || ""
                if (imageUrl) {
                    const image = document.createElement("img")
                    image.src = imageUrl
                    image.alt = "Perchance gallery image"
                    image.loading = "lazy"
                    image.decoding = "async"
                    image.referrerPolicy = "no-referrer"
                    image.addEventListener("error", () => {
                        image.alt = "Perchance gallery image could not be loaded"
                        image.style.minHeight = "80px"
                        core.setStatus("A Perchance gallery image could not be loaded from the local cache.")
                    }, { once: true })
                    image.style.cssText = "display:block;width:100%;height:220px;object-fit:contain;border-radius:4px;background:rgba(0,0,0,.12);"
                    card.appendChild(image)
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
            render(Array.isArray(data.entries) ? data.entries : [])
            fields.cursor.value = data.nextCursor || ""
            save()
            core.setStatus(data.nextCursor ? "Gallery loaded; continuation cursor saved." : "Gallery loaded.")
        }))
        core.element("gallery-get-button").addEventListener("click", () => withBusy("Loading Perchance gallery image…", async () => {
            const request = payload()
            if (!request.gallery_id) throw new Error("Enter a gallery image ID or supported URL.")
            render([await core.requestJson("/perchance/gallery/get", request)])
            core.setStatus("Gallery image loaded.")
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
