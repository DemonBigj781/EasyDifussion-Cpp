;(function () {
    "use strict"
    if (window.__spriteGPTPluginLoaded) return
    window.__spriteGPTPluginLoaded = true

    const model = document.getElementById("stable_diffusion_model")
    const note = document.createElement("tr")
    const description = document.createElement("td")
    description.colSpan = 2
    note.appendChild(description)
    note.id = "sprite-gpt-model-note"
    note.hidden = true
    description.textContent = "Sprite-GPT generates 64×64 images with its bundled text encoder and Euler sampler. Initial/reference images, LoRAs, ControlNet and tiling are not supported."
    model.closest("tr").after(note)
    const fixed = { width: "64", height: "64", sampler_name: "euler", scheduler_name: "simple" }
    const previous = new Map()

    function isSprite(name) {
        return (modelsDB?.["stable-diffusion"]?.[name]?.tags || []).includes("sprite_gpt")
    }

    function update() {
        const active = isSprite(stableDiffusionModelField.value)
        note.hidden = !active
        for (const [id, value] of Object.entries(fixed)) {
            const field = document.getElementById(id)
            if (!field) continue
            if (active) {
                if (!previous.has(id)) previous.set(id, { value: field.value, disabled: field.disabled })
                if (field.tagName === "SELECT" && !Array.from(field.options).some((option) => option.value === value)) {
                    field.add(new Option(value, value))
                }
                field.value = value
                field.disabled = true
            } else if (previous.has(id)) {
                const saved = previous.get(id)
                field.value = saved.value
                field.disabled = saved.disabled
                previous.delete(id)
            }
        }
        if (active) document.getElementById("small_image_warning")?.classList.add("displayNone")
    }

    model.addEventListener("change", update)
    document.addEventListener("refreshModels", update)
    PLUGINS.TASK_BUILD.push(function ({ reqBody }) {
        if (!isSprite(reqBody.use_stable_diffusion_model)) return
        reqBody.width = reqBody.height = 64
        reqBody.sampler_name = "euler"
        reqBody.scheduler_name = "simple"
        reqBody.use_vae_model = ""
        reqBody.use_text_encoder_model = []
        reqBody.clip_skip = false
        reqBody.enable_vae_tiling = false
    })
    update()
})()
