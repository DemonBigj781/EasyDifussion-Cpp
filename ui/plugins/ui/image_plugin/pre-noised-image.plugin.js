// Deterministic, weighted-palette pixel noise Initial Image generator.

;(function () {
    "use strict"
    if (window.__pureNoiseImagePluginLoaded) return
    window.__pureNoiseImagePluginLoaded = true

    const STATE_KEY = "easy-diffusion-pure-noise-image-v1"
    const anchor = document.getElementById("initial-image-settings")
    if (!anchor?.parentNode || typeof window.loadRequiredPluginHTML !== "function") {
        console.error("Pure Noise Image plugin: Initial Image UI is unavailable")
        return
    }

    const panel = document.createElement("div")
    panel.id = "pure-noise-image-panel"
    panel.className = "settings-box panel-box gated-feature"
    panel.dataset.featureKeys = "backend_sdkit3"
    panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/image_plugin/pre-noised-image.plugin.html")
    anchor.after(panel)

    if (!document.getElementById("pure-noise-image-style")) {
        const style = document.createElement("style")
        style.id = "pure-noise-image-style"
        style.textContent = `
            .pure-noise-explanation { margin-top: 0; }
            .pure-noise-palette { display: grid; grid-template-columns: repeat(2, minmax(180px, 1fr)); gap: 7px 12px; }
            .pure-noise-color-row { display: grid; grid-template-columns: auto 52px minmax(64px, 1fr); gap: 7px; align-items: center; }
            .pure-noise-color-row input[type="color"] { width: 48px; height: 30px; padding: 1px; }
            .pure-noise-color-row input[type="number"] { width: 100%; }
            .pure-noise-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 12px; }
            #pure-noise-seed { width: 116px; }
            #pure-noise-preview { display: block; width: min(100%, 256px); height: auto; image-rendering: pixelated; margin: 12px auto 6px; border: 1px solid #555; }
            @media (max-width: 700px) { .pure-noise-palette { grid-template-columns: 1fr; } }
        `
        document.head.appendChild(style)
    }
    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)

    const defaults = [
        ["#000000", 1], ["#ffffff", 1], ["#ff0000", 0], ["#00ff00", 0],
        ["#0000ff", 0], ["#ffff00", 0], ["#00ffff", 0], ["#ff00ff", 0],
    ]

    function readState() {
        try { return JSON.parse(localStorage.getItem(STATE_KEY) || "{}") }
        catch (_) { return {} }
    }

    const state = readState()
    const savedPalette = Array.isArray(state.palette) ? state.palette : defaults
    const paletteContainer = document.getElementById("pure-noise-palette")
    const colorRows = []

    for (let index = 0; index < 8; index += 1) {
        const saved = savedPalette[index] || defaults[index]
        const row = document.createElement("div")
        row.className = "pure-noise-color-row"
        const label = document.createElement("label")
        label.textContent = `Color ${index + 1}`
        const color = document.createElement("input")
        color.type = "color"
        color.value = /^#[0-9a-f]{6}$/i.test(String(saved[0])) ? saved[0] : defaults[index][0]
        color.setAttribute("aria-label", `Noise color ${index + 1}`)
        const weight = document.createElement("input")
        weight.type = "number"
        weight.min = "0"
        weight.step = "0.05"
        weight.value = Number.isFinite(Number(saved[1])) ? Math.max(0, Number(saved[1])) : defaults[index][1]
        weight.setAttribute("aria-label", `Noise color ${index + 1} weight`)
        row.append(label, color, weight)
        paletteContainer.appendChild(row)
        colorRows.push({ color, weight })
    }

    const seedInput = document.getElementById("pure-noise-seed")
    const randomSeed = document.getElementById("pure-noise-random-seed")
    seedInput.value = String(Number.isFinite(Number(state.seed)) ? Number(state.seed) >>> 0 : 1)
    randomSeed.checked = state.random !== false

    function saveState() {
        localStorage.setItem(STATE_KEY, JSON.stringify({
            palette: colorRows.map(({ color, weight }) => [color.value, Math.max(0, Number(weight.value) || 0)]),
            seed: Number(seedInput.value) >>> 0,
            random: randomSeed.checked,
        }))
    }

    function parseColor(hex) {
        return [
            Number.parseInt(hex.slice(1, 3), 16),
            Number.parseInt(hex.slice(3, 5), 16),
            Number.parseInt(hex.slice(5, 7), 16),
        ]
    }

    function makeRandom(seed) {
        let value = (seed >>> 0) || 0x9e3779b9
        return function () {
            value ^= value << 13
            value ^= value >>> 17
            value ^= value << 5
            return (value >>> 0) / 4294967296
        }
    }

    function selectedPalette() {
        return colorRows
            .map(({ color, weight }) => ({ rgb: parseColor(color.value), weight: Math.max(0, Number(weight.value) || 0) }))
            .filter((entry) => entry.weight > 0)
    }

    function createNoise(canvas, width, height, palette, seed) {
        canvas.width = width
        canvas.height = height
        const context = canvas.getContext("2d", { alpha: false })
        const pixels = context.createImageData(width, height)
        const random = makeRandom(seed)
        const totalWeight = palette.reduce((sum, entry) => sum + entry.weight, 0)
        const cumulative = []
        palette.reduce((sum, entry) => {
            const next = sum + entry.weight
            cumulative.push(next)
            return next
        }, 0)
        for (let offset = 0; offset < pixels.data.length; offset += 4) {
            const choice = random() * totalWeight
            let index = cumulative.findIndex((limit) => choice < limit)
            if (index < 0) index = palette.length - 1
            const [red, green, blue] = palette[index].rgb
            pixels.data[offset] = red
            pixels.data[offset + 1] = green
            pixels.data[offset + 2] = blue
            pixels.data[offset + 3] = 255
        }
        context.putImageData(pixels, 0, 0)
    }

    function generate() {
        const palette = selectedPalette()
        if (!palette.length) {
            document.getElementById("pure-noise-status").textContent = "Give at least one color a weight above zero."
            return
        }
        if (randomSeed.checked) {
            const value = new Uint32Array(1)
            crypto.getRandomValues(value)
            seedInput.value = String(value[0])
        }
        const width = Math.max(128, Math.min(2048, Number(document.getElementById("width")?.value) || 512))
        const height = Math.max(128, Math.min(2048, Number(document.getElementById("height")?.value) || 512))
        const seed = Number(seedInput.value) >>> 0
        const canvas = document.getElementById("pure-noise-preview")
        createNoise(canvas, width, height, palette, seed)
        setInitialImageSource(canvas.toDataURL("image/png"), false)
        document.getElementById("pure-noise-status").textContent =
            `Created ${width}×${height} pure pixel noise from ${palette.length} weighted color${palette.length === 1 ? "" : "s"}, seed ${seed}.`
        saveState()
    }

    panel.querySelectorAll("input").forEach((input) => input.addEventListener("change", saveState))
    document.getElementById("pure-noise-generate").addEventListener("click", generate)
})()
