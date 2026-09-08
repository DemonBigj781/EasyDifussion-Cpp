var editorControlsLeft = document.getElementById("image-editor-controls-left")

const IMAGE_EDITOR_MAX_UPSCALE = 2
const IMAGE_EDITOR_MIN_VISIBLE = 48

const IMAGE_EDITOR_BUTTONS = [
    {
        name: "Cancel",
        icon: "fa-regular fa-circle-xmark",
        handler: (editor) => {
            editor.hide()
        },
    },
    {
        name: "Save",
        icon: "fa-solid fa-floppy-disk",
        handler: (editor) => {
            editor.saveImage()
        },
    },
]

const defaultToolBegin = (editor, ctx, x, y, is_overlay = false) => {
    ctx.beginPath()
    ctx.moveTo(x, y)
}
const defaultToolMove = (editor, ctx, x, y, is_overlay = false) => {
    ctx.lineTo(x, y)
    if (is_overlay) {
        ctx.clearRect(0, 0, editor.width, editor.height)
        ctx.stroke()
    }
}
const defaultToolEnd = (editor, ctx, x, y, is_overlay = false) => {
    ctx.stroke()
    if (is_overlay) {
        ctx.clearRect(0, 0, editor.width, editor.height)
    }
}
const toolDoNothing = (editor, ctx, x, y, is_overlay = false) => { }

function magicWandSelect(editor, targetCtx, x, y) {
    const width = editor.width
    const height = editor.height
    const startX = Math.max(0, Math.min(width - 1, Math.floor(x)))
    const startY = Math.max(0, Math.min(height - 1, Math.floor(y)))
    const source = editor.layers.background.ctx.getImageData(0, 0, width, height).data
    const selected = new Uint8Array(width * height)
    const queue = new Int32Array(width * height)
    const start = startY * width + startX
    const baseOffset = start * 4
    const baseR = source[baseOffset]
    const baseG = source[baseOffset + 1]
    const baseB = source[baseOffset + 2]
    const threshold = Number(editor.options.wand_threshold || 30)
    const thresholdSquared = threshold * threshold
    let read = 0
    let write = 0

    selected[start] = 1
    queue[write++] = start
    while (read < write) {
        const pixel = queue[read++]
        const px = pixel % width
        const py = Math.floor(pixel / width)
        const neighbors = [pixel - 1, pixel + 1, pixel - width, pixel + width]
        for (let i = 0; i < neighbors.length; i++) {
            const next = neighbors[i]
            if (next < 0 || next >= selected.length || selected[next]) continue
            if ((i === 0 && px === 0) || (i === 1 && px === width - 1)) continue
            const offset = next * 4
            const dr = source[offset] - baseR
            const dg = source[offset + 1] - baseG
            const db = source[offset + 2] - baseB
            if (dr * dr + dg * dg + db * db > thresholdSquared) continue
            selected[next] = 1
            queue[write++] = next
        }
    }

    // Grow the selected region by the requested pixel buffer. A multi-source
    // breadth-first pass keeps this linear even for large mobile images.
    const buffer = Math.max(0, Math.floor(Number(editor.options.selection_buffer || 0)))
    if (buffer > 0) {
        const distance = new Uint16Array(width * height)
        read = 0
        for (let i = 0; i < selected.length; i++) {
            if (selected[i]) {
                distance[i] = 1
                queue[read++] = i
            }
        }
        write = read
        read = 0
        while (read < write) {
            const pixel = queue[read++]
            const px = pixel % width
            const depth = distance[pixel]
            if (depth > buffer) continue
            const neighbors = [pixel - 1, pixel + 1, pixel - width, pixel + width]
            for (let i = 0; i < neighbors.length; i++) {
                const next = neighbors[i]
                if (next < 0 || next >= selected.length || distance[next]) continue
                if ((i === 0 && px === 0) || (i === 1 && px === width - 1)) continue
                distance[next] = depth + 1
                selected[next] = 1
                queue[write++] = next
            }
        }
    }

    const color = editor.inpainter ? { r: 255, g: 255, b: 255 } : hexToRgb(editor.options.color || "#ffffff")
    const alpha = Math.round(255 * (1 - Number(editor.options.opacity || 0)))
    const mask = new ImageData(width, height)
    for (let i = 0; i < selected.length; i++) {
        if (!selected[i]) continue
        const offset = i * 4
        mask.data[offset] = color.r
        mask.data[offset + 1] = color.g
        mask.data[offset + 2] = color.b
        mask.data[offset + 3] = alpha
    }

    const temporary = document.createElement("canvas")
    temporary.width = width
    temporary.height = height
    temporary.getContext("2d").putImageData(mask, 0, 0)
    targetCtx.save()
    targetCtx.globalCompositeOperation = "source-over"
    targetCtx.globalAlpha = 1
    const feather = Math.floor(Number(editor.options.sharpness || 0) * Number(editor.options.brush_size || 48))
    targetCtx.filter = feather > 0 ? `blur(${feather}px)` : "none"
    targetCtx.drawImage(temporary, 0, 0)
    targetCtx.restore()
}

function expandBinaryMask(mask, width, height, radius) {
    radius = Math.max(0, Math.floor(radius))
    if (!radius) return mask
    const queue = new Int32Array(width * height)
    const distance = new Uint16Array(width * height)
    let read = 0
    let write = 0
    for (let i = 0; i < mask.length; i++) {
        if (mask[i]) {
            distance[i] = 1
            queue[write++] = i
        }
    }
    while (read < write) {
        const pixel = queue[read++]
        const x = pixel % width
        const depth = distance[pixel]
        if (depth > radius) continue
        const neighbors = [pixel - 1, pixel + 1, pixel - width, pixel + width]
        for (let i = 0; i < neighbors.length; i++) {
            const next = neighbors[i]
            if (next < 0 || next >= mask.length || distance[next]) continue
            if ((i === 0 && x === 0) || (i === 1 && x === width - 1)) continue
            distance[next] = depth + 1
            mask[next] = 1
            queue[write++] = next
        }
    }
    return mask
}

function removeImageBackground(editor) {
    const width = editor.width
    const height = editor.height
    const pixels = editor.layers.background.ctx.getImageData(0, 0, width, height).data
    const threshold = Number(editor.options.wand_threshold || 30)
    const thresholdSquared = threshold * threshold
    const background = new Uint8Array(width * height)
    const visited = new Uint8Array(width * height)
    const queue = new Int32Array(width * height)
    const seeds = [0, width - 1, (height - 1) * width, width * height - 1]

    seeds.forEach((seed) => {
        if (visited[seed]) return
        const seedOffset = seed * 4
        const baseR = pixels[seedOffset]
        const baseG = pixels[seedOffset + 1]
        const baseB = pixels[seedOffset + 2]
        let read = 0
        let write = 0
        queue[write++] = seed
        visited[seed] = 1
        while (read < write) {
            const pixel = queue[read++]
            background[pixel] = 1
            const x = pixel % width
            const neighbors = [pixel - 1, pixel + 1, pixel - width, pixel + width]
            for (let i = 0; i < neighbors.length; i++) {
                const next = neighbors[i]
                if (next < 0 || next >= background.length || visited[next]) continue
                if ((i === 0 && x === 0) || (i === 1 && x === width - 1)) continue
                const offset = next * 4
                const dr = pixels[offset] - baseR
                const dg = pixels[offset + 1] - baseG
                const db = pixels[offset + 2] - baseB
                if (dr * dr + dg * dg + db * db > thresholdSquared) continue
                visited[next] = 1
                queue[write++] = next
            }
        }
    })

    expandBinaryMask(background, width, height, Number(editor.options.selection_buffer || 0))
    const maskCanvas = document.createElement("canvas")
    maskCanvas.width = width
    maskCanvas.height = height
    const maskCtx = maskCanvas.getContext("2d")
    const maskData = maskCtx.createImageData(width, height)
    for (let i = 0; i < background.length; i++) {
        if (!background[i]) continue
        const offset = i * 4
        maskData.data[offset] = 255
        maskData.data[offset + 1] = 255
        maskData.data[offset + 2] = 255
        maskData.data[offset + 3] = 255
    }
    maskCtx.putImageData(maskData, 0, 0)

    const target = editor.inpainter ? editor.layers.drawing.ctx : editor.layers.background.ctx
    target.save()
    target.globalAlpha = 1
    target.filter = `blur(${Math.floor(Number(editor.options.sharpness || 0) * Number(editor.options.brush_size || 48))}px)`
    target.globalCompositeOperation = editor.inpainter ? "source-over" : "destination-out"
    target.drawImage(maskCanvas, 0, 0)
    target.restore()
    editor.setBrush()
}

const IMAGE_EDITOR_TOOLS = [
    {
        id: "pan",
        name: "Hand",
        icon: "fa-solid fa-hand",
        cursor: "grab",
        begin: toolDoNothing,
        move: toolDoNothing,
        end: toolDoNothing,
        hotkey: "h",
    },
    {
        id: "draw",
        name: "Draw",
        icon: "fa-solid fa-pencil",
        cursor: "url(/media/images/fa-pencil.svg) 0 24, pointer",
        begin: defaultToolBegin,
        move: defaultToolMove,
        end: defaultToolEnd,
        hotkey: "d",
    },
    {
        id: "erase",
        name: "Erase",
        icon: "fa-solid fa-eraser",
        cursor: "url(/media/images/fa-eraser.svg) 0 14, pointer",
        begin: defaultToolBegin,
        move: (editor, ctx, x, y, is_overlay = false) => {
            ctx.lineTo(x, y)
            if (is_overlay) {
                ctx.clearRect(0, 0, editor.width, editor.height)
                ctx.globalCompositeOperation = "source-over"
                ctx.globalAlpha = 1
                ctx.filter = "none"
                ctx.drawImage(editor.canvas_current, 0, 0)
                editor.setBrush(editor.layers.overlay)
                ctx.stroke()
                editor.canvas_current.style.opacity = 0
            }
        },
        end: (editor, ctx, x, y, is_overlay = false) => {
            ctx.stroke()
            if (is_overlay) {
                ctx.clearRect(0, 0, editor.width, editor.height)
                editor.canvas_current.style.opacity = ""
            }
        },
        setBrush: (editor, layer) => {
            layer.ctx.globalCompositeOperation = "destination-out"
        },
        hotkey: "e",
    },
    {
        id: "fill",
        name: "Fill",
        icon: "fa-solid fa-fill",
        cursor: "url(/media/images/fa-fill.svg) 20 6, pointer",
        begin: (editor, ctx, x, y, is_overlay = false) => {
            if (!is_overlay) {
                var color = hexToRgb(ctx.fillStyle)
                color.a = parseInt(ctx.globalAlpha * 255) // layer.ctx.globalAlpha
                flood_fill(editor, ctx, parseInt(x), parseInt(y), color)
            }
        },
        move: toolDoNothing,
        end: toolDoNothing,
        hotkey: "f",
    },
    {
        id: "magicwand",
        name: "Magic Wand",
        icon: "fa-solid fa-wand-magic-sparkles",
        cursor: "crosshair",
        begin: (editor, ctx, x, y, is_overlay = false) => {
            if (!is_overlay) magicWandSelect(editor, ctx, x, y)
        },
        move: toolDoNothing,
        end: toolDoNothing,
        hotkey: "m",
    },
    {
        id: "colorpicker",
        name: "Picker",
        icon: "fa-solid fa-eye-dropper",
        cursor: "url(/media/images/fa-eye-dropper.svg) 0 24, pointer",
        begin: (editor, ctx, x, y, is_overlay = false) => {
            if (!is_overlay) {
                var img_rgb = editor.layers.background.ctx.getImageData(x, y, 1, 1).data
                var drawn_rgb = editor.ctx_current.getImageData(x, y, 1, 1).data
                var drawn_opacity = drawn_rgb[3] / 255
                editor.custom_color_input.value = rgbToHex({
                    r: drawn_rgb[0] * drawn_opacity + img_rgb[0] * (1 - drawn_opacity),
                    g: drawn_rgb[1] * drawn_opacity + img_rgb[1] * (1 - drawn_opacity),
                    b: drawn_rgb[2] * drawn_opacity + img_rgb[2] * (1 - drawn_opacity),
                })
                editor.custom_color_input.dispatchEvent(new Event("change"))
            }
        },
        move: toolDoNothing,
        end: toolDoNothing,
        hotkey: "p",
    },
]

const IMAGE_EDITOR_ACTIONS = [
    {
        id: "load_mask",
        name: "Load mask from file",
        className: "load_mask",
        icon: "fa-regular fa-folder-open",
        handler: (editor) => {
            let el = document.createElement("input")
            el.setAttribute("type", "file")
            el.addEventListener("change", function () {
                if (this.files.length === 0) {
                    return
                }

                let reader = new FileReader()
                let file = this.files[0]

                reader.addEventListener("load", function (event) {
                    let maskData = reader.result

                    editor.layers.drawing.ctx.clearRect(0, 0, editor.width, editor.height)
                    var image = new Image()
                    image.onload = () => {
                        editor.layers.drawing.ctx.drawImage(image, 0, 0, editor.width, editor.height)
                    }
                    image.src = maskData
                })

                if (file) {
                    reader.readAsDataURL(file)
                }
            })

            el.click()
        },
        trackHistory: true,
    },
    {
        id: "fill_all",
        name: "Fill all",
        icon: "fa-solid fa-paint-roller",
        handler: (editor) => {
            editor.ctx_current.globalCompositeOperation = "source-over"
            editor.ctx_current.rect(0, 0, editor.width, editor.height)
            editor.ctx_current.fill()
            editor.setBrush()
        },
        trackHistory: true,
    },
    {
        id: "remove_background",
        name: "Remove background",
        icon: "fa-solid fa-person-circle-minus",
        handler: removeImageBackground,
        trackHistory: true,
    },
    {
        id: "clear",
        name: "Clear",
        icon: "fa-solid fa-xmark",
        handler: (editor) => {
            editor.ctx_current.clearRect(0, 0, editor.width, editor.height)
            imageEditor.setImage(null, editor.width, editor.height) // properly reset the drawing canvas
        },
        trackHistory: true,
    },
    {
        id: "undo",
        name: "Undo",
        icon: "fa-solid fa-rotate-left",
        handler: (editor) => {
            editor.history.undo()
        },
        trackHistory: false,
    },
    {
        id: "redo",
        name: "Redo",
        icon: "fa-solid fa-rotate-right",
        handler: (editor) => {
            editor.history.redo()
        },
        trackHistory: false,
    },
    {
        id: "reset_view",
        name: "Reset zoom / pan",
        icon: "fa-solid fa-magnifying-glass",
        handler: (editor) => editor.resetView(),
        trackHistory: false,
    },
]

var IMAGE_EDITOR_SECTIONS = [
    {
        name: "tool",
        title: "Tool",
        default: "draw",
        options: Array.from(IMAGE_EDITOR_TOOLS.map((t) => t.id)),
        initElement: (element, option) => {
            var tool_info = IMAGE_EDITOR_TOOLS.find((t) => t.id == option)
            element.className = "image-editor-button button"
            var sub_element = document.createElement("div")
            var icon = document.createElement("i")
            tool_info.icon.split(" ").forEach((c) => icon.classList.add(c))
            sub_element.appendChild(icon)
            var label_element = document.createElement("div")
            label_element.classList.add("image-editor-button-label")
            label_element.textContent = tool_info.name
            sub_element.appendChild(label_element)
            element.appendChild(sub_element)
        },
    },
    {
        name: "color",
        title: "Color",
        default: "#f1c232",
        options: [
            "custom",
            "#ea9999",
            "#e06666",
            "#cc0000",
            "#990000",
            "#660000",
            "#f9cb9c",
            "#f6b26b",
            "#e69138",
            "#b45f06",
            "#783f04",
            "#ffe599",
            "#ffd966",
            "#f1c232",
            "#bf9000",
            "#7f6000",
            "#b6d7a8",
            "#93c47d",
            "#6aa84f",
            "#38761d",
            "#274e13",
            "#a4c2f4",
            "#6d9eeb",
            "#3c78d8",
            "#1155cc",
            "#1c4587",
            "#b4a7d6",
            "#8e7cc3",
            "#674ea7",
            "#351c75",
            "#20124d",
            "#d5a6bd",
            "#c27ba0",
            "#a64d79",
            "#741b47",
            "#4c1130",
            "#ffffff",
            "#c0c0c0",
            "#838383",
            "#525252",
            "#000000",
        ],
        initElement: (element, option) => {
            if (option == "custom") {
                var input = document.createElement("input")
                input.type = "color"
                element.appendChild(input)
                var span = document.createElement("span")
                span.textContent = "Custom"
                span.onclick = function (e) {
                    input.click()
                }
                element.appendChild(span)
            } else {
                element.style.background = option
            }
        },
        getCustom: (editor) => {
            var input = editor.popup.querySelector(".image_editor_color input")
            return input.value
        },
    },
    {
        name: "brush_size",
        title: "Brush Size",
        default: 48,
        options: [6, 12, 16, 24, 30, 40, 48, 64],
        initElement: (element, option) => {
            element.parentElement.style.flex = option
            element.style.width = option + "px"
            element.style.height = option + "px"
            element.style["margin-right"] = "2px"
            element.style["border-radius"] = (option / 2).toFixed() + "px"
        },
    },
    {
        name: "wand_threshold",
        title: "Wand Threshold",
        default: 30,
        options: [10, 20, 30, 45, 65, 90],
        initElement: (element, option) => {
            element.textContent = option
            element.title = `Select colors within ${option} RGB distance`
        },
    },
    {
        name: "selection_buffer",
        title: "Selection Buffer",
        default: 0,
        options: [0, 2, 4, 8, 16, 32],
        initElement: (element, option) => {
            element.textContent = option
            element.title = `Expand Magic Wand selections by ${option} pixels`
        },
    },
    {
        name: "opacity",
        title: "Opacity",
        default: 0,
        options: [0, 0.2, 0.4, 0.6, 0.8],
        initElement: (element, option) => {
            element.style.background = `repeating-conic-gradient(rgba(0, 0, 0, ${option}) 0% 25%, rgba(255, 255, 255, ${option}) 0% 50%) 50% / 10px 10px`
        },
    },
    {
        name: "sharpness",
        title: "Sharpness",
        default: 0,
        options: [0, 0.05, 0.1, 0.2, 0.3],
        initElement: (element, option) => {
            var size = 32
            var blur_amount = parseInt(option * size)
            var sub_element = document.createElement("div")
            sub_element.style.background = `var(--background-color3)`
            sub_element.style.filter = `blur(${blur_amount}px)`
            sub_element.style.width = `${size - 2}px`
            sub_element.style.height = `${size - 2}px`
            sub_element.style["border-radius"] = `${size}px`
            element.style.background = "none"
            element.appendChild(sub_element)
        },
    },
]

class EditorHistory {
    constructor(editor) {
        this.editor = editor
        this.events = [] // stack of all events (actions/edits)
        this.current_edit = null
        this.rewind_index = 0 // how many events back into the history we've rewound to. (current state is just after event at index 'length - this.rewind_index - 1')
    }
    push(event) {
        // probably add something here eventually to save state every x events
        if (this.rewind_index != 0) {
            this.events = this.events.slice(0, 0 - this.rewind_index)
            this.rewind_index = 0
        }
        var snapshot_frequency = 20 // (every x edits, take a snapshot of the current drawing state, for faster rewinding)
        if (this.events.length > 0 && this.events.length % snapshot_frequency == 0) {
            event.snapshot = this.editor.layers.drawing.ctx.getImageData(0, 0, this.editor.width, this.editor.height)
        }
        this.events.push(event)
    }
    pushAction(action) {
        this.push({
            type: "action",
            id: action,
        })
    }
    editBegin(x, y) {
        this.current_edit = {
            type: "edit",
            id: this.editor.getOptionValue("tool"),
            options: Object.assign({}, this.editor.options),
            points: [{ x: x, y: y }],
        }
    }
    editMove(x, y) {
        if (this.current_edit) {
            this.current_edit.points.push({ x: x, y: y })
        }
    }
    editEnd(x, y) {
        if (this.current_edit) {
            this.push(this.current_edit)
            this.current_edit = null
        }
    }
    clear() {
        this.events = []
    }
    undo() {
        this.rewindTo(this.rewind_index + 1)
    }
    redo() {
        this.rewindTo(this.rewind_index - 1)
    }
    rewindTo(new_rewind_index) {
        if (new_rewind_index < 0 || new_rewind_index > this.events.length) {
            return // do nothing if target index is out of bounds
        }

        var ctx = this.editor.layers.drawing.ctx
        ctx.clearRect(0, 0, this.editor.width, this.editor.height)
        if (this.editor.baseBackgroundImageData) {
            this.editor.layers.background.ctx.putImageData(this.editor.baseBackgroundImageData, 0, 0)
        }

        var target_index = this.events.length - 1 - new_rewind_index
        var snapshot_index = target_index
        while (snapshot_index > -1) {
            if (this.events[snapshot_index].snapshot) {
                break
            }
            snapshot_index--
        }

        if (snapshot_index != -1) {
            ctx.putImageData(this.events[snapshot_index].snapshot, 0, 0)
        }

        for (var i = snapshot_index + 1; i <= target_index; i++) {
            var event = this.events[i]
            if (event.type == "action") {
                var action = IMAGE_EDITOR_ACTIONS.find((a) => a.id == event.id)
                action.handler(this.editor)
            } else if (event.type == "edit") {
                var tool = IMAGE_EDITOR_TOOLS.find((t) => t.id == event.id)
                this.editor.setBrush(this.editor.layers.drawing, event.options)

                var first_point = event.points[0]
                tool.begin(this.editor, ctx, first_point.x, first_point.y)
                for (var point_i = 1; point_i < event.points.length; point_i++) {
                    tool.move(this.editor, ctx, event.points[point_i].x, event.points[point_i].y)
                }
                var last_point = event.points[event.points.length - 1]
                tool.end(this.editor, ctx, last_point.x, last_point.y)
            }
        }

        // re-set brush to current settings
        this.editor.setBrush(this.editor.layers.drawing)

        this.rewind_index = new_rewind_index
    }
}

class ImageEditor {
    constructor(popup, inpainter = false) {
        this.inpainter = inpainter
        this.popup = popup
        this.history = new EditorHistory(this)
        if (inpainter) {
            this.popup.classList.add("inpainter")
        }
        this.drawing = false
        this.temp_previous_tool = null // used for the ctrl-colorpicker functionality
        this.container = popup.querySelector(".editor-controls-center > div")
        this.layers = {}
        var layer_names = ["background", "drawing", "overlay"]
        layer_names.forEach((name) => {
            let canvas = document.createElement("canvas")
            canvas.className = `editor-canvas-${name}`
            this.container.appendChild(canvas)
            this.layers[name] = {
                name: name,
                canvas: canvas,
                ctx: canvas.getContext("2d"),
            }
        })
        this.containerScale = 1.0
        this.viewZoom = 1.0
        this.viewPanX = 0
        this.viewPanY = 0
        this.gesture = null
        this.panning = null
        this.activePointers = new Map()
        this.activePointerId = null
        this.touchGestureActive = false
        this.touchStrokeSnapshot = null
        this.spacePressed = false
        this.fitFrame = null
        this.fitShouldReset = false
        this.container.style.touchAction = "none"

        // Pointer capture keeps drawing and panning stable when a transformed
        // canvas moves out from underneath a desktop pointer.
        this.pointerHandlerBound = this.pointerHandler.bind(this)
        ;["pointerdown", "pointermove", "pointerup", "pointercancel", "lostpointercapture"].forEach((type) => {
            this.container.addEventListener(type, this.pointerHandlerBound)
        })
        this.wheelHandlerBound = this.wheelHandler.bind(this)
        this.container.addEventListener("wheel", this.wheelHandlerBound, { passive: false })
        this.container.addEventListener("contextmenu", (event) => event.preventDefault())

        if (typeof ResizeObserver === "function") {
            this.resizeObserver = new ResizeObserver(() => this.scheduleFitToViewport())
            this.resizeObserver.observe(this.popup.querySelector(".editor-controls-center"))
        } else {
            this.windowResizeHandlerBound = () => this.scheduleFitToViewport()
            window.addEventListener("resize", this.windowResizeHandlerBound)
        }

        // initialize editor controls
        this.options = {}
        this.optionElements = {}
        IMAGE_EDITOR_SECTIONS.forEach((section) => {
            section.id = `image_editor_${section.name}`
            var sectionElement = document.createElement("div")
            sectionElement.className = section.id

            var title = document.createElement("h4")
            title.innerText = section.title
            sectionElement.appendChild(title)

            var optionsContainer = document.createElement("div")
            optionsContainer.classList.add("editor-options-container")

            this.optionElements[section.name] = []
            section.options.forEach((option, index) => {
                var optionHolder = document.createElement("div")
                var optionElement = document.createElement("div")
                optionHolder.appendChild(optionElement)
                section.initElement(optionElement, option)
                optionHolder.addEventListener("click", () => this.selectOption(section.name, index))
                optionsContainer.appendChild(optionHolder)
                this.optionElements[section.name].push(optionElement)
            })
            this.selectOption(section.name, section.options.indexOf(section.default))

            sectionElement.appendChild(optionsContainer)

            this.popup.querySelector(".editor-controls-left").appendChild(sectionElement)
        })

        this.custom_color_input = this.popup.querySelector(`input[type="color"]`)
        this.custom_color_input.addEventListener("change", () => {
            this.custom_color_input.parentElement.style.background = this.custom_color_input.value
            this.selectOption("color", 0)
        })

        if (this.inpainter) {
            this.selectOption("color", IMAGE_EDITOR_SECTIONS.find((s) => s.name == "color").options.indexOf("#ffffff"))
            this.selectOption("opacity", IMAGE_EDITOR_SECTIONS.find((s) => s.name == "opacity").options.indexOf(0.4))
        }

        // initialize the right-side controls
        var buttonContainer = document.createElement("div")
        IMAGE_EDITOR_BUTTONS.forEach((button) => {
            var element = document.createElement("div")
            var icon = document.createElement("i")
            element.className = "image-editor-button button"
            icon.className = button.icon
            element.appendChild(icon)
            element.append(button.name)
            buttonContainer.appendChild(element)
            element.addEventListener("click", (event) => button.handler(this))
        })
        var actionsContainer = document.createElement("div")
        var actionsTitle = document.createElement("h4")
        actionsTitle.textContent = "Actions"
        actionsContainer.appendChild(actionsTitle)
        IMAGE_EDITOR_ACTIONS.forEach((action) => {
            var element = document.createElement("div")
            var icon = document.createElement("i")
            element.className = "image-editor-button button"
            if (action.className) {
                element.className += " " + action.className
            }
            icon.className = action.icon
            element.appendChild(icon)
            element.append(action.name)
            actionsContainer.appendChild(element)
            element.addEventListener("click", (event) => this.runAction(action.id))
        })
        this.popup.querySelector(".editor-controls-right").appendChild(actionsContainer)
        this.popup.querySelector(".editor-controls-right").appendChild(buttonContainer)

        this.keyHandlerBound = this.keyHandler.bind(this)
        this.windowBlurHandlerBound = () => this.cancelInteractions()

        this.setSize(512, 512)
    }
    activateInput() {
        if (this.inputActive) return
        this.inputActive = true
        document.addEventListener("keydown", this.keyHandlerBound, true)
        document.addEventListener("keyup", this.keyHandlerBound, true)
        window.addEventListener("blur", this.windowBlurHandlerBound)
    }
    deactivateInput() {
        if (!this.inputActive) return
        this.inputActive = false
        document.removeEventListener("keydown", this.keyHandlerBound, true)
        document.removeEventListener("keyup", this.keyHandlerBound, true)
        window.removeEventListener("blur", this.windowBlurHandlerBound)
        this.spacePressed = false
        this.cancelInteractions()
    }
    show() {
        if (this.popup.classList.contains("editor-page-pane") && typeof window.openImageEditorPage === "function") {
            window.openImageEditorPage(this.inpainter ? "inpaint" : "draw")
        }
        this.popup.classList.add("active")
        this.activateInput()
        this.scheduleFitToViewport()
    }
    hide() {
        this.popup.classList.remove("active")
        this.deactivateInput()
        if (this.popup.classList.contains("editor-page-pane") && typeof window.closeImageEditorPage === "function") {
            window.closeImageEditorPage()
        }
    }
    setSize(width, height) {
        width = parseInt(width)
        height = parseInt(height)

        if (width == this.width && height == this.height) {
            return
        }

        this.width = parseInt(width)
        this.height = parseInt(height)

        Object.values(this.layers).forEach((layer) => {
            layer.canvas.width = width
            layer.canvas.height = height
        })

        this.scheduleFitToViewport(true)

        if (this.inpainter) {
            this.saveImage() // We've reset the size of the image so inpainting is different
        }
        this.setBrush()
        this.history.clear()
    }
    get tool() {
        var tool_id = this.getOptionValue("tool")
        return IMAGE_EDITOR_TOOLS.find((t) => t.id == tool_id)
    }
    loadTool() {
        this.drawing = false
        this.updateEditorCursor()
    }
    updateEditorCursor() {
        if (this.panning) this.container.style.cursor = "grabbing"
        else if (this.spacePressed || this.tool.id === "pan") this.container.style.cursor = "grab"
        else this.container.style.cursor = this.tool.cursor
    }
    scheduleFitToViewport(resetView = false) {
        this.fitShouldReset = this.fitShouldReset || resetView
        if (this.fitFrame !== null) cancelAnimationFrame(this.fitFrame)
        this.fitFrame = requestAnimationFrame(() => {
            const shouldReset = this.fitShouldReset
            this.fitFrame = null
            this.fitShouldReset = false
            this.fitToViewport(shouldReset)
        })
    }
    fitToViewport(resetView = false) {
        if (!this.width || !this.height) return
        const viewport = this.popup.querySelector(".editor-controls-center")
        const style = getComputedStyle(viewport)
        const horizontalPadding = parseFloat(style.paddingLeft || 0) + parseFloat(style.paddingRight || 0)
        const verticalPadding = parseFloat(style.paddingTop || 0) + parseFloat(style.paddingBottom || 0)
        let availableWidth = viewport.clientWidth - horizontalPadding
        let availableHeight = viewport.clientHeight - verticalPadding

        // The legacy popup is measured before it is first shown. Use a sane
        // viewport fallback until the editor page has real layout dimensions.
        if (availableWidth < 64) availableWidth = Math.max(240, window.innerWidth * 0.88)
        if (availableHeight < 64) availableHeight = Math.max(240, window.innerHeight * 0.7)

        const nextScale = Math.max(
            0.01,
            Math.min(availableWidth / this.width, availableHeight / this.height, IMAGE_EDITOR_MAX_UPSCALE)
        )
        const scaleChanged = Math.abs(nextScale - this.containerScale) > 0.001
        this.containerScale = nextScale
        // Preserve the exact source ratio. Rounding each axis separately can
        // make narrow or very wide images visibly drift by a pixel and puts
        // the overlay out of alignment with the image at mobile sizes.
        this.container.style.aspectRatio = `${this.width} / ${this.height}`
        this.container.style.width = `${this.width * nextScale}px`
        this.container.style.height = `${this.height * nextScale}px`
        if (scaleChanged && this.options) this.setBrush()

        if (resetView) this.resetView()
        else {
            this.clampView()
            this.applyViewTransform()
        }
    }
    applyViewTransform() {
        this.container.style.transformOrigin = "0 0"
        this.container.style.transform = `translate(${this.viewPanX}px, ${this.viewPanY}px) scale(${this.viewZoom})`
    }
    clampView() {
        const viewport = this.popup.querySelector(".editor-controls-center")
        const viewportWidth = viewport.clientWidth
        const viewportHeight = viewport.clientHeight
        const baseWidth = this.container.offsetWidth
        const baseHeight = this.container.offsetHeight
        if (!viewportWidth || !viewportHeight || !baseWidth || !baseHeight) return

        const renderedWidth = baseWidth * this.viewZoom
        const renderedHeight = baseHeight * this.viewZoom
        const baseLeft = (viewportWidth - baseWidth) / 2
        const baseTop = (viewportHeight - baseHeight) / 2
        const visibleX = Math.min(IMAGE_EDITOR_MIN_VISIBLE, renderedWidth / 2)
        const visibleY = Math.min(IMAGE_EDITOR_MIN_VISIBLE, renderedHeight / 2)

        if (renderedWidth <= viewportWidth - 16) this.viewPanX = (baseWidth - renderedWidth) / 2
        else {
            const minX = visibleX - baseLeft - renderedWidth
            const maxX = viewportWidth - visibleX - baseLeft
            this.viewPanX = Math.max(minX, Math.min(maxX, this.viewPanX))
        }
        if (renderedHeight <= viewportHeight - 16) this.viewPanY = (baseHeight - renderedHeight) / 2
        else {
            const minY = visibleY - baseTop - renderedHeight
            const maxY = viewportHeight - visibleY - baseTop
            this.viewPanY = Math.max(minY, Math.min(maxY, this.viewPanY))
        }
    }
    resetView() {
        this.viewZoom = 1
        this.viewPanX = 0
        this.viewPanY = 0
        this.gesture = null
        this.panning = null
        this.applyViewTransform()
    }
    setViewZoom(nextZoom, clientX, clientY) {
        nextZoom = Math.max(0.5, Math.min(8, nextZoom))
        const bbox = this.container.getBoundingClientRect()
        const relativeX = clientX - bbox.left
        const relativeY = clientY - bbox.top
        const ratio = nextZoom / this.viewZoom
        this.viewPanX += relativeX - relativeX * ratio
        this.viewPanY += relativeY - relativeY * ratio
        this.viewZoom = nextZoom
        this.clampView()
        this.applyViewTransform()
    }
    wheelHandler(event) {
        event.preventDefault()
        const delta = Math.max(-120, Math.min(120, event.deltaY))
        const multiplier = Math.exp(-delta * 0.0012)
        this.setViewZoom(this.viewZoom * multiplier, event.clientX, event.clientY)
    }
    setImage(url, width, height) {
        this.setSize(width, height)
        this.layers.background.ctx.clearRect(0, 0, this.width, this.height)
        if (!(url && this.inpainter)) {
            this.layers.drawing.ctx.clearRect(0, 0, this.width, this.height)
        }
        if (url) {
            var image = new Image()
            image.onload = () => {
                this.layers.background.ctx.drawImage(image, 0, 0, this.width, this.height)
                this.captureBaseBackground()
            }
            image.src = url
        } else {
            this.layers.background.ctx.fillStyle = "#ffffff"
            this.layers.background.ctx.beginPath()
            this.layers.background.ctx.rect(0, 0, this.width, this.height)
            this.layers.background.ctx.fill()
            this.captureBaseBackground()
        }
        this.history.clear()
    }
    captureBaseBackground() {
        this.baseBackgroundImageData = this.layers.background.ctx.getImageData(0, 0, this.width, this.height)
    }
    saveImage() {
        if (!this.inpainter) {
            // This is not an inpainter, so save the image as the new img2img input
            this.layers.background.ctx.drawImage(this.layers.drawing.canvas, 0, 0, this.width, this.height)
            var base64 = this.layers.background.canvas.toDataURL()
            initImagePreview.src = base64 // this will trigger the rest of the app to use it
        } else {
            // This is an inpainter, so make sure the toggle is set accordingly
            var is_blank = !this.layers.drawing.ctx
                .getImageData(0, 0, this.width, this.height)
                .data.some((channel) => channel !== 0)
            maskSetting.checked = !is_blank
            maskSetting.dispatchEvent(new Event("change"))
        }
        this.hide()
    }
    getImg() {
        // a drop-in replacement of the drawingboard version
        return this.layers.drawing.canvas.toDataURL()
    }
    setImg(dataUrl) {
        // a drop-in replacement of the drawingboard version
        var image = new Image()
        image.onload = () => {
            var ctx = this.layers.drawing.ctx
            ctx.clearRect(0, 0, this.width, this.height)
            ctx.globalCompositeOperation = "source-over"
            ctx.globalAlpha = 1
            ctx.filter = "none"
            ctx.drawImage(image, 0, 0, this.width, this.height)
            this.setBrush(this.layers.drawing)
        }
        image.src = dataUrl
    }
    runAction(action_id) {
        var action = IMAGE_EDITOR_ACTIONS.find((a) => a.id == action_id)
        if (action.trackHistory) {
            this.history.pushAction(action_id)
        }
        action.handler(this)
    }
    setBrush(layer = null, options = null) {
        if (options == null) {
            options = this.options
        }
        if (layer) {
            layer.ctx.lineCap = "round"
            layer.ctx.lineJoin = "round"
            layer.ctx.lineWidth = options.brush_size / this.containerScale
            layer.ctx.fillStyle = options.color
            layer.ctx.strokeStyle = options.color
            var sharpness = parseInt(options.sharpness * options.brush_size)
            layer.ctx.filter = sharpness == 0 ? `none` : `blur(${sharpness}px)`
            layer.ctx.globalAlpha = 1 - options.opacity
            layer.ctx.globalCompositeOperation = "source-over"
            var tool = IMAGE_EDITOR_TOOLS.find((t) => t.id == options.tool)
            if (tool && tool.setBrush) {
                tool.setBrush(this, layer)
            }
        } else {
            Object.values(["drawing", "overlay"])
                .map((name) => this.layers[name])
                .forEach((l) => {
                    this.setBrush(l)
                })
        }
    }
    get ctx_overlay() {
        return this.layers.overlay.ctx
    }
    get ctx_current() {
        // the idea is this will help support having custom layers and editing each one
        return this.layers.drawing.ctx
    }
    get canvas_current() {
        return this.layers.drawing.canvas
    }
    keyHandler(event) {
        // handles keybinds like ctrl+z, ctrl+y
        if (!this.popup.classList.contains("active")) {
            this.deactivateInput()
            return // this catches if something else closes the window but doesnt properly unbind the key handler
        }

        const target = event.target
        const isTyping = target?.matches?.("input, textarea, select, [contenteditable='true']")
        if (event.code === "Space" && (!isTyping || this.spacePressed)) {
            this.spacePressed = event.type === "keydown"
            this.updateEditorCursor()
            event.stopPropagation()
            event.preventDefault()
            return
        }

        // keybindings
        if (event.type == "keydown" && !isTyping) {
            if ((event.key == "z" || event.key == "Z") && event.ctrlKey) {
                if (!event.shiftKey) {
                    this.history.undo()
                } else {
                    this.history.redo()
                }
                event.stopPropagation()
                event.preventDefault()
            }
            else if (event.key == "y" && event.ctrlKey) {
                this.history.redo()
                event.stopPropagation()
                event.preventDefault()
            }
            else if (event.key === "Escape") {
                this.hide()
                event.stopPropagation()
                event.preventDefault()
            } else {
                let toolIndex = IMAGE_EDITOR_TOOLS.findIndex(t => t.hotkey == event.key)
                if (toolIndex != -1) {
                    this.selectOption("tool", toolIndex)
                    event.stopPropagation()
                    event.preventDefault()
                }
            }
        }

        // dropper ctrl holding handler stuff
        var dropper_active = this.temp_previous_tool != null
        if (dropper_active && !event.ctrlKey) {
            this.selectOption(
                "tool",
                IMAGE_EDITOR_TOOLS.findIndex((t) => t.id == this.temp_previous_tool)
            )
            this.temp_previous_tool = null
        } else if (!dropper_active && event.ctrlKey) {
            this.temp_previous_tool = this.getOptionValue("tool")
            this.selectOption(
                "tool",
                IMAGE_EDITOR_TOOLS.findIndex((t) => t.id == "colorpicker")
            )
        }
    }
    eventToCanvasPoint(event) {
        const bbox = this.layers.overlay.canvas.getBoundingClientRect()
        const rawX = bbox.width > 0 ? (event.clientX - bbox.left) * this.width / bbox.width : 0
        const rawY = bbox.height > 0 ? (event.clientY - bbox.top) * this.height / bbox.height : 0
        return {
            x: Math.max(0, Math.min(this.width, rawX)),
            y: Math.max(0, Math.min(this.height, rawY)),
        }
    }
    rememberPointer(event) {
        this.activePointers.set(event.pointerId, {
            pointerId: event.pointerId,
            pointerType: event.pointerType,
            clientX: event.clientX,
            clientY: event.clientY,
        })
    }
    get touchPointers() {
        return Array.from(this.activePointers.values()).filter((pointer) => pointer.pointerType === "touch")
    }
    shouldPan(event) {
        return event.button === 1 || event.button === 2 ||
            (event.button === 0 && (event.altKey || this.spacePressed || this.tool.id === "pan"))
    }
    beginPan(event) {
        this.panning = {
            pointerId: event.pointerId,
            x: event.clientX,
            y: event.clientY,
            panX: this.viewPanX,
            panY: this.viewPanY,
        }
        this.updateEditorCursor()
    }
    movePan(event) {
        if (!this.panning || event.pointerId !== this.panning.pointerId) return
        this.viewPanX = this.panning.panX + event.clientX - this.panning.x
        this.viewPanY = this.panning.panY + event.clientY - this.panning.y
        this.clampView()
        this.applyViewTransform()
    }
    endPan(pointerId) {
        if (!this.panning || pointerId !== this.panning.pointerId) return
        this.panning = null
        this.updateEditorCursor()
    }
    beginDrawing(event) {
        const point = this.eventToCanvasPoint(event)
        this.activePointerId = event.pointerId
        this.drawing = true
        this.tool.begin(this, this.ctx_current, point.x, point.y)
        this.tool.begin(this, this.ctx_overlay, point.x, point.y, true)
        this.history.editBegin(point.x, point.y)
    }
    moveDrawing(event) {
        if (!this.drawing || event.pointerId !== this.activePointerId) return
        const events = typeof event.getCoalescedEvents === "function" ? event.getCoalescedEvents() : [event]
        for (const sample of events.length ? events : [event]) {
            const point = this.eventToCanvasPoint(sample)
            this.tool.move(this, this.ctx_current, point.x, point.y)
            this.tool.move(this, this.ctx_overlay, point.x, point.y, true)
            this.history.editMove(point.x, point.y)
        }
    }
    finishDrawing(event) {
        if (!this.drawing || event.pointerId !== this.activePointerId) return
        const point = this.eventToCanvasPoint(event)
        this.tool.move(this, this.ctx_current, point.x, point.y)
        this.tool.move(this, this.ctx_overlay, point.x, point.y, true)
        this.history.editMove(point.x, point.y)
        this.tool.end(this, this.ctx_current, point.x, point.y)
        this.tool.end(this, this.ctx_overlay, point.x, point.y, true)
        this.history.editEnd(point.x, point.y)
        this.drawing = false
        this.activePointerId = null
        this.touchStrokeSnapshot = null
    }
    cancelDrawing(restoreSnapshot = false) {
        if (restoreSnapshot && this.touchStrokeSnapshot) {
            this.layers.drawing.ctx.putImageData(this.touchStrokeSnapshot, 0, 0)
        }
        this.ctx_overlay.clearRect(0, 0, this.width, this.height)
        this.canvas_current.style.opacity = ""
        this.history.current_edit = null
        this.drawing = false
        this.activePointerId = null
        this.touchStrokeSnapshot = null
        this.setBrush()
    }
    startTouchGesture() {
        const touches = this.touchPointers
        if (touches.length < 2) return
        const centerX = (touches[0].clientX + touches[1].clientX) / 2
        const centerY = (touches[0].clientY + touches[1].clientY) / 2
        const distance = Math.hypot(
            touches[1].clientX - touches[0].clientX,
            touches[1].clientY - touches[0].clientY
        )
        const bbox = this.container.getBoundingClientRect()
        this.gesture = {
            centerX,
            centerY,
            distance: Math.max(1, distance),
            zoom: this.viewZoom,
            panX: this.viewPanX,
            panY: this.viewPanY,
            relativeX: centerX - bbox.left,
            relativeY: centerY - bbox.top,
        }
    }
    moveTouchGesture() {
        const touches = this.touchPointers
        if (touches.length < 2) return
        if (!this.gesture) this.startTouchGesture()
        if (!this.gesture) return
        const centerX = (touches[0].clientX + touches[1].clientX) / 2
        const centerY = (touches[0].clientY + touches[1].clientY) / 2
        const distance = Math.hypot(
            touches[1].clientX - touches[0].clientX,
            touches[1].clientY - touches[0].clientY
        )
        const zoom = Math.max(0.5, Math.min(8, this.gesture.zoom * distance / this.gesture.distance))
        const ratio = zoom / this.gesture.zoom
        this.viewZoom = zoom
        this.viewPanX = this.gesture.panX + (centerX - this.gesture.centerX) + this.gesture.relativeX * (1 - ratio)
        this.viewPanY = this.gesture.panY + (centerY - this.gesture.centerY) + this.gesture.relativeY * (1 - ratio)
        this.clampView()
        this.applyViewTransform()
    }
    releasePointer(pointerId) {
        if (this.container.hasPointerCapture?.(pointerId)) this.container.releasePointerCapture(pointerId)
    }
    cancelInteractions() {
        if (this.drawing) {
            const pointer = this.activePointers.get(this.activePointerId)
            if (pointer) this.finishDrawing(pointer)
            else this.cancelDrawing(false)
        }
        this.panning = null
        this.gesture = null
        this.touchGestureActive = false
        this.activePointers.clear()
        this.updateEditorCursor()
    }
    pointerHandler(event) {
        if (event.type === "pointerdown") {
            if (event.pointerType === "mouse" && ![0, 1, 2].includes(event.button)) return
            event.preventDefault()
            this.rememberPointer(event)
            this.container.setPointerCapture?.(event.pointerId)

            if (event.pointerType === "touch") {
                if (this.touchPointers.length >= 2) {
                    this.cancelDrawing(true)
                    this.touchGestureActive = true
                    this.startTouchGesture()
                } else if (!this.touchGestureActive) {
                    this.touchStrokeSnapshot = this.layers.drawing.ctx.getImageData(0, 0, this.width, this.height)
                    this.beginDrawing(event)
                }
            } else if (this.shouldPan(event)) this.beginPan(event)
            else if (event.button === 0) this.beginDrawing(event)
            return
        }

        if (event.type === "pointermove") {
            if (!this.activePointers.has(event.pointerId)) return
            event.preventDefault()
            this.rememberPointer(event)
            if (event.pointerType === "touch" && this.touchGestureActive) this.moveTouchGesture()
            else if (this.panning) this.movePan(event)
            else this.moveDrawing(event)
            return
        }

        if (event.type === "lostpointercapture") {
            if (!this.activePointers.has(event.pointerId)) return
        } else if (event.type !== "pointerup" && event.type !== "pointercancel") return

        event.preventDefault()
        if (event.pointerType === "touch" && this.touchGestureActive) {
            this.activePointers.delete(event.pointerId)
            if (this.touchPointers.length < 2) this.gesture = null
            if (this.touchPointers.length === 0) this.touchGestureActive = false
        } else if (this.panning) {
            this.endPan(event.pointerId)
            this.activePointers.delete(event.pointerId)
        } else {
            this.finishDrawing(event)
            this.activePointers.delete(event.pointerId)
        }
        this.releasePointer(event.pointerId)
    }
    getOptionValue(section_name) {
        var section = IMAGE_EDITOR_SECTIONS.find((s) => s.name == section_name)
        return this.options && section_name in this.options ? this.options[section_name] : section.default
    }
    selectOption(section_name, option_index) {
        var section = IMAGE_EDITOR_SECTIONS.find((s) => s.name == section_name)
        var value = section.options[option_index]
        this.options[section_name] = value == "custom" ? section.getCustom(this) : value

        this.optionElements[section_name].forEach((element) => element.classList.remove("active"))
        this.optionElements[section_name][option_index].classList.add("active")

        // change the editor
        this.setBrush()
        if (section.name == "tool") {
            this.loadTool()
        }
    }
}

const imageEditor = new ImageEditor(document.getElementById("image-editor"))
const imageInpainter = new ImageEditor(document.getElementById("image-inpainter"), true)

imageEditor.setImage(null, 512, 512)
imageInpainter.setImage(null, 512, 512)

document.getElementById("init_image_button_draw").addEventListener("click", () => {
    imageEditor.show()
})
document.getElementById("init_image_button_inpaint").addEventListener("click", () => {
    imageInpainter.show()
})

img2imgUnload() // no init image when the app starts

function rgbToHex(rgb) {
    function componentToHex(c) {
        var hex = parseInt(c).toString(16)
        return hex.length == 1 ? "0" + hex : hex
    }
    return "#" + componentToHex(rgb.r) + componentToHex(rgb.g) + componentToHex(rgb.b)
}

function hexToRgb(hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
    return result
        ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16),
        }
        : null
}

function pixelCompare(int1, int2) {
    return Math.abs(int1 - int2) < 4
}

// adapted from https://ben.akrin.com/canvas_fill/fill_04.html
// May 2023 - look at using a library instead of custom code: https://github.com/shaneosullivan/example-canvas-fill
function flood_fill(editor, the_canvas_context, x, y, color) {
    pixel_stack = [{ x: x, y: y }]
    pixels = the_canvas_context.getImageData(0, 0, editor.width, editor.height)
    var linear_cords = (y * editor.width + x) * 4
    var original_color = {
        r: pixels.data[linear_cords],
        g: pixels.data[linear_cords + 1],
        b: pixels.data[linear_cords + 2],
        a: pixels.data[linear_cords + 3],
    }

    var opacity = color.a / 255
    var new_color = {
        r: parseInt(color.r * opacity + original_color.r * (1 - opacity)),
        g: parseInt(color.g * opacity + original_color.g * (1 - opacity)),
        b: parseInt(color.b * opacity + original_color.b * (1 - opacity)),
    }

    if (
        pixelCompare(new_color.r, original_color.r) &&
        pixelCompare(new_color.g, original_color.g) &&
        pixelCompare(new_color.b, original_color.b)
    ) {
        return // This color is already the color we want, so do nothing
    }
    var max_stack_size = editor.width * editor.height
    while (pixel_stack.length > 0 && pixel_stack.length < max_stack_size) {
        new_pixel = pixel_stack.shift()
        x = new_pixel.x
        y = new_pixel.y

        linear_cords = (y * editor.width + x) * 4
        while (
            y-- >= 0 &&
            pixelCompare(pixels.data[linear_cords], original_color.r) &&
            pixelCompare(pixels.data[linear_cords + 1], original_color.g) &&
            pixelCompare(pixels.data[linear_cords + 2], original_color.b)
        ) {
            linear_cords -= editor.width * 4
        }
        linear_cords += editor.width * 4
        y++

        var reached_left = false
        var reached_right = false
        while (
            y++ < editor.height &&
            pixelCompare(pixels.data[linear_cords], original_color.r) &&
            pixelCompare(pixels.data[linear_cords + 1], original_color.g) &&
            pixelCompare(pixels.data[linear_cords + 2], original_color.b)
        ) {
            pixels.data[linear_cords] = new_color.r
            pixels.data[linear_cords + 1] = new_color.g
            pixels.data[linear_cords + 2] = new_color.b
            pixels.data[linear_cords + 3] = 255

            if (x > 0) {
                if (
                    pixelCompare(pixels.data[linear_cords - 4], original_color.r) &&
                    pixelCompare(pixels.data[linear_cords - 4 + 1], original_color.g) &&
                    pixelCompare(pixels.data[linear_cords - 4 + 2], original_color.b)
                ) {
                    if (!reached_left) {
                        pixel_stack.push({ x: x - 1, y: y })
                        reached_left = true
                    }
                } else if (reached_left) {
                    reached_left = false
                }
            }

            if (x < editor.width - 1) {
                if (
                    pixelCompare(pixels.data[linear_cords + 4], original_color.r) &&
                    pixelCompare(pixels.data[linear_cords + 4 + 1], original_color.g) &&
                    pixelCompare(pixels.data[linear_cords + 4 + 2], original_color.b)
                ) {
                    if (!reached_right) {
                        pixel_stack.push({ x: x + 1, y: y })
                        reached_right = true
                    }
                } else if (reached_right) {
                    reached_right = false
                }
            }

            linear_cords += editor.width * 4
        }
    }
    the_canvas_context.putImageData(pixels, 0, 0)
}
