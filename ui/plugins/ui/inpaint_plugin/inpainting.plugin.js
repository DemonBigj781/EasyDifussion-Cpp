/*
    Inpainting and Image Editor
    by Patrice

    Image editor improvements:
    - Shows the actual brush in the image editor for increased precision.
    - Add img2img source image via drag & drop from external file or browser image (incl. rendered image). Just drop the image in the editor pane.
    - Add img2img source image by pasting an image from the clipboard
    - Integrates seamlessly with Scrolling Panes 1.8+
    - Adds support for reloading task from metadata embedded in PNG and JPEG images (use Ctrl+Drop image in the editor pane)
    - Automatically sets the size of the output image to the size of the image used for img2img if its dimensions are both valid options (works with both copy/paste and drag & drop).
    - makes the brushes more visible in the image/inpainting editor.
*/
(function () {
    "use strict"

    if (!document.getElementById("inpainting-magic-wand-controls")) {
        document.body.insertAdjacentHTML(
            "beforeend",
            loadRequiredPluginHTML("/plugins/core/inpaint_plugin/inpainting.plugin.html")
        )
    }

    let imageBrushPreview
    let imageCanvas
    const configuredCanvases = new WeakSet()

    function getEditorForCanvas(canvas) {
        return canvas.closest("#image-inpainter") ? imageInpainter : imageEditor
    }

    function isBrushTool(editor) {
        return editor?.tool?.id === "draw" || editor?.tool?.id === "erase"
    }

    function updateBrushPreview(editor, event) {
        if (!imageBrushPreview || !imageCanvas || !isBrushTool(editor)) return
        const canvasBounds = imageCanvas.getBoundingClientRect()
        const renderedCanvasScale = canvasBounds.width > 0 ? canvasBounds.width / editor.width : 1
        const brushSizeInImagePixels = Number(editor.options.brush_size || 1) / editor.containerScale
        const diameter = Math.max(2, brushSizeInImagePixels * renderedCanvasScale)
        imageBrushPreview.style.width = diameter + "px"
        imageBrushPreview.style.height = diameter + "px"
        imageBrushPreview.style.left = event.clientX + "px"
        imageBrushPreview.style.top = event.clientY + "px"
        imageBrushPreview.style.background = editor.inpainter ? "#ffffff" : editor.options.color
    }

    function setupBrush(event) {
        if (event.pointerType === "touch") return
        imageCanvas = event.currentTarget
        const editor = getEditorForCanvas(imageCanvas)
        if (!isBrushTool(editor)) return
        if (imageBrushPreview === undefined) {
            imageBrushPreview = document.createElement("div")
            imageBrushPreview.className = 'image-brush-preview'
            document.body.appendChild(imageBrushPreview)
        }
        imageBrushPreview.style.display = 'block'
        updateBrushPreview(editor, event)
    }

    function cleanupBrush() {
        if (imageBrushPreview !== undefined) {
            imageBrushPreview.remove()
            imageBrushPreview = undefined
        }
    }

    function disableRightClick(e) {
        e.preventDefault()
    }

    function setupCanvas(selector) {
        const canvas = document.querySelector(selector + ' .editor-canvas-overlay')
        if (!canvas || configuredCanvases.has(canvas)) return
        configuredCanvases.add(canvas)
        canvas.addEventListener("contextmenu", disableRightClick)
        canvas.addEventListener("pointermove", updateMouseCursor)
        canvas.addEventListener("pointerenter", setupBrush)
        canvas.addEventListener("pointerleave", cleanupBrush)
    }

    // The stock editor opens immediately while its background image is still
    // being copied by an asynchronous Image.onload callback.  That long-lived
    // race is what occasionally presents a completely white Draw canvas.  Use
    // the already-decoded preview as the source and paint it synchronously
    // before showing the editor.
    document.getElementById("init_image_button_draw").addEventListener("click", (event) => {
        if (!initImagePreviewContainer.classList.contains("has-image") || !initImagePreview.naturalWidth) {
            setupCanvas('#image-editor')
            return
        }
        event.preventDefault()
        event.stopImmediatePropagation()
        const width = initImagePreview.naturalWidth
        const height = initImagePreview.naturalHeight
        imageEditor.setSize(width, height)
        imageEditor.layers.background.ctx.clearRect(0, 0, width, height)
        imageEditor.layers.background.ctx.drawImage(initImagePreview, 0, 0, width, height)
        setupCanvas('#image-editor')
        imageEditor.show()
    }, true)

    document.getElementById("init_image_button_inpaint").addEventListener("click", () => {
        setupCanvas('#image-inpainter')
    })

    // The dedicated editor tab can be opened without either legacy button.
    // Configure both canvases once so the precision preview is always present.
    setupCanvas('#image-editor')
    setupCanvas('#image-inpainter')

    function updateMouseCursor(e) {
        if (e.pointerType === "touch") return
        imageCanvas = e.currentTarget
        const editor = getEditorForCanvas(imageCanvas)
        if (!isBrushTool(editor)) {
            cleanupBrush()
            return
        }
        if (imageBrushPreview === undefined) setupBrush(e)
        else updateBrushPreview(editor, e)
    }

    /* ADD SUPPORT FOR PASTING SOURCE IMAGE FROM CLIPBOARD */
    let imageObj = new Image()

    imageObj.onload = function () {
        const bestWidth = Math.max(IMAGE_STEP_SIZE, Math.round(this.width / IMAGE_STEP_SIZE) * IMAGE_STEP_SIZE)
        const bestHeight = Math.max(IMAGE_STEP_SIZE, Math.round(this.height / IMAGE_STEP_SIZE) * IMAGE_STEP_SIZE)

        addImageSizeOption(bestWidth)
        addImageSizeOption(bestHeight)

        // Set the width and height to the closest aspect ratio and closest to original dimensions
        widthField.value = bestWidth;
        heightField.value = bestHeight;

        // Keep the original pixels. Cropping through a shared canvas here was
        // another source of blank/white images, especially for small inputs.
        initImagePreview.src = this.src
    };

    function handlePaste(e) {
        for (let i = 0; i < e.clipboardData.items.length; i++) {
            const item = e.clipboardData.items[i]
            if (item.type.indexOf("image") != -1) {
                imageObj.src = URL.createObjectURL(item.getAsFile())
            }
        }
    }
    document.addEventListener('paste', handlePaste)

    // Keep Easy Diffusion's native file listener. Paste and drop additions use
    // imageObj, but ordinary Browse/Load Image must retain the native path.


    /* ADD SUPPORT FOR DRAG-AND-DROPPING SOURCE IMAGE (from file or straight from UI) */

    /* DROP AREAS */

    function createDropAreas(container) {
        // Create two drop areas
        const dropAreaI2I = createElement("div", { id: "drop-area-I2I" }, ["drop-area"], "Use as Image2Image source")
        container.appendChild(dropAreaI2I)

        const dropAreaMD = createElement("div", { id: "drop-area-MD" }, ["drop-area"], "Extract embedded metadata")
        container.appendChild(dropAreaMD)

        const dropAreaCN = createElement("div", { id: "drop-area-CN" }, ["drop-area"], "Use as Controlnet image")
        container.appendChild(dropAreaCN)

        // Add event listeners to drop areas
        dropAreaCN.addEventListener("dragenter", function (event) {
            event.preventDefault()
            dropAreaCN.style.backgroundColor = 'darkGreen'
        })
        dropAreaCN.addEventListener("dragleave", function (event) {
            event.preventDefault()
            dropAreaCN.style.backgroundColor = ''
        })
        dropAreaCN.addEventListener("drop", function (event) {
            event.stopPropagation()
            event.preventDefault()
            hideDropAreas()

            getImageFromDropEvent(event, e => controlImagePreview.src = e)
        })

        dropAreaI2I.addEventListener("dragenter", function (event) {
            event.preventDefault()
            dropAreaI2I.style.backgroundColor = 'darkGreen'
        })
        dropAreaI2I.addEventListener("dragleave", function (event) {
            event.preventDefault()
            dropAreaI2I.style.backgroundColor = ''
        })

        function getImageFromDropEvent(event, callback) {
            // Find the first image file, uri, or moz-url in the items list
            let imageItem = null
            for (let i = 0; i < event.dataTransfer.items.length; i++) {
                let item = event.dataTransfer.items[i]
                if (item.kind === 'file' && item.type.startsWith('image/')) {
                    imageItem = item;
                    break;
                }
            }

            if (!imageItem) {
                // If no file matches, try to find a text/uri-list item
                for (let i = 0; i < event.dataTransfer.items.length; i++) {
                    let item = event.dataTransfer.items[i];
                    if (item.type === 'text/uri-list') {
                        imageItem = item;
                        break;
                    }
                }
            }

            if (!imageItem) {
                // If there are no image files or uris, fallback to moz-url
                for (let i = 0; i < event.dataTransfer.items.length; i++) {
                    let item = event.dataTransfer.items[i];
                    if (item.type === 'text/x-moz-url') {
                        imageItem = item;
                        break;
                    }
                }
            }

            if (imageItem) {
                if (imageItem.kind === 'file') {
                    // If the item is an image file, handle it as before
                    let file = imageItem.getAsFile();

                    // Create a FileReader object to read the dropped file as a data URL
                    let reader = new FileReader();
                    reader.onload = function (e) {
                        callback(e.target.result)
                    };
                    reader.readAsDataURL(file);
                } else {
                    // If the item is a URL, retrieve it and use it to load the image
                    imageItem.getAsString(callback)
                }
            }
        }

        dropAreaI2I.addEventListener("drop", function (event) {
            event.stopPropagation()
            event.preventDefault()
            hideDropAreas()

            getImageFromDropEvent(event, e => imageObj.src = e)
        })

        dropAreaMD.addEventListener("dragenter", function (event) {
            event.preventDefault()
            dropAreaMD.style.backgroundColor = 'darkGreen'
        })
        dropAreaMD.addEventListener("dragleave", function (event) {
            event.preventDefault()
            dropAreaMD.style.backgroundColor = ''
        })

        dropAreaMD.addEventListener("drop", function (event) {
            let items = []
            hideDropAreas()
            if (event?.dataTransfer?.items) { // Use DataTransferItemList interface
                items = Array.from(event.dataTransfer.items)
                items = items.filter(item => item.kind === 'file' && (item.type === 'image/png' || item.type === 'image/jpeg' || item.type === 'image/webp'))
                items = items.map(item => item.getAsFile())
            } else if (event?.dataTransfer?.files) { // Use DataTransfer interface
                items = Array.from(event.dataTransfer.files)
            }
            // check if image has embedded metadata, load task if it does
            if (items[0].type === "image/png") {
                readPNGMetadata(items[0])
            } else if (items[0].type === "image/jpeg" || items[0].type === "image/webp") {
                readJPEGMetadata(items[0]);
            } else {
                console.log("File must be a PNG, WEBP or JPEG image.");
            }
            event.preventDefault()
        })

        document.addEventListener("drop", function (event) {
            event.preventDefault()
            hideDropAreas()
        })

        document.addEventListener("dragexit", function (event) {
            event.preventDefault()
            hideDropAreas()
        })
    }

    function showDropAreasDnD(event) {
        event.preventDefault()
        // Find the first image file, uri, or moz-url in the items list
        let imageItem = null;
        for (let i = 0; i < event.dataTransfer.items.length; i++) {
            let item = event.dataTransfer.items[i];
            if ((item.kind === 'file' && item.type.startsWith('image/')) || item.type === 'text/uri-list') {
                imageItem = item;
                break;
            } else if (item.type === 'text/x-moz-url') {
                // If there are no image files or uris, fallback to moz-url
                if (!imageItem) {
                    imageItem = item;
                }
            }
        }

        if (imageItem) {
            showDropAreas()
        }
    }

    function hideDropAreasDnD(event) {
        if (event.fromElement && !document.querySelector('#editor').contains(event.fromElement) && !document.querySelector('#editor').contains(event.fromElement.parentNode.host)) {
            hideDropAreas()
        }
    }

    function showDropAreas() {
        const dropAreas = document.querySelectorAll(".drop-area")
        dropAreas.forEach(function (dropArea) {
            dropArea.style.display = 'inline-block'
        })
    }

    function hideDropAreas() {
        const dropAreas = document.querySelectorAll(".drop-area")
        dropAreas.forEach(function (dropArea) {
            dropArea.style.display = 'none'
            dropArea.style.backgroundColor = ''
        })
    }

    const dndContainer = document.getElementById("editor-inputs-init-image")
    createDropAreas(dndContainer)
    document.querySelector('#editor').addEventListener("dragenter", showDropAreasDnD)
    document.querySelector('#editor').addEventListener("dragleave", hideDropAreasDnD)

    /* METADATA EXTRACTION HELPER FUNCTION */
    function clearAllImageTagCards() {
        // clear existing image tag cards
        editorTagsContainer.style.display = 'none'
        editorModifierTagsList.querySelectorAll('.modifier-card').forEach(modifierCard => {
            modifierCard.remove()
        })

        // reset modifier cards state
        document.querySelector('#editor-modifiers').querySelectorAll('.modifier-card').forEach(modifierCard => {
            const modifierName = modifierCard.querySelector('.modifier-card-label').innerText
            if (activeTags.map(x => x.name).includes(modifierName)) {
                modifierCard.classList.remove(activeCardClass)
                modifierCard.querySelector('.modifier-card-image-overlay').innerText = '+'
            }
        })
        activeTags = []
        document.dispatchEvent(new Event('refreshImageModifiers')) // notify the 
    }

    /* PNG METADATA EXTRACTION */

    function readPNGMetadata(image) {
        const fileReader = new FileReader()
        fileReader.onload = function () {
            extractTextChunks(image).then(function (chunks) {
                let reqBody = {}
                for (let key in chunks) {
                    reqBody[key] = chunks[key]
                }
                if (Object.keys(reqBody).length !== 0) {
                    if (reqBody["seed"] !== undefined) {
                        let task = { numOutputsTotal: reqBody["num_outputs"], seed: reqBody["seed"] }
                        task['reqBody'] = reqBody
                        clearAllImageTagCards()
                        restoreTaskToUI(task, TASK_REQ_NO_EXPORT)
                    }
                }
            }).catch(function (error) {
                console.error(error);
            })
        }
        fileReader.readAsArrayBuffer(image);
    }

    function extractTextChunks(file) {
        return new Promise(function (resolve, reject) {
            let reader = new FileReader();
            reader.onload = function () {
                let arrayBuffer = reader.result;
                let dataView = new DataView(arrayBuffer);

                // Verify that the PNG signature is present
                let signature = new Uint8Array(arrayBuffer, 0, 8);
                if (String.fromCharCode.apply(null, signature) !== "\x89PNG\r\n\x1a\n") {
                    reject(new Error("Invalid PNG file"));
                    return;
                }

                // Iterate through the chunks
                let chunks = {};
                let offset = 8;
                while (offset < arrayBuffer.byteLength) {
                    // Get the length and type of the chunk
                    let length = dataView.getUint32(offset);
                    let type = String.fromCharCode(dataView.getUint8(offset + 4), dataView.getUint8(offset + 5), dataView.getUint8(offset + 6), dataView.getUint8(offset + 7));
                    offset += 8;

                    // Get the data of the chunk
                    let data = new Uint8Array(arrayBuffer, offset, length);
                    offset += length;

                    // Get the CRC of the chunk
                    let crc = dataView.getUint32(offset);
                    offset += 4;

                    // If it's a tEXt chunk, convert the data to a human-readable string
                    if (type === "tEXt") {
                        let nullIndex = data.indexOf(0);
                        let key = String.fromCharCode.apply(null, data.slice(0, nullIndex));
                        let value = String.fromCharCode.apply(null, data.slice(nullIndex + 1));
                        chunks[key] = value;
                    }
                }
                resolve(chunks);
            };
            reader.readAsArrayBuffer(file);
        });
    };

    /* JPEG or WEBP METADATA EXTRACTION */
    function readJPEGMetadata(image) {
        const fileReader = new FileReader()
        fileReader.onload = function (e) {
            ExifReader.load(e.target.result).then(tags => {
                const exifData = String.fromCharCode(...tags['UserComment'].value)
                if (exifData !== undefined) {
                    try {
                        const isUnicode = (exifData.toLowerCase().startsWith('unicode'))
                        let keys = JSON.parse(isUnicode ? decodeUnicode(exifData.slice(8)) : exifData.slice(8))
                        let reqBody = {}
                        for (let key in keys) {
                            reqBody[key] = keys[key]
                        }
                        let task = { numOutputsTotal: reqBody["num_outputs"], seed: reqBody["seed"] }
                        task['reqBody'] = reqBody
                        clearAllImageTagCards()
                        restoreTaskToUI(task, TASK_REQ_NO_EXPORT)
                    } catch (e) {
                        console.error('No valid JSON in EXIF data')
                    }
                }
            })

        }
        fileReader.readAsDataURL(image);
    }

    function decodeUnicode(unicodeString) {
        const encoder = new TextEncoder()
        const input = new Uint16Array(encoder.encode(unicodeString))

        let decodedString = ''
        for (let i = 0; i < input.length; i += 2) {
            decodedString += String.fromCharCode(input[i] << 8 | input[i + 1])
        }

        return decodedString
    }

    /* MAGIC WAND MASK TOOL (INPAINTER ONLY) */
    const magicWandTool = {
        id: "magicwand",
        name: "Magic Wand",
        icon: "fa-solid fa-wand-magic-sparkles",
        cursor: "crosshair",
        begin(editor, targetContext, x, y, isOverlay = false) {
            if (isOverlay || !editor.inpainter) return
            const width = editor.width
            const height = editor.height
            const startX = Math.floor(x)
            const startY = Math.floor(y)
            if (startX < 0 || startY < 0 || startX >= width || startY >= height) return

            const source = editor.layers.background.ctx.getImageData(0, 0, width, height).data
            const mask = targetContext.getImageData(0, 0, width, height)
            const output = mask.data
            const visited = new Uint8Array(width * height)
            const stack = [startY * width + startX]
            const baseOffset = (startY * width + startX) * 4
            const baseR = source[baseOffset]
            const baseG = source[baseOffset + 1]
            const baseB = source[baseOffset + 2]
            const threshold = Number(editor.magicWandThreshold || 30)
            const thresholdSquared = threshold * threshold
            const opacity = Math.round(255 * (1 - Number(editor.options.opacity || 0)))

            while (stack.length) {
                const pixel = stack.pop()
                if (visited[pixel]) continue
                visited[pixel] = 1
                const offset = pixel * 4
                const red = source[offset] - baseR
                const green = source[offset + 1] - baseG
                const blue = source[offset + 2] - baseB
                if (red * red + green * green + blue * blue > thresholdSquared) continue

                output[offset] = 255
                output[offset + 1] = 255
                output[offset + 2] = 255
                output[offset + 3] = opacity

                const px = pixel % width
                if (px > 0) stack.push(pixel - 1)
                if (px + 1 < width) stack.push(pixel + 1)
                if (pixel >= width) stack.push(pixel - width)
                if (pixel + width < width * height) stack.push(pixel + width)
            }
            targetContext.putImageData(mask, 0, 0)
        },
        move() {},
        end() {},
        // No global hotkey: the Draw editor shares the tool registry but does
        // not expose this inpainting-only option.
        hotkey: null,
    }

    function installMagicWand() {
        if (IMAGE_EDITOR_TOOLS.some((tool) => tool.id === magicWandTool.id)) return
        IMAGE_EDITOR_TOOLS.push(magicWandTool)
        const toolSection = IMAGE_EDITOR_SECTIONS.find((section) => section.name === "tool")
        if (!toolSection.options.includes(magicWandTool.id)) toolSection.options.push(magicWandTool.id)

        const holder = document.createElement("div")
        const button = document.createElement("div")
        holder.appendChild(button)
        toolSection.initElement(button, magicWandTool.id)
        const toolIndex = IMAGE_EDITOR_TOOLS.findIndex((tool) => tool.id === magicWandTool.id)
        button.addEventListener("click", () => imageInpainter.selectOption("tool", toolIndex))
        imageInpainter.popup.querySelector(".image_editor_tool .editor-options-container").appendChild(holder)
        imageInpainter.optionElements.tool.push(button)

        const template = document.getElementById("inpainting-magic-wand-controls")
        const controls = template.content.cloneNode(true)
        imageInpainter.popup.querySelector(".editor-controls-left").appendChild(controls)
        const threshold = imageInpainter.popup.querySelector("#inpainting-magic-wand-threshold")
        const value = imageInpainter.popup.querySelector("#inpainting-magic-wand-threshold-value")
        const saved = localStorage.getItem("inpainting_magic_wand_threshold")
        threshold.value = saved || "30"
        imageInpainter.magicWandThreshold = Number(threshold.value)
        value.textContent = threshold.value
        threshold.addEventListener("input", () => {
            imageInpainter.magicWandThreshold = Number(threshold.value)
            value.textContent = threshold.value
            localStorage.setItem("inpainting_magic_wand_threshold", threshold.value)
        })
    }

    installMagicWand()
})()
