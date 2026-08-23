// Initial and Reference Images plugin for Easy Diffusion
// Owns both image-input UIs while preserving Easy Diffusion's native IDs.
// v2.0.0, last updated: 8/22/2026

;(function () {
    "use strict"

    function ensureInitialImageUI() {
        const existing = document.getElementById("initial-image-settings")
        if (existing) return existing

        const editorSettings = document.getElementById("editor-settings")
        if (!editorSettings?.parentNode) {
            throw new Error("Initial and Reference Images plugin: Image Settings panel was not found")
        }

        const panel = document.createElement("div")
        panel.id = "initial-image-settings"
        panel.className = "settings-box panel-box"
        panel.innerHTML = `
            <h4 class="collapsible">
                Initial Image (img2img) <small>(optional)</small>
                <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top">
                    Add an img2img source using Browse, drag and drop, a rendered image, or the clipboard with Ctrl+V.<br><br>
                    You may also reload metadata embedded in a PNG, WEBP, or JPEG image.
                </span></i>
            </h4>
            <div id="editor-inputs-init-image" class="collapsible-content settings-panel-entries">
                <div>
                    <div id="init_image_preview_container" class="image-input-section image-input-section--single">
                        <div id="init_image_wrapper" class="image-input-wrapper">
                            <img id="init_image_preview" class="input-image-preview" src="" crossorigin="anonymous">
                            <span id="init_image_size_box" class="img_bottom_label"></span>
                            <button class="init_image_clear image_clear_btn"><i class="fa-solid fa-xmark"></i></button>
                        </div>
                        <div id="init_image_buttons" class="image-input-buttons">
                            <div class="button">
                                <i class="fa-regular fa-folder-open"></i>
                                Browse
                                <input id="init_image" name="init_image" type="file">
                            </div>
                            <div id="init_image_button_draw" class="button">
                                <i class="fa-solid fa-pencil"></i>
                                Draw
                            </div>
                            <div id="inpaint_button_container">
                                <div id="init_image_button_inpaint" class="button">
                                    <i class="fa-solid fa-paintbrush"></i>
                                    Inpaint
                                </div>
                                <input id="enable_mask" name="enable_mask" type="checkbox">
                            </div>
                        </div>
                    </div>
                    <div id="apply_color_correction_setting" class="pl-5"><input id="apply_color_correction" name="apply_color_correction" type="checkbox"> <label for="apply_color_correction">Preserve color profile <small>(helps during inpainting)</small></label></div>
                    <div id="strict_mask_border_setting" class="pl-5"><input id="strict_mask_border" name="strict_mask_border" type="checkbox"> <label for="strict_mask_border">Strict Mask Border <small>(won't modify outside the mask, but the mask border might be visible)</small></label></div>
                </div>

                <div id="editor-inputs-ref-images" class="row gated-feature" data-feature-keys="backend_sdkit3">
                    <label>Reference Images <small>(optional)</small></label>
                    <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top">
                        Add reference images for vision-based models (e.g. Qwen image editing). Add one or more using Browse or drag and drop.<br><br>
                        Reference images guide the model's visual understanding during generation.
                    </span></i>
                    <div id="ref_images_preview_container" class="image-input-section image-input-section--multi">
                        <div id="ref_images_list" class="image-input-list"></div>
                        <div id="ref_images_buttons" class="image-input-buttons">
                            <div class="button">
                                <i class="fa-regular fa-folder-open"></i>
                                Browse
                                <input id="ref_image_input" name="ref_image_input" type="file" accept="image/*" multiple>
                            </div>
                            <button id="ref_images_clear_all" class="tertiaryButton smallButton displayNone">Clear All</button>
                        </div>
                    </div>
                </div>
            </div>`

        editorSettings.parentNode.insertBefore(panel, editorSettings)
        return panel
    }

    window.ensureInitialImageUI = ensureInitialImageUI
    ensureInitialImageUI()

    // This file is bootstrapped before main.js and discovered again by the
    // normal core-plugin loader. Install its behavior only once.
    if (window.__initialAndReferenceImagesPluginLoaded) return
    window.__initialAndReferenceImagesPluginLoaded = true

    let refImagesManuallyEnabled = false

    const getRefContainer = () => document.getElementById("editor-inputs-ref-images")
    const getToggleRow = () => document.getElementById("ref-images-plugin-toggle-row")
    const getToggleButton = () => document.getElementById("ref-images-plugin-btn")
    const isFlux = () => typeof window.isFluxModel === "function" && window.isFluxModel()

    function applyManualState() {
        getRefContainer()?.classList.toggle("displayNone", !refImagesManuallyEnabled)
    }

    function updateToggleRowVisibility() {
        const toggleRow = getToggleRow()
        const container = getRefContainer()
        if (!toggleRow || !container) return

        const fluxModel = isFlux()
        toggleRow.style.display = container.style.display !== "none" && !fluxModel ? "" : "none"
        if (fluxModel && refImagesManuallyEnabled) {
            refImagesManuallyEnabled = false
            getToggleButton()?.classList.remove("active")
        }
    }

    function patchCheckReferenceImageField() {
        if (typeof window.checkReferenceImageField !== "function") {
            setTimeout(patchCheckReferenceImageField, 200)
            return
        }
        const original = window.checkReferenceImageField
        window.checkReferenceImageField = function () {
            original.apply(this, arguments)
            if (refImagesManuallyEnabled) applyManualState()
            updateToggleRowVisibility()
        }
    }

    function setupModelWatcher() {
        const modelElement = document.querySelector("#editor-settings #stable_diffusion_model")
        if (!modelElement) {
            setTimeout(setupModelWatcher, 500)
            return
        }
        modelElement.addEventListener("change", updateToggleRowVisibility)
        new MutationObserver(updateToggleRowVisibility).observe(modelElement, { attributes: true })
    }

    function insertToggleButton() {
        if (getToggleRow()) return
        const container = getRefContainer()
        if (!container) {
            setTimeout(insertToggleButton, 500)
            return
        }

        const toggleRow = document.createElement("div")
        toggleRow.id = "ref-images-plugin-toggle-row"
        toggleRow.className = "row"
        toggleRow.style.display = "none"

        const button = document.createElement("button")
        button.id = "ref-images-plugin-btn"
        button.className = "tertiaryButton smallButton"
        button.innerHTML = '<i class="fa-solid fa-images"></i> Reference Images'
        button.title = "Toggle reference images for vision-based models (e.g. Qwen image editing)"
        button.addEventListener("click", () => {
            refImagesManuallyEnabled = !refImagesManuallyEnabled
            button.classList.toggle("active", refImagesManuallyEnabled)
            applyManualState()
        })

        toggleRow.appendChild(button)
        container.parentNode.insertBefore(toggleRow, container)
        new MutationObserver(updateToggleRowVisibility).observe(container, { attributes: true, attributeFilter: ["style"] })
        patchCheckReferenceImageField()
        setupModelWatcher()
        updateToggleRowVisibility()
    }

    insertToggleButton()
})()
