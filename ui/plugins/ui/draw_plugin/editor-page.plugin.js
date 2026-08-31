;(function () {
    "use strict"

    function initializeEditorPage() {
        if (
            typeof createTab !== "function" ||
            typeof imageEditor === "undefined" ||
            typeof imageInpainter === "undefined" ||
            document.getElementById("tab-image-editor-page")
        ) {
            return false
        }

        const panel = document.createElement("div")
        panel.id = "image-editor-page-panel"
        panel.innerHTML = window.loadRequiredPluginHTML("/plugins/core/draw_plugin/editor-page.plugin.html")

        let activeMode = "draw"
        const activate = (mode) => {
            if (mode === "generate") {
                imageEditor.deactivateInput?.()
                imageInpainter.deactivateInput?.()
                selectTab("tab-main")
                return
            }
            activeMode = mode === "inpaint" ? "inpaint" : "draw"
            const activeEditor = activeMode === "inpaint" ? imageInpainter : imageEditor
            const inactiveEditor = activeMode === "inpaint" ? imageEditor : imageInpainter
            inactiveEditor.popup.classList.remove("active")
            inactiveEditor.deactivateInput?.()
            activeEditor.popup.classList.add("active")
            activeEditor.activateInput?.()
            panel.querySelectorAll("[data-editor-mode]").forEach((button) => {
                if (button.dataset.editorMode === "generate") return
                button.classList.toggle("primaryButton", button.dataset.editorMode === activeMode)
                button.classList.toggle("secondaryButton", button.dataset.editorMode !== activeMode)
            })
            requestAnimationFrame(() => activeEditor.fitToViewport?.())
        }

        createTab({
            id: "image-editor-page",
            label: "Inpaint / Draw",
            icon: "paintbrush",
            content: panel,
            onOpen: () => activate(activeMode),
            css: `
                #tab-content-image-editor-page .tab-content-inner {
                    max-width: none;
                    width: 100%;
                    min-width: 0;
                    padding: clamp(6px, 1vw, 12px);
                }
                #tab-content-image-editor-page {
                    overflow-x: clip;
                }
                #image-editor-page-panel,
                #image-editor-page-content {
                    min-width: 0;
                }
                .image-editor-page-toolbar {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    flex-wrap: wrap;
                    text-align: left;
                    margin-bottom: 10px;
                }
                .image-editor-page-toolbar span {
                    color: var(--text-color2);
                    font-size: 0.85em;
                    flex: 1 1 280px;
                }
                .editor-page-pane:not(.active) { display: none; }
                .editor-page-pane {
                    position: relative;
                    background: none;
                    opacity: 1;
                    visibility: visible;
                    z-index: auto;
                }
                .editor-page-pane > div {
                    max-width: none;
                    min-width: 0;
                    width: 100%;
                    margin: 0;
                    padding: 8px;
                    border: 0;
                    box-shadow: none;
                    background: none;
                }
                .editor-page-pane > div > .flex-container {
                    --editor-stage-height: clamp(420px, calc(100vh - 190px), 900px);
                    --editor-stage-height: clamp(420px, calc(100dvh - 190px), 900px);
                    display: grid;
                    grid-template-columns:
                        minmax(180px, clamp(200px, 16vw, 240px))
                        minmax(320px, 1fr)
                        minmax(180px, clamp(200px, 16vw, 240px));
                    grid-template-areas: "tools canvas actions";
                    align-items: stretch;
                    gap: clamp(8px, 1vw, 14px);
                    min-width: 0;
                }
                .editor-page-pane .close-button { display: none; }
                .editor-page-pane .editor-controls-left {
                    grid-area: tools;
                }
                .editor-page-pane .editor-controls-center {
                    grid-area: canvas;
                    position: relative;
                    width: 100%;
                    min-width: 0;
                    height: var(--editor-stage-height);
                    min-height: 0;
                    overflow: hidden;
                    padding: clamp(8px, 1vw, 14px);
                    border-radius: 8px;
                    background: var(--background-color1);
                    box-shadow: inset 0 0 0 1px var(--background-color3);
                    contain: layout paint;
                    isolation: isolate;
                }
                .editor-page-pane .editor-controls-center > div {
                    flex: 0 0 auto;
                    margin: 0;
                    will-change: transform;
                }
                .editor-page-pane .editor-controls-left,
                .editor-page-pane .editor-controls-right {
                    width: 100%;
                    min-width: 0;
                    max-width: none;
                    max-height: var(--editor-stage-height);
                    overflow-x: hidden;
                    overflow-y: auto;
                    padding: 10px;
                    border: 1px solid var(--background-color3);
                    border-radius: 8px;
                    background: var(--background-color2);
                    scrollbar-gutter: stable;
                }
                .editor-page-pane .editor-controls-right {
                    grid-area: actions;
                }
                .editor-page-pane .editor-controls-right > div {
                    min-width: 0;
                }
                .editor-page-pane .editor-controls-right > div:last-child {
                    min-width: 0;
                }
                .editor-page-pane .editor-options-container {
                    flex-wrap: wrap;
                }
                .editor-page-pane.inpainter .editor-controls-left > *,
                .editor-page-pane.inpainter .editor-controls-right > * {
                    min-width: 0;
                    max-width: 100%;
                }
                .editor-page-pane.inpainter .image_editor_magicwand_threshold label {
                    display: grid;
                    grid-template-columns: auto minmax(0, 1fr) auto;
                    align-items: center;
                    gap: 8px;
                }
                .editor-page-pane.inpainter .image_editor_magicwand_threshold input[type="range"] {
                    width: 100%;
                    min-width: 0;
                }
                .editor-page-pane.inpainter .image_editor_magicwand_threshold small {
                    display: block;
                    margin-top: 5px;
                    overflow-wrap: anywhere;
                }

                @media screen and (max-width: 1100px) {
                    .editor-page-pane > div > .flex-container {
                        --editor-stage-height: clamp(360px, 60vh, 720px);
                        --editor-stage-height: clamp(360px, 60dvh, 720px);
                        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                        grid-template-areas:
                            "canvas canvas"
                            "tools actions";
                    }
                    .editor-page-pane .editor-controls-left,
                    .editor-page-pane .editor-controls-right {
                        max-height: none;
                        overflow: visible;
                    }
                }
                @media screen and (max-width: 700px) {
                    .image-editor-page-toolbar {
                        position: sticky;
                        top: 0;
                        z-index: 5;
                    }
                    .image-editor-page-toolbar button {
                        min-height: 42px;
                    }
                    .image-editor-page-toolbar span {
                        flex-basis: 100%;
                        font-size: 0.78em;
                    }
                    .editor-page-pane > div > .flex-container {
                        grid-template-columns: 1fr;
                        grid-template-areas:
                            "canvas"
                            "tools"
                            "actions";
                        --editor-stage-height: clamp(280px, 54vh, 620px);
                        --editor-stage-height: clamp(280px, 54dvh, 620px);
                    }
                    .editor-page-pane .editor-controls-left,
                    .editor-page-pane .editor-controls-right {
                        padding: 8px;
                    }
                    .editor-page-pane .editor-controls-left {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                        align-items: start;
                        gap: 8px 12px;
                    }
                    .editor-page-pane .image_editor_tool,
                    .editor-page-pane .image_editor_brush_size {
                        grid-column: 1 / -1;
                    }
                    .editor-page-pane .image_editor_tool .editor-options-container {
                        grid-template-columns: repeat(2, minmax(0, 1fr));
                    }
                    .editor-page-pane .editor-controls-center {
                        padding: 6px;
                    }
                    .editor-page-pane .editor-options-container > * {
                        min-width: 40px;
                        min-height: 40px;
                        cursor: pointer;
                    }
                    .editor-page-pane .image-editor-button {
                        min-height: 42px;
                    }
                }
            `,
        })

        const pageContent = panel.querySelector("#image-editor-page-content")
        ;[imageEditor, imageInpainter].forEach((editor) => {
            editor.popup.classList.remove("popup", "active")
            editor.popup.classList.add("editor-page-pane")
            pageContent.appendChild(editor.popup)
        })

        panel.querySelectorAll("[data-editor-mode]").forEach((button) => {
            button.addEventListener("click", () => activate(button.dataset.editorMode))
        })

        // Civitai is also loaded as a plugin and may finish after this page.
        // Retry until both tab pairs exist so the requested ordering is stable.
        const placeBesideCivitai = () => {
            const civitaiTab = document.getElementById("tab-civitai")
            const editorTab = document.getElementById("tab-image-editor-page")
            const civitaiContent = document.getElementById("tab-content-civitai")
            const editorContent = document.getElementById("tab-content-image-editor-page")
            if (!civitaiTab || !editorTab || !civitaiContent || !editorContent) return false
            civitaiTab.insertAdjacentElement("afterend", editorTab)
            civitaiContent.insertAdjacentElement("afterend", editorContent)
            return true
        }
        if (!placeBesideCivitai()) {
            let remainingAttempts = 100
            const placementTimer = setInterval(() => {
                remainingAttempts -= 1
                if (placeBesideCivitai() || remainingAttempts <= 0) clearInterval(placementTimer)
            }, 100)
        }

        window.openImageEditorPage = (mode) => {
            selectTab("tab-image-editor-page")
            activate(mode)
        }
        window.closeImageEditorPage = () => {
            imageEditor.deactivateInput?.()
            imageInpainter.deactivateInput?.()
            selectTab("tab-main")
        }
        activate("draw")
        imageEditor.popup.classList.remove("active")
        imageEditor.deactivateInput?.()

        document.addEventListener("tabClick", (event) => {
            if (event.detail?.name === "image-editor-page") return
            imageEditor.deactivateInput?.()
            imageInpainter.deactivateInput?.()
        })
        return true
    }

    const timer = setInterval(() => {
        if (initializeEditorPage()) clearInterval(timer)
    }, 100)
})()
