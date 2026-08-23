;(function () {
    "use strict"

    const preprocessor = document.querySelector("#control_image_filter")
    const outputSettings = document.querySelector("#output-settings")
    if (!preprocessor || !outputSettings?.parentNode || document.querySelector("#sdkit3-controlnet-preprocessor-panel")) return

    const oldLabel = preprocessor.previousElementSibling
    const oldBreak = preprocessor.nextElementSibling
    const panel = document.createElement("div")
    panel.id = "sdkit3-controlnet-preprocessor-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel"
    panel.innerHTML = `
        <h4 class="collapsible">ControlNet Preprocessor <small>image processing</small></h4>
        <div class="collapsible-content controlled-generation-content">
            <div class="controlled-generation-grid">
                <label for="control_image_filter">Preprocessor</label>
                <div id="controlled-generation-controlnet-preprocessor"></div>
            </div>
            <small>Select None to pass the control image through without preprocessing.</small>
        </div>`

    const anchor = document.querySelector("#sdkit3-controlnet-panel") || outputSettings
    anchor.parentNode.insertBefore(panel, anchor.nextSibling)
    panel.querySelector("#controlled-generation-controlnet-preprocessor").appendChild(preprocessor)
    if (oldLabel?.tagName === "LABEL") oldLabel.remove()
    if (oldBreak?.tagName === "BR") oldBreak.remove()

    const style = document.createElement("style")
    style.textContent = `
        #sdkit3-controlnet-preprocessor-panel { margin-top: 10px; }
        #sdkit3-controlnet-preprocessor-panel h4 { cursor: pointer; }
        #sdkit3-controlnet-preprocessor-panel h4 small { float: right; }
        #sdkit3-controlnet-preprocessor-panel .controlled-generation-content { padding-top: 8px; }
        #sdkit3-controlnet-preprocessor-panel .controlled-generation-grid { display: grid; grid-template-columns: minmax(105px, auto) minmax(0, 1fr); gap: 6px 9px; align-items: center; margin: 7px 0; }
        #controlled-generation-controlnet-preprocessor #control_image_filter { width: 100%; max-width: 100%; }
    `
    document.head.appendChild(style)

    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)
})()
