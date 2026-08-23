;(function () {
    "use strict"

    const row = document.querySelector("#controlnet_model_container")
    const outputSettings = document.querySelector("#output-settings")
    if (!row || !outputSettings?.parentNode || document.querySelector("#sdkit3-controlnet-panel")) return

    const panel = document.createElement("div")
    panel.id = "sdkit3-controlnet-panel"
    panel.className = "settings-box panel-box sdkit3-extra-settings-panel"
    panel.innerHTML = `
        <h4 class="collapsible">ControlNet <small>image conditioning</small></h4>
        <div class="collapsible-content controlled-generation-content">
            <table class="controlled-generation-image-table"><tbody id="controlled-generation-controlnet"></tbody></table>
        </div>`

    outputSettings.parentNode.insertBefore(panel, outputSettings.nextSibling)
    panel.querySelector("#controlled-generation-controlnet").appendChild(row)

    const style = document.createElement("style")
    style.textContent = `
        #sdkit3-controlnet-panel { margin-top: 10px; }
        #sdkit3-controlnet-panel h4 { cursor: pointer; }
        #sdkit3-controlnet-panel h4 small { float: right; }
        #sdkit3-controlnet-panel .controlled-generation-content { padding-top: 8px; }
        #controlled-generation-controlnet, #sdkit3-controlnet-panel table { width: 100%; }
        #sdkit3-controlnet-panel table > tbody > tr > td:first-child { padding-right: 4px; white-space: nowrap; vertical-align: top; text-align: right; }
    `
    document.head.appendChild(style)

    if (typeof createCollapsibles === "function") createCollapsibles(panel)
    if (typeof prettifyInputs === "function") prettifyInputs(panel)
})()
