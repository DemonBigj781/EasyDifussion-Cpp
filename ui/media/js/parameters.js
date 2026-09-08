/**
 * Enum of parameter types
 * @readonly
 * @enum {string}
 */
var ParameterType = {
    checkbox: "checkbox",
    select: "select",
    select_multiple: "select_multiple",
    slider: "slider",
    custom: "custom",
}

/**
 * Element shortcuts
 */
let parametersTable = document.querySelector("#system-settings-table")
let networkParametersTable = document.querySelector("#system-settings-network-table")
let installExtrasTable = document.querySelector("#system-settings-install-extras-table")

/**
 * JSDoc style
 * @typedef {object} Parameter
 * @property {string} id
 * @property {keyof ParameterType} type
 * @property {string | (parameter: Parameter) => (HTMLElement | string)} label
 * @property {string | (parameter: Parameter) => (HTMLElement | string) | undefined} note
 * @property {(parameter: Parameter) => (HTMLElement | string) | undefined} render
 * @property {string | undefined} icon
 * @property {number|boolean|string} default
 * @property {boolean?} saveInAppConfig
 */

const NATIVE_BACKEND_BOOLEAN_OPTIONS = [
    { flag: "--diffusion-fa", label: "Diffusion flash attention", group: "performance" },
    { flag: "--flash-attention", label: "Flash attention for all modules", group: "performance" },
    { flag: "--sage-attention", label: "SageAttention SM80", group: "performance" },
    { flag: "--xformers", label: "xFormers-compatible fused attention", group: "performance" },
    { flag: "--cuda-malloc", label: "Legacy cudaMalloc pool", group: "performance" },
    { flag: "--keep-model-loaded", label: "Keep model weights loaded", group: "performance" },
    { flag: "--offload-to-cpu", label: "Image/model parameter CPU offload", group: "image" },
    { flag: "--image-clip-on-cpu", label: "Image text encoder on CPU", group: "image" },
    { flag: "--image-vae-on-cpu", label: "Image VAE on CPU", group: "image" },
    { flag: "--control-net-cpu", label: "ControlNet on CPU", group: "image" },
    { flag: "--vae-tiling", label: "VAE tiling", group: "image" },
    { flag: "--stream-layers", label: "Stream image/model layers", group: "image" },
    { flag: "--video-clip-on-cpu", label: "Video text encoder on CPU", group: "video" },
    { flag: "--video-vae-on-cpu", label: "Video VAE on CPU", group: "video" },
    { flag: "--video-offload-to-cpu", label: "Video diffusion CPU offload", group: "video" },
    { flag: "--video-stream-layers", label: "Stream video diffusion layers", group: "video" },
]

const NATIVE_BACKEND_VALUE_OPTIONS = [
    { flag: "--vae-tiles", label: "VAE latent tile size", group: "image", type: "number", min: 4, step: 1, placeholder: 32 },
    { flag: "--vae-tiled-overlap", label: "VAE tile overlap", group: "image", type: "number", min: 0, step: 1, placeholder: 16 },
    { flag: "--vae-tile-size", label: "VAE pixel tile size", group: "image", type: "text", placeholder: "256x256" },
    { flag: "--max-vram", label: "Image/model VRAM budget", group: "image", type: "text", placeholder: "6 or cuda=6,cpu=0" },
    { flag: "--video-max-vram", label: "Video VRAM budget (GiB)", group: "video", type: "number", min: 0, step: 0.25, placeholder: 8 },
]

function nativeBackendArgumentId(flag) {
    return `native-backend-arg-${flag.slice(2).replaceAll("-", "_")}`
}

function nativeBackendToggleMarkup(option) {
    const id = nativeBackendArgumentId(option.flag)
    return `<div class="native-backend-argument-toggle">
        <div><label for="${id}">${option.label}</label><small>${option.flag}</small></div>
        <div class="input-toggle"><input id="${id}" type="checkbox"><label for="${id}"></label></div>
    </div>`
}

function nativeBackendValueMarkup(option) {
    const id = nativeBackendArgumentId(option.flag)
    const min = option.min === undefined ? "" : ` min="${option.min}"`
    const step = option.step === undefined ? "" : ` step="${option.step}"`
    return `<label class="native-backend-argument-value" for="${id}">
        <span>${option.label}<small>${option.flag}</small></span>
        <input id="${id}" type="${option.type}"${min}${step} placeholder="${option.placeholder || ""}">
    </label>`
}

function renderNativeBackendArgumentsEditor(parameter) {
    const optionsFor = (group) => [
        ...NATIVE_BACKEND_BOOLEAN_OPTIONS.filter((option) => option.group === group).map(nativeBackendToggleMarkup),
        ...NATIVE_BACKEND_VALUE_OPTIONS.filter((option) => option.group === group).map(nativeBackendValueMarkup),
    ].join("")
    return `<div id="native-backend-arguments-editor" class="native-backend-arguments-editor">
        <input id="${parameter.id}" name="${parameter.id}" type="hidden">
        <label class="native-backend-argument-value" for="native-backend-arg-log_level">
            <span>Log level<small>--log-level</small></span>
            <select id="native-backend-arg-log_level">
                <option value="">Default (info)</option>
                <option value="verbose">Verbose</option>
                <option value="debug">Debug</option>
                <option value="info">Info</option>
                <option value="warning">Warning</option>
                <option value="error">Error</option>
            </select>
        </label>
        <label class="native-backend-argument-value" for="native-backend-arg-mmap_mode">
            <span>Weight memory mapping<small>--mmap / --no-mmap</small></span>
            <select id="native-backend-arg-mmap_mode">
                <option value="">Automatic</option>
                <option value="--mmap">Enabled</option>
                <option value="--no-mmap">Disabled</option>
            </select>
        </label>
        <details open><summary>Performance</summary><div class="native-backend-argument-grid">${optionsFor("performance")}</div></details>
        <details open><summary>Image and shared memory</summary><div class="native-backend-argument-grid">${optionsFor("image")}</div></details>
        <details open><summary>Native video memory</summary><div class="native-backend-argument-grid">${optionsFor("video")}</div></details>
        <label class="native-backend-additional-arguments" for="native-backend-additional-arguments">
            Additional arguments
            <textarea id="native-backend-additional-arguments" rows="2" cols="48" spellcheck="false" placeholder="Uncommon or newly added flags"></textarea>
        </label>
        <small id="native-backend-arguments-preview" class="native-backend-arguments-preview"></small>
    </div>`
}

/** @type {Array.<Parameter>} */
var PARAMETERS = [
    {
        id: "theme",
        type: ParameterType.select,
        label: "Theme",
        default: "theme-default",
        note: "customize the look and feel of the ui",
        options: [
            // Note: options expanded dynamically
            {
                value: "theme-default",
                label: "Default",
            },
        ],
        icon: "fa-palette",
    },
    {
        id: "save_to_disk",
        type: ParameterType.checkbox,
        label: "Auto-Save Images",
        note: "automatically saves images to the specified location",
        icon: "fa-download",
        default: false,
    },
    {
        id: "diskPath",
        type: ParameterType.custom,
        label: "Save Location",
        render: (parameter) => {
            return `<input id="${parameter.id}" name="${parameter.id}" size="30" disabled>`
        },
    },
    {
        id: "metadata_output_format",
        type: ParameterType.select,
        label: "Metadata format",
        note: "will be saved to disk in this format",
        default: "txt",
        options: [
            {
                value: "none",
                label: "none",
            },
            {
                value: "txt",
                label: "txt",
            },
            {
                value: "json",
                label: "json",
            },
            {
                value: "embed",
                label: "embed",
            },
            {
                value: "embed,txt",
                label: "embed & txt",
            },
            {
                value: "embed,json",
                label: "embed & json",
            },
        ],
    },
    {
        id: "models_dir",
        type: ParameterType.custom,
        icon: "fa-folder-tree",
        label: "Models Folder",
        note: "Path to the 'models' folder. Please save and restart Easy Diffusion after changing this.",
        saveInAppConfig: true,
        render: (parameter) => {
            return `<input id="${parameter.id}" name="${parameter.id}" size="30">`
        },
    },
    {
        id: "block_nsfw",
        type: ParameterType.checkbox,
        label: "Block NSFW images",
        note: "blurs out NSFW images",
        icon: "fa-land-mine-on",
        default: false,
    },
    {
        id: "sound_toggle",
        type: ParameterType.checkbox,
        label: "Enable Sound",
        note: "plays a sound on task completion",
        icon: "fa-volume-low",
        default: true,
    },
    {
        id: "process_order_toggle",
        type: ParameterType.checkbox,
        label: "Process newest jobs first",
        note: "reverse the normal processing order",
        icon: "fa-arrow-down-short-wide",
        default: false,
    },
    {
        id: "extract_lora_from_prompt",
        type: ParameterType.checkbox,
        label: "Extract LoRA tags from the prompt",
        note:
            "Automatically extract lora tags like &lt;lora:name:0.4&gt; from the prompt, and apply the correct LoRA (if present)",
        icon: "fa-code",
        default: true,
    },
    {
        id: "ui_open_browser_on_start",
        type: ParameterType.checkbox,
        label: "Open browser on startup",
        note: "starts the default browser on startup",
        icon: "fa-window-restore",
        default: true,
        saveInAppConfig: true,
    },
    {
        id: "vram_usage_level",
        type: ParameterType.select,
        label: "GPU Memory Usage",
        note:
            "Faster performance requires more GPU memory (VRAM)<br/><br/>" +
            "<b>Balanced:</b> nearly as fast as High, much lower VRAM usage<br/>" +
            "<b>High:</b> fastest, maximum GPU memory usage</br>" +
            "<b>Low:</b> slowest, recommended for GPUs with 3 to 4 GB memory",
        icon: "fa-forward",
        default: "balanced",
        saveInAppConfig: true,
        options: [
            { value: "balanced", label: "Balanced" },
            { value: "high", label: "High" },
            { value: "low", label: "Low" },
        ],
    },
    {
        id: "use_cpu",
        type: ParameterType.checkbox,
        label: "Use CPU (not GPU)",
        note: "warning: this will be *very* slow",
        icon: "fa-microchip",
        default: false,
    },
    {
        id: "auto_pick_gpus",
        type: ParameterType.checkbox,
        label: "Automatically pick the GPUs (experimental)",
        default: false,
    },
    {
        id: "use_gpus",
        type: ParameterType.select_multiple,
        label: "GPUs to use (experimental)",
        note: "to process in parallel",
        default: false,
    },
    {
        id: "auto_save_settings",
        type: ParameterType.checkbox,
        label: "Auto-Save Settings",
        note: "restores settings on browser load",
        icon: "fa-gear",
        default: true,
    },
    {
        id: "confirm_dangerous_actions",
        type: ParameterType.checkbox,
        label: "Confirm dangerous actions",
        note:
            "Actions that might lead to data loss must either be clicked with the shift key pressed, or confirmed in an 'Are you sure?' dialog",
        icon: "fa-check-double",
        default: true,
    },
    {
        id: "profileName",
        type: ParameterType.custom,
        label: "Profile Name",
        note:
            "Name of the profile for model manager settings, e.g. thumbnails for embeddings. Use this to have different settings for different users.",
        render: (parameter) => {
            return `<input id="${parameter.id}" name="${parameter.id}" value="default" size="12">`
        },
        icon: "fa-user-gear",
    },
    {
        id: "listen_to_network",
        type: ParameterType.checkbox,
        label: "Make Stable Diffusion available on your network",
        note: "Other devices on your network can access this web page. Please restart the program after changing this.",
        icon: "fa-network-wired",
        default: true,
        saveInAppConfig: true,
        table: networkParametersTable,
    },
    {
        id: "listen_port",
        type: ParameterType.custom,
        label: "Network port",
        note:
            "Port that this server listens to. The '10000' part in 'http://localhost:10000'. Please restart the program after changing this.",
        icon: "fa-anchor",
        render: (parameter) => {
            return `<input id="${parameter.id}" name="${parameter.id}" size="6" value="10000" onkeypress="preventNonNumericalInput(event)">`
        },
        saveInAppConfig: true,
        table: networkParametersTable,
    },
    {
        id: "backend_platform",
        type: ParameterType.select,
        label: "Backend platform",
        note:
            "GPU platform to use",
        saveInAppConfig: true,
        default: "auto",
        options: [
            { value: "auto", label: "Auto" },
            { value: "sycl", label: "Intel oneAPI / SYCL (local build)" },
            { value: "vulkan", label: "Vulkan (experimental)" },
        ],
    },
    {
        id: "backend_commandline_args",
        type: ParameterType.custom,
        label: "Native backend arguments",
        note:
            "Structured sdkit3 startup options. Saving with Reload enabled applies them immediately while the queue is idle.",
        icon: "fa-terminal",
        saveInAppConfig: true,
        render: renderNativeBackendArgumentsEditor,
    },
    {
        id: "reload_backend",
        type: ParameterType.checkbox,
        label: "Reload native backend when saving",
        note: "Applies native backend arguments immediately. This is allowed only while generation is idle and the queue is empty.",
        icon: "fa-rotate",
        default: false,
        saveInAppConfig: true,
    },
    {
        id: "cloudflare",
        type: ParameterType.custom,
        label: "Cloudflare tunnel",
        note: `<span id="cloudflare-off">Create a VPN tunnel to share your Easy Diffusion instance with your friends. This will
               generate a web server address on the public Internet for your Easy Diffusion instance. </span>
               <div id="cloudflare-on" class="displayNone"><div>This Easy Diffusion server is available on the Internet using the
               address:</div><div><input id="cloudflare-address" value="" readonly><button id="copy-cloudflare-address">Copy</button></div></div>
               <b>Anyone knowing this address can access your server.</b> The address of your server will change each time
               you share a session.<br>
               Uses <a href="https://try.cloudflare.com/" target="_blank">Cloudflare services</a>.`,
        icon: ["fa-brands", "fa-cloudflare"],
        render: () => '<button id="toggle-cloudflare-tunnel" class="primaryButton">Start</button>',
        table: networkParametersTable,
    },
    {
        id: "nvidia_tensorrt",
        type: ParameterType.custom,
        label: "NVIDIA TensorRT",
        note: `Faster image generation by converting your Stable Diffusion models to the NVIDIA TensorRT format. You can choose the
               models to convert. Download size: approximately 2 GB.<br/><br/>
               <b>Early access version:</b> support for LoRA is still under development.
               <div id="trt-build-config" class="displayNone">
                    <h3>Build Config:</h3>
                    Batch size range:
                    <label>Min:</label> <input id="trt-build-min-batch" type="number" min="1" value="1" style="width: 40pt" />
                    <label>Max:</label> <input id="trt-build-max-batch" type="number" min="1" value="1" style="width: 40pt" /><br/><br/>
                    <b>Build for resolutions</b>:<br/>
                    <input id="trt-build-res-512" type="checkbox" value="1" /> 512x512 to 768x768<br/>
                    <input id="trt-build-res-768" type="checkbox" value="1" checked /> 768x768 to 1024x1024<br/>
                    <input id="trt-build-res-1024" type="checkbox" value="1" /> 1024x1024 to 1280x1280<br/>
                    <input id="trt-build-res-1280" type="checkbox" value="1" /> 1280x1280 to 1536x1536<br/>
                    <input id="trt-build-res-1536" type="checkbox" value="1" /> 1536x1536 to 1792x1792<br/>
               </div>`,
        icon: "fa-angles-up",
        render: () => '<button id="toggle-tensorrt-install" class="primaryButton">Install</button>',
        table: installExtrasTable,
    },
]

function getParameterSettingsEntry(id) {
    let parameter = PARAMETERS.filter((p) => p.id === id)
    if (parameter.length === 0) {
        return
    }
    return parameter[0].settingsEntry
}

function sliderUpdate(event) {
    if (event.srcElement.id.endsWith("-input")) {
        let slider = document.getElementById(event.srcElement.id.slice(0, -6))
        slider.value = event.srcElement.value
        slider.dispatchEvent(new Event("change"))
    } else {
        let field = document.getElementById(event.srcElement.id + "-input")
        field.value = event.srcElement.value
        field.dispatchEvent(new Event("change"))
    }
}

/**
 * @param {Parameter} parameter
 * @returns {string | HTMLElement}
 */
function getParameterElement(parameter) {
    switch (parameter.type) {
        case ParameterType.checkbox:
            var is_checked = parameter.default ? " checked" : ""
            return `<input id="${parameter.id}" name="${parameter.id}"${is_checked} type="checkbox">`
        case ParameterType.select:
        case ParameterType.select_multiple:
            var options = (parameter.options || [])
                .map((option) => `<option value="${option.value}">${option.label}</option>`)
                .join("")
            var multiple = parameter.type == ParameterType.select_multiple ? "multiple" : ""
            return `<select id="${parameter.id}" name="${parameter.id}" ${multiple}>${options}</select>`
        case ParameterType.slider:
            return `<input id="${parameter.id}" name="${parameter.id}" class="editor-slider" type="range" value="${parameter.default}" min="${parameter.slider_min}" max="${parameter.slider_max}" oninput="sliderUpdate(event)"> <input id="${parameter.id}-input" name="${parameter.id}-input" size="4" value="${parameter.default}" pattern="^[0-9\.]+$" onkeypress="preventNonNumericalInput(event)" oninput="sliderUpdate(event)">&nbsp;${parameter.slider_unit}`
        case ParameterType.custom:
            return parameter.render(parameter)
        default:
            console.error(`Invalid type ${parameter.type} for parameter ${parameter.id}`)
            return "ERROR: Invalid Type"
    }
}

/**
 * fill in the system settings popup table
 * @param {Array<Parameter> | undefined} parameters
 * */
function initParameters(parameters) {
    parameters.forEach((parameter) => {
        const element = getParameterElement(parameter)
        const elementWrapper = createElement("div")
        if (element instanceof Node) {
            elementWrapper.appendChild(element)
        } else {
            elementWrapper.innerHTML = element
        }

        const note = typeof parameter.note === "function" ? parameter.note(parameter) : parameter.note
        const noteElements = []
        if (note) {
            const noteElement = createElement("small")
            if (note instanceof Node) {
                noteElement.appendChild(note)
            } else {
                noteElement.innerHTML = note || ""
            }
            noteElements.push(noteElement)
        }

        if (typeof parameter.icon == "string") {
            parameter.icon = [parameter.icon]
        }
        const icon = parameter.icon ? [createElement("i", undefined, ["fa", ...parameter.icon])] : []

        const label = typeof parameter.label === "function" ? parameter.label(parameter) : parameter.label
        const labelElement = createElement("label", { for: parameter.id })
        if (label instanceof Node) {
            labelElement.appendChild(label)
        } else {
            labelElement.innerHTML = label
        }

        const newrow = createElement(
            "div",
            { "data-setting-id": parameter.id, "data-save-in-app-config": parameter.saveInAppConfig },
            undefined,
            [
                createElement("div", undefined, undefined, icon),
                createElement("div", undefined, undefined, [labelElement, ...noteElements]),
                elementWrapper,
            ]
        )

        let p = parametersTable
        if (parameter.table) {
            p = parameter.table
        }
        p.appendChild(newrow)

        parameter.settingsEntry = newrow
    })
}

initParameters(PARAMETERS)

function splitNativeBackendArguments(value) {
    const argumentsList = []
    let token = ""
    let quote = ""
    let escaped = false
    let tokenStarted = false
    for (const character of String(value || "")) {
        if (escaped) {
            token += character
            escaped = false
            tokenStarted = true
        } else if (character === "\\" && quote !== "'") {
            escaped = true
            tokenStarted = true
        } else if (quote) {
            if (character === quote) quote = ""
            else token += character
        } else if (character === "'" || character === '"') {
            quote = character
            tokenStarted = true
        } else if (/\s/.test(character)) {
            if (tokenStarted) {
                argumentsList.push(token)
                token = ""
                tokenStarted = false
            }
        } else {
            token += character
            tokenStarted = true
        }
    }
    if (escaped) token += "\\"
    if (tokenStarted) argumentsList.push(token)
    return argumentsList
}

function quoteNativeBackendArgument(value) {
    const text = String(value)
    if (/^[a-zA-Z0-9_@%+=:,./-]+$/.test(text)) return text
    return `'${text.replaceAll("'", `'"'"'`)}'`
}

function validateNativeBackendArgumentEditor() {
    const tiles = document.getElementById(nativeBackendArgumentId("--vae-tiles"))
    const overlap = document.getElementById(nativeBackendArgumentId("--vae-tiled-overlap"))
    if (!tiles || !overlap) return true
    const tileValue = tiles.value === "" ? null : Number(tiles.value)
    const overlapValue = overlap.value === "" ? null : Number(overlap.value)
    const invalidOverlap = tileValue !== null && overlapValue !== null && overlapValue > tileValue / 2
    overlap.setCustomValidity(invalidOverlap ? "VAE tile overlap cannot exceed half the latent tile size." : "")
    return !invalidOverlap && tiles.checkValidity() && overlap.checkValidity()
}

function initNativeBackendArgumentEditor() {
    const editor = document.getElementById("native-backend-arguments-editor")
    const hidden = document.getElementById("backend_commandline_args")
    if (!editor || !hidden) return

    const additional = document.getElementById("native-backend-additional-arguments")
    const preview = document.getElementById("native-backend-arguments-preview")
    const logLevel = document.getElementById("native-backend-arg-log_level")
    const mmapMode = document.getElementById("native-backend-arg-mmap_mode")
    const booleanOptions = new Map(NATIVE_BACKEND_BOOLEAN_OPTIONS.map((option) => [option.flag, option]))
    const valueOptions = new Map(NATIVE_BACKEND_VALUE_OPTIONS.map((option) => [option.flag, option]))
    let applyingSavedArguments = false

    function updatePreview() {
        preview.textContent = hidden.value || "No custom native backend arguments."
    }

    function serializeEditor() {
        if (applyingSavedArguments) return
        validateNativeBackendArgumentEditor()
        const tokens = []
        if (logLevel.value) tokens.push("--log-level", logLevel.value)
        for (const option of NATIVE_BACKEND_BOOLEAN_OPTIONS) {
            if (document.getElementById(nativeBackendArgumentId(option.flag)).checked) tokens.push(option.flag)
        }
        if (mmapMode.value) tokens.push(mmapMode.value)
        for (const option of NATIVE_BACKEND_VALUE_OPTIONS) {
            const value = document.getElementById(nativeBackendArgumentId(option.flag)).value.trim()
            if (value) tokens.push(option.flag, value)
        }
        tokens.push(...splitNativeBackendArguments(additional.value))
        hidden.value = tokens.map(quoteNativeBackendArgument).join(" ")
        updatePreview()
    }

    function applySavedArguments() {
        applyingSavedArguments = true
        for (const option of NATIVE_BACKEND_BOOLEAN_OPTIONS) {
            document.getElementById(nativeBackendArgumentId(option.flag)).checked = false
        }
        for (const option of NATIVE_BACKEND_VALUE_OPTIONS) {
            document.getElementById(nativeBackendArgumentId(option.flag)).value = ""
        }
        logLevel.value = ""
        mmapMode.value = ""

        const extras = []
        const tokens = splitNativeBackendArguments(hidden.value)
        for (let index = 0; index < tokens.length; index += 1) {
            const token = tokens[index]
            const equalIndex = token.indexOf("=")
            const flag = equalIndex > 0 ? token.slice(0, equalIndex) : token
            const inlineValue = equalIndex > 0 ? token.slice(equalIndex + 1) : null
            if (booleanOptions.has(flag) && inlineValue === null) {
                document.getElementById(nativeBackendArgumentId(flag)).checked = true
            } else if ((flag === "--mmap" || flag === "--no-mmap") && inlineValue === null) {
                mmapMode.value = flag
            } else if (flag === "--log-level" && (inlineValue !== null || index + 1 < tokens.length)) {
                logLevel.value = inlineValue !== null ? inlineValue : tokens[++index]
            } else if (valueOptions.has(flag) && (inlineValue !== null || index + 1 < tokens.length)) {
                document.getElementById(nativeBackendArgumentId(flag)).value = inlineValue !== null ? inlineValue : tokens[++index]
            } else {
                extras.push(token)
            }
        }
        additional.value = extras.map(quoteNativeBackendArgument).join(" ")
        applyingSavedArguments = false
        validateNativeBackendArgumentEditor()
        updatePreview()
    }

    editor.querySelectorAll("input:not([type=hidden]), select, textarea").forEach((control) => {
        control.addEventListener("change", serializeEditor)
        control.addEventListener("input", serializeEditor)
    })
    hidden.addEventListener("change", applySavedArguments)
    applySavedArguments()
}

initNativeBackendArgumentEditor()

// listen to parameters from plugins
PARAMETERS.addEventListener("push", (...items) => {
    initParameters(items)

    if (items.find((item) => item.saveInAppConfig)) {
        console.log(
            "Reloading app config for new parameters",
            items.map((p) => p.id)
        )
        getAppConfig()
    }
})

let vramUsageLevelField = document.querySelector("#vram_usage_level")
let useCPUField = document.querySelector("#use_cpu")
let autoPickGPUsField = document.querySelector("#auto_pick_gpus")
let useGPUsField = document.querySelector("#use_gpus")
let saveToDiskField = document.querySelector("#save_to_disk")
let diskPathField = document.querySelector("#diskPath")
let metadataOutputFormatField = document.querySelector("#metadata_output_format")
let listenToNetworkField = document.querySelector("#listen_to_network")
let listenPortField = document.querySelector("#listen_port")
let uiOpenBrowserOnStartField = document.querySelector("#ui_open_browser_on_start")
let confirmDangerousActionsField = document.querySelector("#confirm_dangerous_actions")
let testDiffusers = document.querySelector("#use_v3_engine")
let backendPlatformField = document.querySelector("#backend_platform")
let profileNameField = document.querySelector("#profileName")
let modelsDirField = document.querySelector("#models_dir")

let saveSettingsBtn = document.querySelector("#save-system-settings-btn")

async function changeAppConfig(configDelta) {
    try {
        let res = await fetch("/app_config", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(configDelta),
        })
        const response = res
        res = await response.json()
        if (!response.ok) {
            throw new Error(res.detail || `HTTP ${response.status}`)
        }

        console.log("set config status response", res)
        return res
    } catch (e) {
        console.log("set config status error", e)
        showToast(`Settings were not saved: ${e.message}`, 7000, true)
        throw e
    }
}

function getDefaultDisplay(element) {
    const tag = element.tagName.toLowerCase();
    const defaultDisplays = {
        div: 'block',
        span: 'inline',
        p: 'block',
        tr: 'table-row',
        table: 'table',
        li: 'list-item',
        ul: 'block',
        ol: 'block',
        button: 'inline',
        // Add more if needed
    };
    return defaultDisplays[tag] || 'block'; // Default to 'block' if not listed
}

async function getAppConfig() {
    try {
        let res = await fetch("/get/app_config")
        const config = await res.json()

        applySettingsFromConfig(config)

        // custom overrides
        if (config.ui && config.ui.open_browser_on_start === false) {
            uiOpenBrowserOnStartField.checked = false
        }
        if (config.net && config.net.listen_to_network === false) {
            listenToNetworkField.checked = false
        }
        if (config.net && config.net.listen_port !== undefined) {
            listenPortField.value = config.net.listen_port
        }
        modelsDirField.value = config.models_dir

        testDiffusers.checked = true
        document.querySelector("#test_diffusers").checked = testDiffusers.checked // don't break plugins
        document.querySelector("#use_v3_engine").checked = testDiffusers.checked // don't break plugins

        if (config.backend_config?.platform) {
            backendPlatformField.value = config.backend_config.platform
        }

        document.body.classList.add("diffusers-enabled-on-startup")
        document.body.classList.remove("diffusers-disabled-on-startup")
        IMAGE_STEP_SIZE = 8

        customWidthField.step = IMAGE_STEP_SIZE
        customHeightField.step = IMAGE_STEP_SIZE

        const currentBackendKey = "backend_sdkit3"

        document.querySelectorAll('.gated-feature').forEach((element) => {
            const featureKeys = element.getAttribute('data-feature-keys').split(' ')
            const isSdkit3ControlNetPreprocessor = currentBackendKey === 'backend_sdkit3'
                && element.closest('#control_image_filter')
                && featureKeys.includes('backend_webui')

            if (featureKeys.includes(currentBackendKey) || isSdkit3ControlNetPreprocessor) {
                element.style.display = getDefaultDisplay(element)
            } else {
                element.style.display = 'none'
            }
        });

        if (config.force_save_metadata) {
            metadataOutputFormatField.value = config.force_save_metadata
        }

        console.log("get config status response", config)

        return config
    } catch (e) {
        console.log("get config status error", e)

        return {}
    }
}

function applySettingsFromConfig(config) {
    Array.from(parametersTable.children).forEach((parameterRow) => {
        if (parameterRow.dataset.settingId in config && parameterRow.dataset.saveInAppConfig === "true") {
            const configValue = config[parameterRow.dataset.settingId]
            const parameterElement =
                document.getElementById(parameterRow.dataset.settingId) ||
                parameterRow.querySelector("input") ||
                parameterRow.querySelector("select")

            switch (parameterElement?.tagName) {
                case "INPUT":
                    if (parameterElement.type === "checkbox") {
                        parameterElement.checked = configValue
                    } else {
                        parameterElement.value = configValue
                    }
                    parameterElement.dispatchEvent(new Event("change"))
                    break
                case "TEXTAREA":
                    parameterElement.value = Array.isArray(configValue) ? configValue.join(" ") : configValue
                    parameterElement.dispatchEvent(new Event("change"))
                    break
                case "SELECT":
                    if (Array.isArray(configValue)) {
                        Array.from(parameterElement.options).forEach((option) => {
                            if (configValue.includes(option.value || option.text)) {
                                option.selected = true
                            }
                        })
                    } else {
                        parameterElement.value = configValue
                    }
                    parameterElement.dispatchEvent(new Event("change"))
                    break
            }
        }
    })
}

saveToDiskField.addEventListener("change", function (e) {
    diskPathField.disabled = !this.checked
    metadataOutputFormatField.disabled = !this.checked
})

function getCurrentRenderDeviceSelection() {
    let selectedGPUs = $("#use_gpus").val()

    if (useCPUField.checked && !autoPickGPUsField.checked) {
        return "cpu"
    }
    if (autoPickGPUsField.checked || selectedGPUs.length == 0) {
        return "auto"
    }

    return selectedGPUs.join(",")
}

useCPUField.addEventListener("click", function () {
    let gpuSettingEntry = getParameterSettingsEntry("use_gpus")
    let autoPickGPUSettingEntry = getParameterSettingsEntry("auto_pick_gpus")
    if (this.checked) {
        gpuSettingEntry.style.display = "none"
        autoPickGPUSettingEntry.style.display = "none"
        autoPickGPUsField.setAttribute("data-old-value", autoPickGPUsField.checked)
        autoPickGPUsField.checked = false
    } else if (useGPUsField.options.length >= MIN_GPUS_TO_SHOW_SELECTION) {
        gpuSettingEntry.style.display = ""
        autoPickGPUSettingEntry.style.display = ""
        let oldVal = autoPickGPUsField.getAttribute("data-old-value")
        if (oldVal === null || oldVal === undefined) {
            // the UI started with CPU selected by default
            autoPickGPUsField.checked = true
        } else {
            autoPickGPUsField.checked = oldVal === "true"
        }
        gpuSettingEntry.style.display = autoPickGPUsField.checked ? "none" : ""
    }
})

useGPUsField.addEventListener("click", function () {
    let selectedGPUs = $("#use_gpus").val()
    autoPickGPUsField.checked = selectedGPUs.length === 0
})

autoPickGPUsField.addEventListener("click", function () {
    if (this.checked) {
        $("#use_gpus").val([])
    }

    let gpuSettingEntry = getParameterSettingsEntry("use_gpus")
    gpuSettingEntry.style.display = this.checked ? "none" : ""
})

async function setDiskPath(defaultDiskPath, force = false) {
    var diskPath = getSetting("diskPath")
    if (force || diskPath == "" || diskPath == undefined || diskPath == "undefined") {
        setSetting("diskPath", defaultDiskPath)
    }
}

function setDeviceInfo(devices) {
    let cpu = devices.all.cpu.name
    let allGPUs = Object.keys(devices.all).filter((d) => d != "cpu")
    let activeGPUs = Object.keys(devices.active)

    function ID_TO_TEXT(d) {
        let info = devices.all[d]
        if ("mem_free" in info && "mem_total" in info && info["mem_total"] > 0) {
            return `${info.name} <small>(${d}) (${info.mem_free.toFixed(1)}Gb free / ${info.mem_total.toFixed(
                1
            )} Gb total)</small>`
        } else {
            return `${info.name} <small>(${d}) (no memory info)</small>`
        }
    }

    allGPUs = allGPUs.map(ID_TO_TEXT)
    activeGPUs = activeGPUs.map(ID_TO_TEXT)

    let systemInfoEl = document.querySelector("#system-info")
    systemInfoEl.querySelector("#system-info-cpu").innerText = cpu
    systemInfoEl.querySelector("#system-info-gpus-all").innerHTML = allGPUs.join("</br>")
    systemInfoEl.querySelector("#system-info-rendering-devices").innerHTML = activeGPUs.join("</br>")

    // tensorRT
    if (devices.active && testDiffusers.checked && devices.enable_trt === true) {
        let nvidiaGPUs = Object.keys(devices.active).filter((d) => {
            let gpuName = devices.active[d].name
            gpuName = gpuName.toLowerCase()
            return (
                gpuName.includes("nvidia") ||
                gpuName.includes("geforce") ||
                gpuName.includes("quadro") ||
                gpuName.includes("tesla")
            )
        })
        if (nvidiaGPUs.length > 0) {
            document.querySelector("#install-extras-container").classList.remove("displayNone")
        }
    }
}

function setHostInfo(hosts) {
    let port = listenPortField.value
    hosts = hosts.map((addr) => `http://${addr}:${port}/`).map((url) => `<div><a href="${url}">${url}</a></div>`)
    document.querySelector("#system-info-server-hosts").innerHTML = hosts.join("")
}

async function getSystemInfo() {
    try {
        const res = await SD.getSystemInfo()
        let devices = res["devices"]

        let allDeviceIds = Object.keys(devices["all"]).filter((d) => d !== "cpu")
        let activeDeviceIds = Object.keys(devices["active"]).filter((d) => d !== "cpu")

        if (activeDeviceIds.length === 0) {
            useCPUField.checked = true
        }

        if (allDeviceIds.length < MIN_GPUS_TO_SHOW_SELECTION || useCPUField.checked) {
            let gpuSettingEntry = getParameterSettingsEntry("use_gpus")
            gpuSettingEntry.style.display = "none"
            let autoPickGPUSettingEntry = getParameterSettingsEntry("auto_pick_gpus")
            autoPickGPUSettingEntry.style.display = "none"
        }

        if (allDeviceIds.length === 0) {
            useCPUField.checked = true
            useCPUField.disabled = true // no compatible GPUs, so make the CPU mandatory

            getParameterSettingsEntry("use_cpu").addEventListener("click", function () {
                alert(
                    "Sorry, we could not find a compatible graphics card! Easy Diffusion supports graphics cards with minimum 2 GB of RAM. " +
                    "This Linux fork supports compatible NVIDIA and AMD graphics cards.<br/><br/>" +
                    "If you have a compatible graphics card, please try updating to the latest drivers.<br/><br/>" +
                    "Only the CPU can be used for generating images, without a compatible graphics card.",
                    "No compatible graphics card found!"
                )
            })
        }

        autoPickGPUsField.checked = devices["config"] === "auto"

        useGPUsField.innerHTML = ""
        allDeviceIds.forEach((device) => {
            let deviceName = devices["all"][device]["name"]
            let deviceOption = `<option value="${device}">${deviceName} (${device})</option>`
            useGPUsField.insertAdjacentHTML("beforeend", deviceOption)
        })

        if (autoPickGPUsField.checked) {
            let gpuSettingEntry = getParameterSettingsEntry("use_gpus")
            gpuSettingEntry.style.display = "none"
        } else {
            $("#use_gpus").val(activeDeviceIds)
        }

        if (useCPUField.checked || activeDeviceIds.includes("mps")) {
            let backendPlatformEntry = getParameterSettingsEntry("backend_platform")
            backendPlatformEntry.style.display = "none"
        }

        document.dispatchEvent(new CustomEvent("system_info_update", { detail: devices }))
        setHostInfo(res["hosts"])
        let force = false
        if (res["enforce_output_dir"] !== undefined) {
            force = res["enforce_output_dir"]
            if (force == true) {
                saveToDiskField.checked = true
                metadataOutputFormatField.disabled = res["enforce_output_metadata"]
                diskPathField.disabled = true
            }
            saveToDiskField.disabled = force
        } else {
            diskPathField.disabled = !saveToDiskField.checked
            metadataOutputFormatField.disabled = !saveToDiskField.checked
        }
        setDiskPath(res["default_output_dir"], force)

    } catch (e) {
        console.log("error fetching devices", e)
    }
}

saveSettingsBtn.addEventListener("click", function () {
    validateNativeBackendArgumentEditor()
    const invalidBackendArgument = document.querySelector("#native-backend-arguments-editor :invalid")
    if (invalidBackendArgument) {
        invalidBackendArgument.reportValidity()
        return
    }
    if (listenPortField.value == "") {
        alert("The network port field must not be empty.")
        return
    }
    if (listenPortField.value < 1 || listenPortField.value > 65535) {
        alert("The network port must be a number from 1 to 65535")
        return
    }
    const updateAppConfigRequest = {
        render_devices: getCurrentRenderDeviceSelection(),
    }

    document.querySelectorAll("#system-settings [data-setting-id]").forEach((parameterRow) => {
        if (parameterRow.dataset.saveInAppConfig === "true") {
            const parameterElement =
                document.getElementById(parameterRow.dataset.settingId) ||
                parameterRow.querySelector("input") ||
                parameterRow.querySelector("select")

            switch (parameterElement?.tagName) {
                case "INPUT":
                    if (parameterElement.type === "checkbox") {
                        updateAppConfigRequest[parameterRow.dataset.settingId] = parameterElement.checked
                    } else {
                        updateAppConfigRequest[parameterRow.dataset.settingId] = parameterElement.value
                    }
                    break
                case "SELECT":
                    if (parameterElement.multiple) {
                        updateAppConfigRequest[parameterRow.dataset.settingId] = Array.from(parameterElement.options)
                            .filter((option) => option.selected)
                            .map((option) => option.value || option.text)
                    } else {
                        updateAppConfigRequest[parameterRow.dataset.settingId] = parameterElement.value
                    }
                    break
                case "TEXTAREA":
                    updateAppConfigRequest[parameterRow.dataset.settingId] = parameterElement.value
                    break
                default:
                    console.error(
                        `Setting parameter ${parameterRow.dataset.settingId} couldn't be saved to app.config - element #${parameterRow.dataset.settingId} is a <${parameterElement?.tagName} /> instead of an <input />, <select />, or <textarea />!`
                    )
                    break
            }
        }
    })

    const savePromise = changeAppConfig(updateAppConfigRequest)
        .then((result) => {
            const reloadBackendField = document.getElementById("reload_backend")
            if (reloadBackendField) reloadBackendField.checked = false
            showToast(result.backend_restarted ? "Settings saved and native backend reloaded" : "Settings saved")
        })
    saveSettingsBtn.classList.add("active")
    Promise.allSettled([savePromise, asyncDelay(300)]).then(() => saveSettingsBtn.classList.remove("active"))
})

listenToNetworkField.addEventListener(
    "change",
    debounce(() => {
        saveSettingsBtn.click()
    }, 1000)
)

listenPortField.addEventListener(
    "change",
    debounce(() => {
        saveSettingsBtn.click()
    }, 1000)
)

let copyCloudflareAddressBtn = document.querySelector("#copy-cloudflare-address")
let cloudflareAddressField = document.getElementById("cloudflare-address")

navigator.permissions.query({ name: "clipboard-write" }).then(function (result) {
    if (result.state === "granted") {
        // you can read from the clipboard
        copyCloudflareAddressBtn.addEventListener("click", (e) => {
            navigator.clipboard.writeText(cloudflareAddressField.innerHTML)
            showToast("Copied server address to clipboard")
        })
    } else {
        copyCloudflareAddressBtn.classList.add("displayNone")
    }
})

document.addEventListener("system_info_update", (e) => setDeviceInfo(e.detail))
