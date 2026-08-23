;(function () {
  "use strict"

  const HOST_NAME = window.location.hostname
  const ORIGIN = window.location.origin
  const SAME_ORIGIN_API_BASE = `${ORIGIN}/civitai-api`
  const PLUGIN_ID = "civitai-downloader"
  const SEARCH_PATH = "/civitai/search"
  const IMAGES_PATH = "/civitai/images"
  const GLOBAL_IMAGES_PATH = "/civitai/global-images"
  const ME_PATH = "/civitai/me"
  const TAGS_PATH = "/civitai/tags"
  const CREATORS_PATH = "/civitai/creators"
  const ENUMS_PATH = "/civitai/enums"
  const LOOKUP_HASH_PATH = "/civitai/lookup/hash"
  const LOOKUP_MINI_PATH = "/civitai/lookup/mini"
  const LOOKUP_HASHES_PATH = "/civitai/lookup/hashes"
  const COLLECTIONS_PATH = "/civitai/collections"
  const PERMISSIONS_PATH = "/civitai/permissions"
  const DOWNLOAD_PATHS = ["/civitai/download", "/civitai/search/download"]
  const DOWNLOAD_STATUS_PREFIX = "/civitai/download/"
  const LS_SETTINGS_KEY = "civitaiSettings"
  const LS_COLLAPSE_KEY = "civitaiPanelCollapsed"
  const LS_RESOLVED_BASE_KEY = "civitaiResolvedBaseUrl"

  // CivitAI enforces different sort enums for models vs. images; keep them separate.
  const SORT_OPTIONS_TYPE = {
    models: ["Highest Rated", "Most Downloaded", "Newest"],
    images: ["Newest", "Most Reactions", "Most Comments"],
    "global-images": ["Relevance"],
  }
  const MODEL_TYPE_OPTIONS = [
    "",
    "Checkpoint",
    "TextualInversion",
    "Hypernetwork",
    "AestheticGradient",
    "LORA",
    "LoCon",
    "DoRA",
    "Controlnet",
    "Upscaler",
    "MotionModule",
    "VAE",
    "TextEncoder",
    "UNet",
    "CLIPVision",
    "Poses",
    "Wildcards",
    "Workflows",
    "ComfyWorkflows",
    "Detection",
    "VisionLanguage",
    "CLIP",
    "LLM",
    "Other",
  ]
  const BASE_MODEL_OPTIONS = [
    "",
    "Anima",
    "AuraFlow",
    "Chroma",
    "CogVideoX",
    "Ernie",
    "Flux.1 S",
    "Flux.1 D",
    "Flux.1 Krea",
    "Flux.1 Kontext",
    "Flux.2 D",
    "Flux.2 Klein 9B",
    "Flux.2 Klein 9B-base",
    "Flux.2 Klein 4B",
    "Flux.2 Klein 4B-base",
    "Flux 3 Video",
    "Grok",
    "HappyHorse",
    "HiDream",
    "HiDream-O1",
    "Hunyuan 1",
    "Hunyuan Video",
    "Ideogram 4.0",
    "Boogu",
    "Illustrious",
    "Imagen4",
    "Kolors",
    "Krea 2",
    "LTXV",
    "LTXV2",
    "LTXV 2.3",
    "LTXV 2.5",
    "Lens",
    "Lumina",
    "MageFlow",
    "MAI",
    "Mochi",
    "Nano Banana",
    "NoobAI",
    "ODOR",
    "OpenAI",
    "Upscaler",
    "Other",
    "PixArt a",
    "PixArt E",
    "Playground v2",
    "Pony",
    "Pony V7",
    "Qwen",
    "Qwen 2",
    "Qwen 3",
    "Stable Cascade",
    "SD 1.4",
    "SD 1.5",
    "SD 1.5 LCM",
    "SD 1.5 Hyper",
    "SD 2.0",
    "SD 2.0 768",
    "SD 2.1",
    "SD 2.1 768",
    "SD 2.1 Unclip",
    "SD 3",
    "SD 3.5",
    "SD 3.5 Large",
    "SD 3.5 Large Turbo",
    "SD 3.5 Medium",
    "SDXL 0.9",
    "SDXL 1.0",
    "SDXL 1.0 LCM",
    "SDXL Lightning",
    "SDXL Hyper",
    "SDXL Turbo",
    "SDXL Distilled",
    "Reve",
    "Seedream",
    "SVD",
    "SVD XT",
    "Sora 2",
    "Veo 3",
    "Wan Video",
    "Wan Video 1.3B t2v",
    "Wan Video 14B t2v",
    "Wan Video 14B i2v 480p",
    "Wan Video 14B i2v 720p",
    "Wan Video 2.2 TI2V-5B",
    "Wan Video 2.2 I2V-A14B",
    "Wan Video 2.2 T2V-A14B",
    "Wan Video 2.5 T2V",
    "Wan Video 2.5 I2V",
    "Wan Image 2.7",
    "Wan Video 2.7",
    "ZImageTurbo",
    "ZImageBase",
    "Vidu Q1",
    "MiniMax H3",
    "Kling",
    "Seedance",
    "ACE Audio",
    "PolyGen",
    "Tripo",
    "Hunyuan3D",
  ]

  function normalizeBaseUrl(raw) {
    const v = (raw || "").trim()
    if (!v) return ""
    return v.replace(/\/+$/, "")
  }

  function airBaseModelKey(baseModel) {
    const normalized = String(baseModel || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")

    if (/\b(sdxl|pony|illustrious|noobai)\b/.test(normalized)) return "sdxl"
    if (/^sd 1\b/.test(normalized) || /^sd 2\b/.test(normalized)) return "sd"
    if (/^sd 3 5\b/.test(normalized)) return "sd35"
    if (/^sd 3\b/.test(normalized)) return "sd3"
    if (/^flux 1\b/.test(normalized)) return "flux1"
    if (/^flux 2\b/.test(normalized)) return "flux2"

    return normalized.replace(/\s+/g, "") || "other"
  }

  function isLoopbackHost(host) {
    const h = (host || "").trim().toLowerCase()
    return h === "127.0.0.1" || h === "localhost" || h === "::1"
  }

  function withTimeout(ms = 2500) {
    const ctl = new AbortController()
    const timer = setTimeout(() => ctl.abort(), ms)
    return {
      signal: ctl.signal,
      done: () => clearTimeout(timer),
    }
  }

  async function discoverBaseCandidates() {
    const candidates = []
    const add = (url) => {
      const v = normalizeBaseUrl(url)
      if (v && !candidates.includes(v)) candidates.push(v)
    }

    add(SAME_ORIGIN_API_BASE)
    if (window.location.protocol !== "https:") {
      add(`http://${HOST_NAME}:9004`)
      if (isLoopbackHost(HOST_NAME)) {
        add("http://127.0.0.1:9004")
        add("http://localhost:9004")
      }
    }

    try {
      const t = withTimeout(2000)
      const res = await fetch("/get/system_info", { signal: t.signal })
      t.done()
      if (res.ok) {
        const info = await res.json()
        const hosts = Array.isArray(info && info.hosts) ? info.hosts : []
        hosts.forEach((host) => {
          if (!host || host === "0.0.0.0") return
          if (window.location.protocol !== "https:") {
            add(`http://${host}:9004`)
          }
        })
      }
    } catch (_) {}

    return candidates
  }

  async function resolveDynamicBaseUrl(customBaseUrl = "") {
    const custom = normalizeBaseUrl(customBaseUrl)
    if (custom) return custom

    let cached = normalizeBaseUrl(localStorage.getItem(LS_RESOLVED_BASE_KEY) || "")
    if (window.location.protocol === "https:" && cached.startsWith("http:")) {
      localStorage.removeItem(LS_RESOLVED_BASE_KEY)
      cached = ""
    }
    const probe = async (base) => {
      try {
        const t = withTimeout(2000)
        const res = await fetch(`${base}/health`, { signal: t.signal })
        t.done()
        return res.ok
      } catch (_) {
        return false
      }
    }

    if (cached && await probe(cached)) {
      return cached
    }

    const candidates = await discoverBaseCandidates()
    for (const base of candidates) {
      if (await probe(base)) {
        localStorage.setItem(LS_RESOLVED_BASE_KEY, base)
        return base
      }
    }

    return ""
  }

  function endpointCandidates(pathWithQuery, customBaseUrl = "") {
    const custom = normalizeBaseUrl(customBaseUrl)
    const candidates = [
      custom ? `${custom}${pathWithQuery}` : "",
      `${SAME_ORIGIN_API_BASE}${pathWithQuery}`,
      `${ORIGIN}${pathWithQuery}`,
      window.location.protocol !== "https:" ? `http://${HOST_NAME}:9004${pathWithQuery}` : "",
    ].filter(Boolean)
    return [...new Set(candidates)]
  }

  async function fetchWithFallback(pathWithQuery, options = {}, customBaseUrl = "") {
    const candidates = endpointCandidates(pathWithQuery, customBaseUrl)
    let lastError = null

    for (const url of candidates) {
      try {
        const res = await fetch(url, options)
        // If one route isn't available in this setup, try the next candidate.
        if (res.status === 404 || res.status === 405) {
          continue
        }
        return res
      } catch (err) {
        lastError = err
      }
    }

    if (lastError) throw lastError
    throw new Error("CivitAI endpoint unavailable")
  }

  async function fetchWithPathFallback(paths, options = {}, customBaseUrl = "") {
    let lastError = null

    for (const path of paths) {
      try {
        const res = await fetchWithFallback(path, options, customBaseUrl)
        if (res.status === 404 || res.status === 405) {
          continue
        }
        return res
      } catch (err) {
        lastError = err
      }
    }

    if (lastError) throw lastError
    throw new Error("CivitAI endpoint unavailable")
  }

  async function safeJson(res) {
    const ct = (res.headers.get("content-type") || "").toLowerCase()
    if (ct.includes("application/json")) return res.json()
    const text = await res.text().catch(() => "")
    throw new Error(`HTTP ${res.status} (non-JSON): ${text.slice(0, 200)}`)
  }

  function formatSize(bytes) {
    const value = Number(bytes)
    if (!Number.isFinite(value) || value < 0) return "0 B"
    if (value < 1024) return `${value.toFixed(0)} B`
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
    if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`
    return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`
  }

  async function pollDownloadStatus(
    downloadId,
    statusEl,
    requestHeaders,
    customBaseUrl = "",
    suffixProvider = () => ""
  ) {
    const path = `${DOWNLOAD_STATUS_PREFIX}${encodeURIComponent(downloadId)}`
    while (true) {
      const res = await fetchWithFallback(path, { headers: requestHeaders }, customBaseUrl)
      const json = await safeJson(res)
      if (!json.ok) throw new Error(json.error || "Download status failed")

      const downloaded = Number(json.downloadedBytes || 0)
      const total = Number(json.totalBytes || 0)
      const percent =
        typeof json.percent === "number"
          ? json.percent
          : total > 0
            ? (downloaded / total) * 100
            : null

      if (json.status === "queued") {
        statusEl.textContent = `Queued...${suffixProvider()}`
      } else if (json.status === "downloading") {
        if (percent != null && Number.isFinite(percent)) {
          const totalText = total > 0 ? ` / ${formatSize(total)}` : ""
          statusEl.textContent = `Downloading ${percent.toFixed(1)}% (${formatSize(downloaded)}${totalText})${suffixProvider()}`
        } else {
          statusEl.textContent = `Downloading ${formatSize(downloaded)}${suffixProvider()}`
        }
      } else if (json.status === "completed") {
        statusEl.textContent = `Downloaded to ${json.path} (${formatSize(json.size)})${suffixProvider()}`
        return
      } else if (json.status === "failed") {
        throw new Error(json.error || "Download failed")
      }

      await new Promise((resolve) => setTimeout(resolve, 800))
    }
  }

  function ready() {
    const btn = document.querySelector("#makeImage")
    if (!btn || document.getElementById(`${PLUGIN_ID}-panel`)) return

    const panel = document.createElement("div")
    panel.id = `${PLUGIN_ID}-panel`
    panel.className = "panel-box"

    // Collapsible style copied from your other script: header click + ➕/➖ + display:none
    panel.innerHTML = `
      <div id="${PLUGIN_ID}-header" style="cursor:pointer; user-select:none; display:flex; align-items:center;">
        <span id="${PLUGIN_ID}-handle" class="collapsible-handle">➕</span>
        <h4 style="margin:0;">CivitAI Browser</h4>
      </div>

      <div id="${PLUGIN_ID}-body" class="collapsible-content" style="display:none; margin-top:8px;">
        <div style="display:flex; gap:8px; align-items:center; margin-bottom:8px;">
          <label style="display:flex; align-items:center; gap:6px; font-size:0.95em;">
            <input type="radio" name="${PLUGIN_ID}-mode" value="models" checked />
            Models
          </label>
          <label style="display:flex; align-items:center; gap:6px; font-size:0.95em;">
            <input type="radio" name="${PLUGIN_ID}-mode" value="images" />
            Images
          </label>
          <label style="display:flex; align-items:center; gap:6px; font-size:0.95em;">
            <input type="radio" name="${PLUGIN_ID}-mode" value="global-images" />
            Global Images
          </label>

          <span style="flex:1;"></span>

          <label style="display:flex; align-items:center; gap:6px; font-size:0.9em;">
            <input type="checkbox" id="${PLUGIN_ID}-nsfw" /> Include NSFW
          </label>
        </div>

        <div style="display:flex; gap:6px; margin-bottom:6px;">
          <input id="${PLUGIN_ID}-query" type="text" placeholder="Search term, model/version API URL, SHA256, image-search URL, or URN" style="flex:1;" />
          <button class="secondaryButton" id="${PLUGIN_ID}-search">Search</button>
        </div>

        <div id="${PLUGIN_ID}-sortfilters" style="display:flex; gap:6px; margin-bottom:6px;">
          <select id="${PLUGIN_ID}-sort" style="flex:1; background:var(--background-color3); color:var(--text-color);">
            <!-- options populated dynamically per mode -->
          </select>
          <select id="${PLUGIN_ID}-period" style="flex:1; background:var(--background-color3); color:var(--text-color);">
            <option value="AllTime">All Time</option>
            <option value="Year">Year</option>
            <option value="Month">Month</option>
            <option value="Week">Week</option>
            <option value="Day">Day</option>
          </select>
        </div>

        <div id="${PLUGIN_ID}-sorthelp" style="font-size:0.8em; opacity:0.7; margin-bottom:6px;">
          Sort options are limited by the CivitAI API for models and images.
        </div>

        <!-- Images-only filters (hidden unless Images mode) -->
        <div id="${PLUGIN_ID}-imgfilters" style="display:none; gap:6px; margin-bottom:6px;">
          <input id="${PLUGIN_ID}-postid" type="number" placeholder="postId (optional)" style="flex:1;" />
          <input id="${PLUGIN_ID}-username" type="text" list="${PLUGIN_ID}-creator-options" placeholder="username (optional)" style="flex:1;" />
        </div>
        <div id="${PLUGIN_ID}-imgsortfilters" style="display:none; gap:6px; margin-bottom:6px;">
          <select id="${PLUGIN_ID}-modelsort" style="flex:1; background:var(--background-color3); color:var(--text-color);"></select>
        </div>

        <!-- Global image-search filters -->
        <div id="${PLUGIN_ID}-globalfilters" style="display:none; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:6px; margin-bottom:6px;">
          <input id="${PLUGIN_ID}-globaltag" type="text" list="${PLUGIN_ID}-tag-options" placeholder="Tag, e.g. accessories" />
          <datalist id="${PLUGIN_ID}-tag-options"></datalist>
          <input id="${PLUGIN_ID}-globaltechnique" type="text" placeholder="Technique, e.g. txt2img" />
          <input id="${PLUGIN_ID}-globaltool" type="text" placeholder="Tool, e.g. A1111" />
          <select id="${PLUGIN_ID}-globalaspect" style="background:var(--background-color3); color:var(--text-color);">
            <option value="">All Aspect Ratios</option>
            <option value="Portrait">Portrait</option>
            <option value="Square">Square</option>
            <option value="Landscape">Landscape</option>
          </select>
          <select id="${PLUGIN_ID}-globalbase" style="background:var(--background-color3); color:var(--text-color);"></select>
          <select id="${PLUGIN_ID}-globaltype" style="background:var(--background-color3); color:var(--text-color);">
            <option value="image" selected>Image</option>
            <option value="">All Media Types</option>
            <option value="video">Video</option>
          </select>
          <input id="${PLUGIN_ID}-globalusername" type="text" list="${PLUGIN_ID}-creator-options" placeholder="Username, e.g. lzfr" />
          <input id="${PLUGIN_ID}-globalcreatedfrom" type="datetime-local" title="Created from" />
          <input id="${PLUGIN_ID}-globalcreatedto" type="datetime-local" title="Created through" />
        </div>
        <datalist id="${PLUGIN_ID}-creator-options"></datalist>

        <!-- Models-only filters -->
        <div id="${PLUGIN_ID}-modelfilters" style="display:flex; gap:6px; margin-bottom:6px;">
          <select id="${PLUGIN_ID}-modeltype" style="flex:1; background:var(--background-color3); color:var(--text-color);"></select>
          <select id="${PLUGIN_ID}-basemodel" style="flex:1; background:var(--background-color3); color:var(--text-color);"></select>
        </div>

        <div style="display:flex; gap:6px; align-items:center; margin-bottom:6px;">
          <input id="${PLUGIN_ID}-apikey" type="password" placeholder="API token (optional)" style="flex:1;" />
          <button class="secondaryButton" id="${PLUGIN_ID}-verify-key">Verify Token</button>
          <button class="secondaryButton" id="${PLUGIN_ID}-clear">Clear</button>
        </div>

        <div style="display:flex; gap:6px; align-items:center; margin-bottom:6px;">
          <input id="${PLUGIN_ID}-searchkey" type="password" placeholder="Global image search key (required for Global Images)" style="flex:1;" />
        </div>

        <div style="display:flex; gap:6px; align-items:center; margin-bottom:6px;">
          <input id="${PLUGIN_ID}-serverurl" type="text" placeholder="CivitAI Server URL (optional), e.g. http://192.168.1.10:9004" style="flex:1;" />
        </div>

        <details id="${PLUGIN_ID}-advanced" style="margin-bottom:6px; border:1px solid var(--background-color3); border-radius:6px; padding:6px;">
          <summary style="cursor:pointer; user-select:none;">Advanced CivitAI lookups</summary>
          <div style="display:grid; grid-template-columns:minmax(130px,0.7fr) minmax(180px,1.5fr) minmax(120px,0.7fr) auto; gap:6px; margin-top:6px;">
            <select id="${PLUGIN_ID}-advanced-type" style="background:var(--background-color3); color:var(--text-color);">
              <option value="hash">SHA256 details</option>
              <option value="mini">Mini version</option>
              <option value="hashes">Bulk hash details</option>
              <option value="hash-ids">Bulk hash IDs</option>
              <option value="collections">Collections search</option>
              <option value="collection">Collection ID</option>
              <option value="permissions">Permission check</option>
            </select>
            <input id="${PLUGIN_ID}-advanced-value" type="text" placeholder="SHA256" />
            <input id="${PLUGIN_ID}-advanced-userid" type="number" placeholder="userId (optional)" style="display:none;" />
            <button class="secondaryButton" id="${PLUGIN_ID}-advanced-run">Lookup</button>
          </div>
          <pre id="${PLUGIN_ID}-advanced-result" style="display:none; white-space:pre-wrap; overflow-wrap:anywhere; max-height:220px; overflow:auto; margin:6px 0 0; padding:6px; background:var(--background-color3); border-radius:6px;"></pre>
        </details>

        <div id="${PLUGIN_ID}-status" style="font-size:0.9em; min-height:18px; margin-bottom:6px;"></div>
        <div id="${PLUGIN_ID}-results" style="max-height:340px; overflow:auto; display:flex; flex-direction:column; gap:8px;"></div>

        <div style="display:flex; justify-content:space-between; margin-top:6px;">
          <button class="secondaryButton" id="${PLUGIN_ID}-prev" disabled>Prev</button>
          <span id="${PLUGIN_ID}-page" style="align-self:center;">Page 1</span>
          <button class="secondaryButton" id="${PLUGIN_ID}-next" disabled>Next</button>
        </div>
      </div>
    `

    // ✅ PLACE BELOW the line separator:
    const sep =
      document.querySelector("#editor-inputs span.line-separator") ||
      document.querySelector("span.line-separator")

    if (sep && sep.parentNode) {
      sep.insertAdjacentElement("afterend", panel)
    } else {
      // fallback (keeps working if the separator isn't found)
      btn.insertAdjacentElement("afterend", panel)
    }


    const queryEl = panel.querySelector(`#${PLUGIN_ID}-query`)
    const searchBtn = panel.querySelector(`#${PLUGIN_ID}-search`)
    const resultsEl = panel.querySelector(`#${PLUGIN_ID}-results`)
    const statusEl = panel.querySelector(`#${PLUGIN_ID}-status`)
    const prevBtn = panel.querySelector(`#${PLUGIN_ID}-prev`)
    const nextBtn = panel.querySelector(`#${PLUGIN_ID}-next`)
    const pageLabel = panel.querySelector(`#${PLUGIN_ID}-page`)
    const apiKeyEl = panel.querySelector(`#${PLUGIN_ID}-apikey`)
    const verifyKeyBtn = panel.querySelector(`#${PLUGIN_ID}-verify-key`)
    const searchKeyEl = panel.querySelector(`#${PLUGIN_ID}-searchkey`)
    const serverUrlEl = panel.querySelector(`#${PLUGIN_ID}-serverurl`)
    const clearBtn = panel.querySelector(`#${PLUGIN_ID}-clear`)
    const advancedTypeEl = panel.querySelector(`#${PLUGIN_ID}-advanced-type`)
    const advancedValueEl = panel.querySelector(`#${PLUGIN_ID}-advanced-value`)
    const advancedUserIdEl = panel.querySelector(`#${PLUGIN_ID}-advanced-userid`)
    const advancedRunBtn = panel.querySelector(`#${PLUGIN_ID}-advanced-run`)
    const advancedResultEl = panel.querySelector(`#${PLUGIN_ID}-advanced-result`)
    const nsfwEl = panel.querySelector(`#${PLUGIN_ID}-nsfw`)
    const sortEl = panel.querySelector(`#${PLUGIN_ID}-sort`)
    const periodEl = panel.querySelector(`#${PLUGIN_ID}-period`)
    const sortFiltersRow = panel.querySelector(`#${PLUGIN_ID}-sortfilters`)
    const sortHelpEl = panel.querySelector(`#${PLUGIN_ID}-sorthelp`)
    const modeEls = panel.querySelectorAll(`input[name="${PLUGIN_ID}-mode"]`)

    const imgFiltersRow = panel.querySelector(`#${PLUGIN_ID}-imgfilters`)
    const imgSortFiltersRow = panel.querySelector(`#${PLUGIN_ID}-imgsortfilters`)
    const modelFiltersRow = panel.querySelector(`#${PLUGIN_ID}-modelfilters`)
    const modelTypeEl = panel.querySelector(`#${PLUGIN_ID}-modeltype`)
    const baseModelEl = panel.querySelector(`#${PLUGIN_ID}-basemodel`)
    const postIdEl = panel.querySelector(`#${PLUGIN_ID}-postid`)
    const usernameEl = panel.querySelector(`#${PLUGIN_ID}-username`)
    const modelSortEl = panel.querySelector(`#${PLUGIN_ID}-modelsort`)
    const globalFiltersRow = panel.querySelector(`#${PLUGIN_ID}-globalfilters`)
    const globalTagEl = panel.querySelector(`#${PLUGIN_ID}-globaltag`)
    const globalTagOptionsEl = panel.querySelector(`#${PLUGIN_ID}-tag-options`)
    const creatorOptionsEl = panel.querySelector(`#${PLUGIN_ID}-creator-options`)
    const globalTechniqueEl = panel.querySelector(`#${PLUGIN_ID}-globaltechnique`)
    const globalToolEl = panel.querySelector(`#${PLUGIN_ID}-globaltool`)
    const globalAspectEl = panel.querySelector(`#${PLUGIN_ID}-globalaspect`)
    const globalBaseEl = panel.querySelector(`#${PLUGIN_ID}-globalbase`)
    const globalTypeEl = panel.querySelector(`#${PLUGIN_ID}-globaltype`)
    const globalUsernameEl = panel.querySelector(`#${PLUGIN_ID}-globalusername`)
    const globalCreatedFromEl = panel.querySelector(`#${PLUGIN_ID}-globalcreatedfrom`)
    const globalCreatedToEl = panel.querySelector(`#${PLUGIN_ID}-globalcreatedto`)
    modelTypeEl.innerHTML = MODEL_TYPE_OPTIONS
      .map((v) => `<option value="${v}">${v || "All Model Types"}</option>`)
      .join("")
    baseModelEl.innerHTML = BASE_MODEL_OPTIONS
      .map((v) => `<option value="${v}">${v || "All Base Models"}</option>`)
      .join("")
    globalBaseEl.innerHTML = BASE_MODEL_OPTIONS
      .map((v) => `<option value="${v}">${v || "All Base Models"}</option>`)
      .join("")
    modelSortEl.innerHTML = SORT_OPTIONS_TYPE.models
      .map((v) => `<option value="${v}">${`Model Query Sort: ${v}`}</option>`)
      .join("")
    modelSortEl.value = "Newest"

    // Collapse elements (header click + handle)
    const headerEl = panel.querySelector(`#${PLUGIN_ID}-header`)
    const bodyEl = panel.querySelector(`#${PLUGIN_ID}-body`)
    const handleEl = panel.querySelector(`#${PLUGIN_ID}-handle`)

    // Paging state
    let currentPage = 1
    let lastQuery = ""
    let modelSearchQuery = ""
    let currentMode = "models" // models | images | global-images
    let selectedModel = null
    let activeDownloadId = null
    let isProcessingQueue = false
    const downloadQueue = []

    // Cursor paging (models)
    let currentCursor = null
    let nextCursor = null
    const prevCursorStack = []

    function getMode() {
      const checked = panel.querySelector(`input[name="${PLUGIN_ID}-mode"]:checked`)
      return checked ? checked.value : "models"
    }

    function refreshSortOptions(mode) {
      const allowed = SORT_OPTIONS_TYPE[mode] || SORT_OPTIONS_TYPE.models
      const current = sortEl.value
      sortEl.innerHTML = allowed.map((s) => `<option value="${s}">${s}</option>`).join("")
      sortEl.value = allowed.includes(current) ? current : allowed[0]
    }

    function applyModeUI() {
      currentMode = getMode()
      const imageMode = currentMode === "images"
      const globalImageMode = currentMode === "global-images"
      imgFiltersRow.style.display = imageMode ? "flex" : "none"
      imgSortFiltersRow.style.display = "none"
      globalFiltersRow.style.display = globalImageMode ? "grid" : "none"
      modelFiltersRow.style.display = currentMode === "models" ? "flex" : "none"
      sortFiltersRow.style.display = globalImageMode ? "none" : "flex"
      sortHelpEl.style.display = globalImageMode ? "none" : "block"
      queryEl.disabled = imageMode
      queryEl.placeholder = imageMode
        ? selectedModel
          ? `Showing images made with ${selectedModel.name}`
          : "Select View Images on a model result first"
        : globalImageMode
          ? "Search all CivitAI images or paste a civitai.red/search/images URL"
          : "Search term, model/version API URL, SHA256, or URN (blank = browse)"
      searchBtn.textContent = imageMode
        ? "Reload Images"
        : globalImageMode
          ? "Search Images"
          : "Search"
      refreshSortOptions(currentMode)
    }

    function headersForRequest() {
      const h = {}
      const k = (apiKeyEl.value || "").trim()
      if (k) h["x-civitai-key"] = k
      const searchKey = (searchKeyEl.value || "").trim()
      if (searchKey) h["x-civitai-search-key"] = searchKey
      return h
    }

    function extractCursor(token) {
      if (!token) return null
      try {
        if (typeof token === "string" && token.startsWith("http")) {
          const url = new URL(token)
          return url.searchParams.get("cursor") || null
        }
      } catch (_) {}
      if (typeof token !== "string") return null
      const cleaned = token.trim()
      return cleaned || null
    }

    function renderEmptyState(message) {
      resultsEl.innerHTML = `
        <div style="border:1px dashed var(--background-color3); border-radius:8px; padding:10px; font-size:0.9em; color:var(--text-color2);">
          ${message}
        </div>
      `
    }

    function saveSettings() {
      const settings = {
        sort: sortEl.value,
        period: periodEl.value,
        nsfw: nsfwEl.checked,
        apikey: apiKeyEl.value.trim(),
        searchKey: searchKeyEl.value.trim(),
        serverUrl: serverUrlEl.value.trim(),
        mode: getMode(),
        modelType: modelTypeEl.value,
        baseModel: baseModelEl.value,
        postId: (postIdEl.value || "").trim(),
        username: (usernameEl.value || "").trim(),
        modelSort: modelSortEl.value || "Newest",
        globalTag: globalTagEl.value.trim(),
        globalTechnique: globalTechniqueEl.value.trim(),
        globalTool: globalToolEl.value.trim(),
        globalAspect: globalAspectEl.value,
        globalBase: globalBaseEl.value,
        globalType: globalTypeEl.value,
        globalUsername: globalUsernameEl.value.trim(),
        globalCreatedFrom: globalCreatedFromEl.value,
        globalCreatedTo: globalCreatedToEl.value,
        selectedModel: selectedModel
          ? { id: selectedModel.id, name: selectedModel.name }
          : null,
      }
      localStorage.setItem(LS_SETTINGS_KEY, JSON.stringify(settings))
    }

    function loadSettings() {
      const saved = localStorage.getItem(LS_SETTINGS_KEY)
      if (saved) {
        try {
          const settings = JSON.parse(saved)
          periodEl.value = settings.period || "AllTime"
          nsfwEl.checked = !!settings.nsfw
          apiKeyEl.value = settings.apikey || ""
          searchKeyEl.value = settings.searchKey || ""
          serverUrlEl.value = settings.serverUrl || ""
          modelTypeEl.value = settings.modelType || ""
          baseModelEl.value = settings.baseModel || ""
          postIdEl.value = settings.postId || ""
          usernameEl.value = settings.username || ""
          modelSortEl.value = settings.modelSort || "Newest"
          globalTagEl.value = settings.globalTag || ""
          globalTechniqueEl.value = settings.globalTechnique || ""
          globalToolEl.value = settings.globalTool || ""
          globalAspectEl.value = settings.globalAspect || ""
          globalBaseEl.value = settings.globalBase || ""
          globalTypeEl.value = settings.globalType || "image"
          globalUsernameEl.value = settings.globalUsername || ""
          globalCreatedFromEl.value = settings.globalCreatedFrom || ""
          globalCreatedToEl.value = settings.globalCreatedTo || ""

          const savedModel = settings.selectedModel
          if (savedModel && Number.isFinite(Number(savedModel.id))) {
            selectedModel = {
              id: Number(savedModel.id),
              name: String(savedModel.name || `Model ${savedModel.id}`),
            }
          }

          const m =
            settings.mode === "images" && selectedModel
              ? "images"
              : settings.mode === "global-images"
                ? "global-images"
                : "models"
          const radio = panel.querySelector(`input[name="${PLUGIN_ID}-mode"][value="${m}"]`)
          if (radio) radio.checked = true
          currentMode = m
          refreshSortOptions(currentMode)
          const allowed = SORT_OPTIONS_TYPE[currentMode] || []
          sortEl.value = allowed.includes(settings.sort) ? settings.sort : allowed[0]
        } catch (e) {
          console.warn("Failed to parse CivitAI settings", e)
        }
      }
      applyModeUI()
    }

    // Collapse behavior (copied style)
    if (!headerEl || !bodyEl || !handleEl) {
      throw new Error("Collapse UI missing: header/body/handle not found")
    }

    function setCollapsed(collapsed) {
      bodyEl.style.display = collapsed ? "none" : "block"
      handleEl.textContent = collapsed ? "➕" : "➖"
      localStorage.setItem(LS_COLLAPSE_KEY, collapsed ? "1" : "0")
    }

    const startCollapsed = localStorage.getItem(LS_COLLAPSE_KEY) !== "0" // default collapsed
    setCollapsed(startCollapsed)

    headerEl.addEventListener("click", () => {
      const isCollapsed = bodyEl.style.display === "none"
      setCollapsed(!isCollapsed)
    })

    loadSettings()

    async function loadCommonTags() {
      try {
        const dynamicBaseUrl = await resolveDynamicBaseUrl(serverUrlEl.value)
        const res = await fetchWithFallback(
          `${TAGS_PATH}?limit=20`,
          {},
          dynamicBaseUrl
        )
        const json = await safeJson(res)
        if (!json.ok || !Array.isArray(json.items)) return
        globalTagOptionsEl.innerHTML = ""
        json.items.forEach((tag) => {
          const name = String((tag && tag.name) || "").trim()
          if (!name) return
          const option = document.createElement("option")
          option.value = name
          globalTagOptionsEl.appendChild(option)
        })
      } catch (err) {
        console.debug("CivitAI common tags unavailable", err)
      }
    }

    loadCommonTags()

    async function loadEnums() {
      try {
        const dynamicBaseUrl = await resolveDynamicBaseUrl(serverUrlEl.value)
        const res = await fetchWithFallback(ENUMS_PATH, {}, dynamicBaseUrl)
        const json = await safeJson(res)
        if (!json.ok) return

        const populate = (select, values, emptyLabel) => {
          if (!Array.isArray(values) || values.length === 0) return
          const current = select.value
          select.innerHTML = [
            `<option value="">${emptyLabel}</option>`,
            ...values.map((value) => {
              const clean = String(value || "").trim()
              return clean ? `<option value="${clean}">${clean}</option>` : ""
            }),
          ].join("")
          select.value = values.includes(current) || current === "" ? current : ""
        }

        populate(modelTypeEl, json.modelTypes, "All Model Types")
        populate(baseModelEl, json.baseModels, "All Base Models")
        populate(globalBaseEl, json.baseModels, "All Base Models")
      } catch (err) {
        console.debug("CivitAI enums unavailable; using built-in options", err)
      }
    }

    loadEnums()

    let creatorRequestSequence = 0
    async function loadCreators(query = "") {
      const requestSequence = ++creatorRequestSequence
      try {
        const dynamicBaseUrl = await resolveDynamicBaseUrl(serverUrlEl.value)
        const queryPart = query.trim()
          ? `&query=${encodeURIComponent(query.trim())}`
          : ""
        const res = await fetchWithFallback(
          `${CREATORS_PATH}?limit=20${queryPart}`,
          {},
          dynamicBaseUrl
        )
        const json = await safeJson(res)
        if (
          requestSequence !== creatorRequestSequence ||
          !json.ok ||
          !Array.isArray(json.items)
        ) {
          return
        }
        creatorOptionsEl.innerHTML = ""
        json.items.forEach((creator) => {
          const username = String((creator && creator.username) || "").trim()
          if (!username) return
          const option = document.createElement("option")
          option.value = username
          creatorOptionsEl.appendChild(option)
        })
      } catch (err) {
        console.debug("CivitAI creator lookup unavailable", err)
      }
    }

    loadCreators()
    let creatorSearchTimer = null
    ;[usernameEl, globalUsernameEl].forEach((el) =>
      el.addEventListener("input", () => {
        clearTimeout(creatorSearchTimer)
        creatorSearchTimer = setTimeout(() => {
          const query = el.value.trim()
          if (query.length >= 2 || !query) loadCreators(query)
        }, 300)
      })
    )

    ;[sortEl, periodEl, nsfwEl, apiKeyEl, searchKeyEl, serverUrlEl, postIdEl, usernameEl, modelSortEl].forEach((el) =>
      el.addEventListener("change", saveSettings)
    )
    ;[
      modelTypeEl,
      baseModelEl,
      globalTagEl,
      globalTechniqueEl,
      globalToolEl,
      globalAspectEl,
      globalBaseEl,
      globalTypeEl,
      globalUsernameEl,
      globalCreatedFromEl,
      globalCreatedToEl,
    ].forEach((el) => el.addEventListener("change", saveSettings))

    modeEls.forEach((el) =>
      el.addEventListener("change", () => {
        if (getMode() === "images" && !selectedModel) {
          const modelRadio = panel.querySelector(
            `input[name="${PLUGIN_ID}-mode"][value="models"]`
          )
          if (modelRadio) modelRadio.checked = true
          applyModeUI()
          saveSettings()
          statusEl.textContent = "Select View Images on a model result first."
          return
        }

        applyModeUI()
        saveSettings()

        currentPage = 1
        currentCursor = null
        nextCursor = null
        prevCursorStack.length = 0

        pageLabel.textContent = `Page ${currentPage}`
        resultsEl.innerHTML = ""
        if (currentMode === "images") {
          runSearch("", undefined)
        } else if (currentMode === "global-images") {
          runSearch(queryEl.value || "", undefined)
        } else {
          runSearch(modelSearchQuery || queryEl.value || "", undefined)
        }
      })
    )

    clearBtn.addEventListener("click", () => {
      apiKeyEl.value = ""
      searchKeyEl.value = ""
      saveSettings()
    })

    verifyKeyBtn.addEventListener("click", async () => {
      verifyKeyBtn.disabled = true
      statusEl.textContent = "Verifying CivitAI API token..."
      try {
        const dynamicBaseUrl = await resolveDynamicBaseUrl(serverUrlEl.value)
        const res = await fetchWithFallback(
          ME_PATH,
          { headers: headersForRequest() },
          dynamicBaseUrl
        )
        const json = await safeJson(res)
        if (!json.ok) {
          throw new Error(json.error || json.detail || "Token verification failed")
        }
        const user = json.user || {}
        const identity = user.username || user.name || user.id
        statusEl.textContent = identity
          ? `CivitAI token verified for ${identity}.`
          : "CivitAI token verified."
      } catch (err) {
        console.error(err)
        statusEl.textContent =
          "Token verification error: " +
          (err && err.message ? err.message : "Server Error")
      } finally {
        verifyKeyBtn.disabled = false
      }
    })

    function updateAdvancedLookupUI() {
      const type = advancedTypeEl.value
      const placeholders = {
        hash: "Complete SHA256",
        mini: "Model version ID",
        hashes: "SHA256 values separated by spaces or commas",
        "hash-ids": "SHA256 values separated by spaces or commas",
        collections: "Collection search query",
        collection: "Collection ID",
        permissions: "Entity IDs separated by commas",
      }
      advancedValueEl.placeholder = placeholders[type] || "Lookup value"
      advancedUserIdEl.style.display = type === "permissions" ? "block" : "none"
    }

    advancedTypeEl.addEventListener("change", updateAdvancedLookupUI)
    updateAdvancedLookupUI()

    async function runAdvancedLookup() {
      const type = advancedTypeEl.value
      const value = advancedValueEl.value.trim()
      if (!value && type !== "collections") {
        advancedResultEl.style.display = "block"
        advancedResultEl.textContent = "Enter a lookup value."
        return
      }

      advancedRunBtn.disabled = true
      advancedResultEl.style.display = "block"
      advancedResultEl.textContent = "Loading..."
      try {
        let path = ""
        let options = { headers: headersForRequest() }
        if (type === "hash") {
          path = `${LOOKUP_HASH_PATH}/${encodeURIComponent(value)}`
        } else if (type === "mini") {
          path = `${LOOKUP_MINI_PATH}/${encodeURIComponent(value)}`
        } else if (type === "hashes" || type === "hash-ids") {
          const hashes = value.split(/[\s,]+/).filter(Boolean)
          path = LOOKUP_HASHES_PATH
          options = {
            method: "POST",
            headers: {
              ...headersForRequest(),
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              hashes,
              idsOnly: type === "hash-ids",
            }),
          }
        } else if (type === "collections") {
          path =
            `${COLLECTIONS_PATH}?limit=5&sort=${encodeURIComponent("Newest")}` +
            `&query=${encodeURIComponent(value)}`
        } else if (type === "collection") {
          path = `${COLLECTIONS_PATH}/${encodeURIComponent(value)}`
        } else if (type === "permissions") {
          path = `${PERMISSIONS_PATH}?entityIds=${encodeURIComponent(value)}`
          const userId = advancedUserIdEl.value.trim()
          if (userId) path += `&userId=${encodeURIComponent(userId)}`
        }

        const dynamicBaseUrl = await resolveDynamicBaseUrl(serverUrlEl.value)
        const res = await fetchWithFallback(path, options, dynamicBaseUrl)
        const json = await safeJson(res)
        if (!json.ok) {
          throw new Error(json.error || json.detail || "Lookup failed")
        }
        advancedResultEl.textContent = JSON.stringify(
          Object.prototype.hasOwnProperty.call(json, "result")
            ? json.result
            : json,
          null,
          2
        )
      } catch (err) {
        console.error(err)
        advancedResultEl.textContent =
          "Lookup error: " +
          (err && err.message ? err.message : "Server Error")
      } finally {
        advancedRunBtn.disabled = false
      }
    }

    advancedRunBtn.addEventListener("click", runAdvancedLookup)
    advancedValueEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") runAdvancedLookup()
    })

    function parseUrn(raw) {
      const s = (raw || "").trim()
      if (!s.toLowerCase().startsWith("urn:")) return null
      const m = s.match(/civitai:(\d+)(?:@(\d+))?$/i)
      if (!m) return null
      return { modelId: parseInt(m[1], 10), versionId: m[2] ? parseInt(m[2], 10) : null }
    }

    function parseModelReference(raw) {
      const urn = parseUrn(raw)
      if (urn) return urn

      const value = (raw || "").trim()
      if (/^[a-f0-9]{64}$/i.test(value)) {
        return { hash: value.toUpperCase() }
      }

      let match = value.match(/^model:(\d+)$/i)
      if (match) return { modelId: parseInt(match[1], 10), versionId: null }
      match = value.match(/^version:(\d+)$/i)
      if (match) return { modelId: null, versionId: parseInt(match[1], 10) }
      match = value.match(/^hash:([a-f0-9]{64})$/i)
      if (match) return { hash: match[1].toUpperCase() }

      if (!/^https?:\/\//i.test(value)) return null
      try {
        const url = new URL(value)
        const host = url.hostname.toLowerCase().replace(/^www\./, "")
        if (!["civitai.com", "civitai.red"].includes(host)) return null
        const path = url.pathname.replace(/\/+$/, "")

        match = path.match(/\/api\/v1\/model-versions\/by-hash\/([a-f0-9]{64})$/i)
        if (match) return { hash: match[1].toUpperCase() }
        match = path.match(/\/api\/v1\/model-versions\/(?:mini\/)?(\d+)$/i)
        if (match) return { modelId: null, versionId: parseInt(match[1], 10) }
        match = path.match(/\/api\/v1\/models\/(\d+)$/i)
        if (match) return { modelId: parseInt(match[1], 10), versionId: null }
      } catch (_) {}
      return null
    }

    function importGlobalSearchUrl(raw) {
      const value = (raw || "").trim()
      if (!/^https?:\/\//i.test(value)) return null

      try {
        const url = new URL(value)
        const host = url.hostname.toLowerCase().replace(/^www\./, "")
        if (!["civitai.red", "civitai.com"].includes(host)) return null
        if (!/^\/search\/images\/?$/.test(url.pathname)) return null

        const first = (...names) => {
          for (const name of names) {
            const match = url.searchParams.getAll(name).find((item) => item.trim())
            if (match) return match.trim()
          }
          return ""
        }

        const timestampToLocalInput = (rawTimestamp) => {
          const timestamp = Number(rawTimestamp)
          if (!Number.isFinite(timestamp) || timestamp <= 0) return ""
          const date = new Date(timestamp)
          if (Number.isNaN(date.getTime())) return ""
          const pad = (part) => String(part).padStart(2, "0")
          return (
            `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
            `T${pad(date.getHours())}:${pad(date.getMinutes())}`
          )
        }
        const createdAt = first("createdAt")
        const [createdFrom = "", createdTo = ""] = createdAt.split(":", 2)
        const imported = {
          query: first("query", "q"),
          tag: first("tags", "tag"),
          technique: first("techniques", "technique"),
          tool: first("tools", "tool"),
          aspectRatio: first("aspectRatio"),
          baseModel: first("baseModel"),
          mediaType: first("type", "mediaType") || "image",
          username: first("users", "user", "username"),
          createdFrom: timestampToLocalInput(createdFrom),
          createdTo: timestampToLocalInput(createdTo),
        }

        globalTagEl.value = imported.tag
        globalTechniqueEl.value = imported.technique
        globalToolEl.value = imported.tool
        globalAspectEl.value = imported.aspectRatio
        globalBaseEl.value = imported.baseModel
        globalTypeEl.value = imported.mediaType
        globalUsernameEl.value = imported.username
        globalCreatedFromEl.value = imported.createdFrom
        globalCreatedToEl.value = imported.createdTo
        queryEl.value = imported.query
        saveSettings()
        return imported.query
      } catch (_) {
        return null
      }
    }

    function tagChip(text) {
      return `<span style="font-size:0.75em; padding:2px 6px; border-radius:999px; background:var(--background-color3); color:var(--text-color2);">${text}</span>`
    }

    function describeDownload(item, file) {
      const modelName = (item && item.name) || "Untitled"
      const fileName = (file && file.name) || "model file"
      return `${modelName} / ${fileName}`
    }

    function queueSuffix() {
      return downloadQueue.length > 0 ? ` (${downloadQueue.length} queued)` : ""
    }

    async function runSearch(q, cursorOverride = undefined) {
      applyModeUI()

      if (currentMode === "images" && !selectedModel) {
        renderEmptyState("Select View Images on a model result first.")
        statusEl.textContent = "No model selected."
        prevBtn.disabled = true
        nextBtn.disabled = true
        return
      }

      if (currentMode === "global-images") {
        const importedQuery = importGlobalSearchUrl(q)
        if (importedQuery !== null) q = importedQuery
      }

      const isPaging = cursorOverride !== undefined
      if (!isPaging) {
        currentPage = 1
        currentCursor = null
        nextCursor = null
        prevCursorStack.length = 0
      }

      if (currentMode === "models") {
        modelSearchQuery = q || ""
      }
      lastQuery = currentMode === "images" ? "" : q || ""
      resultsEl.innerHTML = ""
      statusEl.textContent =
        currentMode === "images"
          ? `Loading images made with ${selectedModel.name}...`
          : currentMode === "global-images"
            ? "Searching all CivitAI images..."
            : "Searching CivitAI models..."
      pageLabel.textContent = `Page ${currentPage}`

      try {
        const urn = currentMode === "models" ? parseModelReference(q) : null
        const path =
          currentMode === "images"
            ? IMAGES_PATH
            : currentMode === "global-images"
              ? GLOBAL_IMAGES_PATH
              : SEARCH_PATH
        const resultLimit =
          currentMode === "images" ? 60 : currentMode === "global-images" ? 51 : 20
        let query = `?limit=${encodeURIComponent(resultLimit)}`
        const useCursor = cursorOverride !== undefined ? cursorOverride : currentCursor
        const qTrim = currentMode === "images" ? "" : (q || "").trim()

        if (currentMode === "images" || currentMode === "global-images") {
          if (useCursor) query += `&cursor=${encodeURIComponent(useCursor)}`
          else query += `&page=${encodeURIComponent(String(currentPage))}`
        } else {
          const isUrn = !!urn
          const isQuerySearch = !isUrn && qTrim.length > 0

          if (isQuerySearch) {
            if (useCursor) query += `&cursor=${encodeURIComponent(useCursor)}`
          } else {
            if (useCursor) query += `&cursor=${encodeURIComponent(useCursor)}`
            else query += `&page=${encodeURIComponent(String(currentPage))}`
          }
        }

        if (nsfwEl.checked) query += `&nsfw=true`
        if (currentMode !== "global-images") {
          if (sortEl.value) query += `&sort=${encodeURIComponent(sortEl.value)}`
          if (periodEl.value) query += `&period=${encodeURIComponent(periodEl.value)}`
        }

        if (currentMode === "images") {
          query += `&modelId=${encodeURIComponent(String(selectedModel.id))}`
        } else if (currentMode === "global-images") {
          query += `&query=${encodeURIComponent(q || "")}`
        } else if (urn) {
          if (urn.hash) {
            query += `&hash=${encodeURIComponent(urn.hash)}`
          } else {
            if (urn.modelId != null) {
              query += `&modelId=${encodeURIComponent(String(urn.modelId))}`
            }
            if (urn.versionId != null) {
              query += `&modelVersionId=${encodeURIComponent(String(urn.versionId))}`
            }
          }
        } else {
          query += `&query=${encodeURIComponent(q || "")}`
        }

        if (currentMode === "images") {
          const pid = (postIdEl.value || "").trim()
          const un = (usernameEl.value || "").trim()
          if (pid) query += `&postId=${encodeURIComponent(pid)}`
          if (un) query += `&username=${encodeURIComponent(un)}`
        } else if (currentMode === "models") {
          const modelType = (modelTypeEl.value || "").trim()
          const baseModel = (baseModelEl.value || "").trim()
          if (modelType) query += `&types=${encodeURIComponent(modelType)}`
          if (baseModel) query += `&baseModels=${encodeURIComponent(baseModel)}`
        } else if (currentMode === "global-images") {
          const toUnixSeconds = (value) => {
            if (!value) return ""
            const timestamp = new Date(value).getTime()
            return Number.isFinite(timestamp) ? String(Math.floor(timestamp / 1000)) : ""
          }
          const globalFilters = {
            tag: globalTagEl.value.trim(),
            technique: globalTechniqueEl.value.trim(),
            tool: globalToolEl.value.trim(),
            aspectRatio: globalAspectEl.value,
            baseModel: globalBaseEl.value,
            mediaType: globalTypeEl.value,
            username: globalUsernameEl.value.trim(),
            createdFrom: toUnixSeconds(globalCreatedFromEl.value),
            createdTo: toUnixSeconds(globalCreatedToEl.value),
          }
          Object.entries(globalFilters).forEach(([key, value]) => {
            if (value) query += `&${key}=${encodeURIComponent(value)}`
          })
        }

        const dynamicBaseUrl = await resolveDynamicBaseUrl(serverUrlEl.value)
        const res = await fetchWithFallback(
          `${path}${query}`,
          { headers: headersForRequest() },
          dynamicBaseUrl
        )
        const json = await safeJson(res)
        if (!json.ok) throw new Error(json.error || json.detail || "Search failed")

        if (currentMode === "images" || currentMode === "global-images") {
          const items = json.items || []
          if (items.length === 0) {
            renderEmptyState(
              currentMode === "images"
                ? `No images found for ${selectedModel.name}.`
                : "No global images found for this search."
            )
          } else {
            renderImages(items, currentMode === "images")
          }
          nextCursor = extractCursor(json.metadata?.nextCursor || json.metadata?.nextPage) || null
          const totalPages = json.metadata?.pages || null
          prevBtn.disabled = currentPage <= 1 && prevCursorStack.length === 0
          nextBtn.disabled = nextCursor ? false : totalPages ? currentPage >= totalPages : items.length === 0
          const unifiedTotal = Number(json.metadata?.total)
          if (currentMode === "global-images") {
            statusEl.textContent = Number.isFinite(unifiedTotal)
              ? `${items.length} images on page ${currentPage} of ${Math.max(
                  1,
                  Math.ceil(unifiedTotal / resultLimit)
                )}`
              : `${items.length} global images`
          } else if (Number.isFinite(unifiedTotal) && unifiedTotal > items.length) {
            statusEl.textContent = `${items.length} of ${unifiedTotal} images for ${selectedModel.name}`
          } else {
            statusEl.textContent = `${items.length} images for ${selectedModel.name}`
          }
        } else {
          const items = json.items || []
          if (items.length === 0) {
            renderEmptyState("No models found. Try clearing model type/base model filters.")
          } else {
            renderModels(items)
          }
          nextCursor = extractCursor(json.metadata?.nextCursor) || null

          const qTrim = (q || "").trim()
          const isUrn = !!urn
          const isQuerySearch = !isUrn && qTrim.length > 0
          const totalPages = json.metadata?.pages || null

          prevBtn.disabled = currentPage <= 1 && prevCursorStack.length === 0

          if (isQuerySearch) {
            nextBtn.disabled = !nextCursor
          } else {
            nextBtn.disabled = nextCursor ? false : totalPages ? currentPage >= totalPages : false
          }

          statusEl.textContent = `${items.length} models`
        }
      } catch (err) {
        console.error(err)
        statusEl.textContent = "Search error: " + (err && err.message ? err.message : "Server Error")
        prevBtn.disabled = true
        nextBtn.disabled = true
      }
    }

    function renderModels(items) {
      resultsEl.innerHTML = ""
      items.forEach((item) => {
        const mv =
          item.modelVersion ||
          (Array.isArray(item.modelVersions) ? item.modelVersions[0] : {}) ||
          {}

        const previewImage = (mv.images && mv.images[0]) || {}
        const thumb = previewImage.thumbnailUrl || previewImage.url || ""
        const files = mv.files || []

        const creatorName =
          (item.creator && (item.creator.username || item.creator.name || item.creator.handle)) ||
          item.creator ||
          "?"

        const card = document.createElement("div")
        card.style =
          "border:1px solid var(--background-color3); border-radius:8px; padding:8px; display:grid; grid-template-columns:76px 1fr; gap:8px; margin-bottom:4px;"
        card.innerHTML = `
          <div style="width:76px;height:76px;overflow:hidden;border-radius:6px;background:var(--background-color3);display:flex;align-items:center;justify-content:center;">
            ${
              thumb
                ? `<img src="${thumb}" style="width:100%;height:100%;object-fit:cover;" />`
                : `<span style="font-size:10px;">No preview</span>`
            }
          </div>
          <div style="display:flex;flex-direction:column;gap:6px;">
            <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
              <div style="font-weight:600; font-size:0.95em;">${item.name || "Untitled"}</div>
            </div>

            <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
              ${tagChip(`Type: ${item.type || "?"}`)}
              ${tagChip(`Base: ${mv.baseModel || "?"}`)}
              ${tagChip(`Creator: ${creatorName}`)}
            </div>

            <div style="font-size:0.8em;color:var(--text-color2);">
              ModelID: ${item.id || "?"} • VersionID: ${mv.id || "?"}
            </div>

            <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
              ${
                previewImage.url
                  ? `<a class="secondaryButton" href="${previewImage.url}" target="_blank" rel="noopener">Preview</a>`
                  : ""
              }
              ${
                item.id && mv.id
                  ? `<button class="secondaryButton" data-action="copy-urn">Copy URN</button>`
                  : ""
              }
              ${
                item.id
                  ? `<button class="primaryButton" data-action="view-images">View Images</button>`
                  : ""
              }
            </div>

            <div style="display:flex;flex-direction:column;gap:4px;border-top:1px solid var(--background-color3);padding-top:4px;max-height:140px;overflow:auto;">
              ${
                files.length
                  ? files
                      .map((f, idx) => {
                        const sizeMB = f.sizeKB ? (f.sizeKB / 1024).toFixed(1) + " MB" : "size?"
                        return `
                          <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.8em;background:var(--background-color2);padding:2px 6px;border-radius:4px;">
                            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:140px;">${
                              f.name || `File ${idx + 1}`
                            }</span>
                            <button class="primaryButton" style="padding:1px 6px;" data-action="dl" data-idx="${idx}">Get</button>
                            <span style="font-size:0.75em;color:var(--text-color2);margin-left:6px;">${sizeMB}</span>
                          </div>`
                      })
                      .join("")
                  : `<div style="font-size:0.85em;color:var(--text-color2);">No downloadable files</div>`
              }
            </div>
          </div>
        `

        card.querySelectorAll("[data-action=dl]").forEach((b) => {
          b.addEventListener("click", () => {
            const idx = parseInt(b.dataset.idx || "0", 10)
            handleDownload(item, files[idx])
          })
        })

        const copyBtn = card.querySelector("[data-action=copy-urn]")
        if (copyBtn && item.id && mv.id) {
          copyBtn.addEventListener("click", async () => {
            const urn = `urn:air:${airBaseModelKey(mv.baseModel)}:${String(
              item.type || "model"
            ).toLowerCase()}:civitai:${item.id}@${mv.id}`
            try {
              await navigator.clipboard.writeText(urn)
              statusEl.textContent = "Copied URN to clipboard."
            } catch (_) {
              statusEl.textContent = urn
            }
          })
        }

        const viewImagesBtn = card.querySelector("[data-action=view-images]")
        if (viewImagesBtn && item.id) {
          viewImagesBtn.addEventListener("click", () => {
            selectModelImages(item)
          })
        }

        resultsEl.appendChild(card)
      })
    }

    function selectModelImages(item) {
      const modelId = Number(item && item.id)
      if (!Number.isFinite(modelId)) {
        statusEl.textContent = "This result does not include a valid model ID."
        return
      }

      selectedModel = {
        id: modelId,
        name: String((item && item.name) || `Model ${modelId}`),
      }
      const imageRadio = panel.querySelector(
        `input[name="${PLUGIN_ID}-mode"][value="images"]`
      )
      if (imageRadio) imageRadio.checked = true
      applyModeUI()
      saveSettings()

      currentPage = 1
      currentCursor = null
      nextCursor = null
      prevCursorStack.length = 0
      runSearch("", undefined)
    }

    function returnToModels() {
      const modelRadio = panel.querySelector(
        `input[name="${PLUGIN_ID}-mode"][value="models"]`
      )
      if (modelRadio) modelRadio.checked = true
      applyModeUI()
      saveSettings()

      currentPage = 1
      currentCursor = null
      nextCursor = null
      prevCursorStack.length = 0
      runSearch(modelSearchQuery || queryEl.value || "", undefined)
    }

    function renderImages(items, showSelectedHeader = false) {
      resultsEl.innerHTML = ""

      if (showSelectedHeader && selectedModel) {
        const selectedHeader = document.createElement("div")
        selectedHeader.style =
          "display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;"
        const selectedLabel = document.createElement("strong")
        selectedLabel.textContent = `Images made with ${selectedModel.name}`
        const backButton = document.createElement("button")
        backButton.className = "secondaryButton"
        backButton.textContent = "Back to Models"
        backButton.addEventListener("click", returnToModels)
        selectedHeader.append(selectedLabel, backButton)
        resultsEl.appendChild(selectedHeader)
      }

      const grid = document.createElement("div")
      grid.style = "display:grid; grid-template-columns:repeat(auto-fill, minmax(120px, 1fr)); gap:8px;"
      resultsEl.appendChild(grid)

      items.forEach((img) => {
        const previewUrl = img.thumbnailUrl || img.url || ""
        const openUrl = img.url || img.thumbnailUrl || ""
        const hasMetadata =
          img.meta &&
          (typeof img.meta !== "object" || Object.keys(img.meta).length > 0)
        const card = document.createElement("div")
        card.style =
          "border:1px solid var(--background-color3); border-radius:8px; overflow:hidden; background:var(--background-color2); display:flex; flex-direction:column;"
        card.innerHTML = `
          <div style="width:100%; aspect-ratio:1/1; background:var(--background-color3); display:flex; align-items:center; justify-content:center; overflow:hidden;">
            ${
              previewUrl
                ? img.mediaType === "video"
                  ? `<video src="${previewUrl}" controls muted preload="metadata" style="width:100%;height:100%;object-fit:cover;"></video>`
                  : `<img src="${previewUrl}" style="width:100%;height:100%;object-fit:cover;" />`
                : `<span style="font-size:10px;">No image</span>`
            }
          </div>
          <div style="padding:6px; display:flex; flex-direction:column; gap:4px;">
            <div style="font-size:0.75em; color:var(--text-color2); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
              ID: ${img.id || "?"}
            </div>
            <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
              ${img.width && img.height ? tagChip(`${img.width}x${img.height}`) : ""}
              ${img.username ? tagChip(`${img.username}`) : ""}
              ${img.mediaType ? tagChip(`${img.mediaType}`) : ""}
            </div>
            ${
              openUrl
                ? `<a class="secondaryButton" href="${openUrl}" target="_blank" rel="noopener" style="text-align:center;">Open</a>`
                : ""
            }
            <div data-civitai-image-meta></div>
          </div>
        `
        const metadataSlot = card.querySelector("[data-civitai-image-meta]")
        if (hasMetadata && metadataSlot) {
          const details = document.createElement("details")
          details.style = "font-size:0.78em;"
          const summary = document.createElement("summary")
          summary.style = "cursor:pointer; user-select:none;"
          summary.textContent = "Generation metadata"
          const pre = document.createElement("pre")
          pre.style =
            "white-space:pre-wrap; overflow-wrap:anywhere; max-height:180px; overflow:auto; margin:6px 0 0; padding:6px; background:var(--background-color3); border-radius:6px;"
          pre.textContent =
            typeof img.meta === "string"
              ? img.meta
              : JSON.stringify(img.meta, null, 2)
          details.append(summary, pre)
          metadataSlot.appendChild(details)
        }
        grid.appendChild(card)
      })
    }

    async function startQueuedDownload(task) {
      statusEl.textContent = `Starting download: ${task.label}${queueSuffix()}`
      try {
        const headers = { "Content-Type": "application/json", ...headersForRequest() }

        const dynamicBaseUrl = await resolveDynamicBaseUrl(serverUrlEl.value)
        const res = await fetchWithPathFallback(
          DOWNLOAD_PATHS,
          {
            method: "POST",
            headers,
            body: JSON.stringify({ model: task.item, file: task.file }),
          },
          dynamicBaseUrl
        )
        const json = await safeJson(res)
        if (!json.ok) throw new Error(json.error || "Download failed")
        if (json.downloadId) {
          activeDownloadId = json.downloadId
          await pollDownloadStatus(
            json.downloadId,
            statusEl,
            headersForRequest(),
            dynamicBaseUrl,
            queueSuffix
          )
          activeDownloadId = null
          return
        }
        statusEl.textContent = `Downloaded to ${json.path} (${formatSize(json.size)})`
      } catch (err) {
        console.error(err)
        statusEl.textContent = "Download error: " + (err && err.message ? err.message : "Server Error")
      } finally {
        activeDownloadId = null
      }
    }

    async function processDownloadQueue() {
      if (isProcessingQueue) return
      isProcessingQueue = true
      try {
        while (downloadQueue.length > 0) {
          const task = downloadQueue.shift()
          await startQueuedDownload(task)
        }
        if (!activeDownloadId && !statusEl.textContent.startsWith("Download error:")) {
          statusEl.textContent = "Download queue complete."
        }
      } finally {
        isProcessingQueue = false
      }
    }

    function handleDownload(item, file) {
      const task = { item, file, label: describeDownload(item, file) }
      downloadQueue.push(task)
      if (isProcessingQueue || activeDownloadId) {
        statusEl.textContent = `Queued: ${task.label}${queueSuffix()}`
      } else {
        statusEl.textContent = `Queued: ${task.label}`
      }
      processDownloadQueue()
    }

    function run() {
      if (getMode() === "images" && !selectedModel) {
        statusEl.textContent = "Select View Images on a model result first."
        return
      }
      const q = getMode() === "images" ? "" : (queryEl.value || "").trim()
      currentPage = 1
      currentCursor = null
      nextCursor = null
      prevCursorStack.length = 0
      runSearch(q, undefined)
    }

    searchBtn.addEventListener("click", run)
    queryEl.addEventListener("keydown", (e) => e.key === "Enter" && run())

    nextBtn.addEventListener("click", () => {
      applyModeUI()
      const urn = parseUrn(lastQuery)
      const isQuerySearch = !urn && (lastQuery || "").trim().length > 0

      if (currentMode === "global-images") {
        currentPage++
        currentCursor = null
        runSearch(lastQuery, null)
        return
      }

      if (currentMode === "images") {
        if (nextCursor) {
          prevCursorStack.push(currentCursor)
          currentCursor = nextCursor
          currentPage++
          runSearch(lastQuery, currentCursor)
        } else if (!isQuerySearch) {
          currentPage++
          currentCursor = null
          runSearch(lastQuery, null)
        }
        return
      }

      if (nextCursor) {
        prevCursorStack.push(currentCursor)
        currentCursor = nextCursor
        currentPage++
        runSearch(lastQuery, currentCursor)
      } else if (!isQuerySearch) {
        currentPage++
        currentCursor = null
        runSearch(lastQuery, undefined)
      }
    })

    prevBtn.addEventListener("click", () => {
      applyModeUI()
      const urn = parseUrn(lastQuery)
      const isQuerySearch = !urn && (lastQuery || "").trim().length > 0

      if (currentMode === "global-images" && currentPage > 1) {
        currentPage = Math.max(1, currentPage - 1)
        currentCursor = null
        runSearch(lastQuery, null)
        return
      }

      if (currentMode === "images") {
        if (prevCursorStack.length > 0) {
          currentCursor = prevCursorStack.pop() || null
          currentPage = Math.max(1, currentPage - 1)
          runSearch(lastQuery, currentCursor)
        } else if (!isQuerySearch) {
          currentPage = Math.max(1, currentPage - 1)
          currentCursor = null
          runSearch(lastQuery, null)
        }
        return
      }

      if (prevCursorStack.length > 0) {
        currentCursor = prevCursorStack.pop() || null
        currentPage = Math.max(1, currentPage - 1)
        runSearch(lastQuery, currentCursor)
      } else if (!isQuerySearch && currentPage > 1) {
        currentPage = Math.max(1, currentPage - 1)
        currentCursor = null
        runSearch(lastQuery, undefined)
      }
    })

    statusEl.textContent =
      currentMode === "images" && selectedModel
        ? `Selected ${selectedModel.name}. Click Reload Images to refresh.`
        : currentMode === "global-images"
          ? "Enter a query to search all CivitAI images."
        : "Search models, then choose View Images on a result."
    return true
  }

  const timer = setInterval(() => {
    if (ready()) clearInterval(timer)
  }, 500)
})()
