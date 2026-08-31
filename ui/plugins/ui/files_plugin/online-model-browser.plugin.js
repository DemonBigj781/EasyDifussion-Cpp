;(function () {
  "use strict"

  const PLUGIN_ID = "civitai-downloader"
  const API_ROOT = "/civitai-api/civitai"
  const SEARCH_ENDPOINT = `${API_ROOT}/search`
  const IMAGES_ENDPOINT = `${API_ROOT}/global-images`
  const DOWNLOAD_ENDPOINT = `${API_ROOT}/download`
  const HF_API_ROOT = "/huggingface-api"
  const LS_SETTINGS_KEY = "civitaiSettings"

  // CivitAI enforces different sort enums for models vs. images; keep them separate.
  const SORT_OPTIONS = {
    models: [
      "Highest Rated",
      "Most Downloaded",
      "Newest",
    ],
    images: [
      "Most Reactions",
      "Most Comments",
      "Newest",
    ],
    huggingface: [
      "Most Downloaded",
      "Most Liked",
      "Recently Updated",
    ],
  }

  function ready() {
    if (typeof createTab !== "function") return false
    if (
      document.getElementById(`${PLUGIN_ID}-panel`) ||
      document.getElementById("tab-civitai") ||
      document.getElementById("tab-content-civitai")
    ) return true

    const panel = document.createElement("div")
    panel.id = `${PLUGIN_ID}-panel`
    panel.className = "panel-box"
    panel.style =
      "padding:12px; margin-bottom:8px; border:1px solid var(--background-color3); border-radius:8px;"
    panel.innerHTML = loadRequiredPluginHTML("/plugins/core/files_plugin/online-model-browser.plugin.html")
    createTab({ id: "civitai", label: "Online Model Browser", icon: "cloud-arrow-down", content: panel })

    const settingsTable = document.querySelector("#system-settings-table")
    if (settingsTable && !document.getElementById(`${PLUGIN_ID}-settings`)) {
      const row = document.createElement("div")
      row.id = `${PLUGIN_ID}-settings`
      row.innerHTML = `
        <div><i class="fa-solid fa-cloud-arrow-down"></i></div>
        <div>
          <label>Online Model Browser</label>
          <small>Credentials are saved in this browser and sent only to their matching provider.</small>
        </div>
        <div style="display:flex; flex-direction:column; gap:6px; min-width:0;">
          <div style="display:flex; gap:6px; align-items:center;">
            <input id="${PLUGIN_ID}-apikey" type="password" autocomplete="off"
              placeholder="Civitai API key" style="flex:1; min-width:0;">
            <button class="secondaryButton" id="${PLUGIN_ID}-clear" type="button">Clear</button>
          </div>
          <div style="display:flex; gap:6px; align-items:center;">
            <input id="${PLUGIN_ID}-hf-token" type="password" autocomplete="off"
              placeholder="Hugging Face token (optional)" style="flex:1; min-width:0;">
            <button class="secondaryButton" id="${PLUGIN_ID}-clear-hf" type="button">Clear</button>
          </div>
          <label style="display:flex; gap:6px; align-items:center;">
            <input type="checkbox" id="${PLUGIN_ID}-nsfw"> Include NSFW Civitai results
          </label>
        </div>`
      settingsTable.appendChild(row)
    }

    const providerEl = panel.querySelector(`#${PLUGIN_ID}-provider`)
    const queryEl = panel.querySelector(`#${PLUGIN_ID}-query`)
    const searchBtn = panel.querySelector(`#${PLUGIN_ID}-search`)
    const resultsEl = panel.querySelector(`#${PLUGIN_ID}-results`)
    const statusEl = panel.querySelector(`#${PLUGIN_ID}-status`)
    const prevBtn = panel.querySelector(`#${PLUGIN_ID}-prev`)
    const nextBtn = panel.querySelector(`#${PLUGIN_ID}-next`)
    const pageLabel = panel.querySelector(`#${PLUGIN_ID}-page`)
    const apiKeyEl = document.querySelector(`#${PLUGIN_ID}-apikey`)
    const hfTokenEl = document.querySelector(`#${PLUGIN_ID}-hf-token`)
    const clearBtn = document.querySelector(`#${PLUGIN_ID}-clear`)
    const clearHfBtn = document.querySelector(`#${PLUGIN_ID}-clear-hf`)
    const nsfwEl = document.querySelector(`#${PLUGIN_ID}-nsfw`)
    const sortEl = panel.querySelector(`#${PLUGIN_ID}-sort`)
    const periodEl = panel.querySelector(`#${PLUGIN_ID}-period`)
    const modeEls = panel.querySelectorAll(`input[name="${PLUGIN_ID}-mode"]`)
    const imagesLabel = panel.querySelector(`#${PLUGIN_ID}-images-label`)
    const helpEl = panel.querySelector(`#${PLUGIN_ID}-help`)

    const imgFiltersRow = panel.querySelector(`#${PLUGIN_ID}-imgfilters`)
    const postIdEl = panel.querySelector(`#${PLUGIN_ID}-postid`)
    const usernameEl = panel.querySelector(`#${PLUGIN_ID}-username`)

    // Paging state
    let currentPage = 1
    let currentMode = "models" // models | images
    let currentProvider = "civitai"
    let activeSearch = null
    let activeSearchController = null
    let searchGeneration = 0

    // Cursor paging (models)
    let currentCursor = null
    let nextCursor = null
    const prevCursorStack = [] // stack of cursors for previous pages (cursor used to fetch that page)

    function getMode() {
      const checked = panel.querySelector(`input[name="${PLUGIN_ID}-mode"]:checked`)
      return checked ? checked.value : "models"
    }

    function applyModeUI() {
      currentMode = getMode()
      imgFiltersRow.style.display = currentMode === "images" ? "flex" : "none"
      refreshSortOptions()
    }

    function applyProviderUI() {
      currentProvider = providerEl.value === "huggingface" ? "huggingface" : "civitai"
      const imageRadio = panel.querySelector(`input[name="${PLUGIN_ID}-mode"][value="images"]`)
      const isHuggingFace = currentProvider === "huggingface"
      imageRadio.disabled = isHuggingFace
      imagesLabel.style.opacity = isHuggingFace ? "0.5" : "1"
      if (isHuggingFace && getMode() === "images") {
        panel.querySelector(`input[name="${PLUGIN_ID}-mode"][value="models"]`).checked = true
      }
      periodEl.disabled = isHuggingFace
      queryEl.placeholder = isHuggingFace
        ? "Search Hugging Face models or enter owner/repository"
        : "Search term or Civitai URN (blank browses)"
      helpEl.textContent = isHuggingFace
        ? "Browse Hugging Face model repositories and download supported weight files into your configured models folder."
        : "Image search uses Civitai's images-v6 query when available and falls back to the public image API."
      applyModeUI()
    }

    function refreshSortOptions() {
      const key = currentProvider === "huggingface" ? "huggingface" : currentMode
      const allowed = SORT_OPTIONS[key] || SORT_OPTIONS.models
      const current = sortEl.value
      sortEl.innerHTML = allowed.map((s) => `<option value="${s}">${s}</option>`).join("")
      // Preserve value if still valid; otherwise reset to first option
      sortEl.value = allowed.includes(current) ? current : allowed[0]
    }

    function headersForRequest(keyOverride = undefined, providerOverride = currentProvider) {
      const h = {}
      const provider = providerOverride === "huggingface" ? "huggingface" : "civitai"
      const input = provider === "huggingface" ? hfTokenEl : apiKeyEl
      const k = String(keyOverride === undefined ? input.value || "" : keyOverride).trim()
      if (k) h[provider === "huggingface" ? "x-huggingface-token" : "x-civitai-key"] = k
      return h
    }

    function extractCursor(token) {
      if (!token) return null
      try {
        if (typeof token === "string" && token.startsWith("http")) {
          const url = new URL(token)
          return url.searchParams.get("cursor") || null
        }
      } catch (_) {
        /* ignore */
      }
      return token
    }

    async function safeJson(res) {
      const ct = (res.headers.get("content-type") || "").toLowerCase()
      if (ct.includes("application/json")) return res.json()
      const text = await res.text().catch(() => "")
      throw new Error(`HTTP ${res.status} (non-JSON): ${text.slice(0, 200)}`)
    }

    function saveSettings() {
      const settings = {
        sort: sortEl.value,
        period: periodEl.value,
        nsfw: nsfwEl.checked,
        apikey: apiKeyEl.value.trim(),
        huggingfaceToken: hfTokenEl.value.trim(),
        provider: currentProvider,
        mode: getMode(),
        postId: (postIdEl.value || "").trim(),
        username: (usernameEl.value || "").trim(),
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
          hfTokenEl.value = settings.huggingfaceToken || ""
          providerEl.value = settings.provider === "huggingface" ? "huggingface" : "civitai"
          currentProvider = providerEl.value
          postIdEl.value = settings.postId || ""
          usernameEl.value = settings.username || ""

          const m = settings.mode === "images" ? "images" : "models"
          const radio = panel.querySelector(`input[name="${PLUGIN_ID}-mode"][value="${m}"]`)
          if (radio) radio.checked = true
          currentMode = m
          refreshSortOptions()
          const sortKey = currentProvider === "huggingface" ? "huggingface" : currentMode
          const allowed = SORT_OPTIONS[sortKey] || []
          sortEl.value = allowed.includes(settings.sort) ? settings.sort : allowed[0]
        } catch (e) {
          console.warn("Failed to parse CivitAI settings", e)
        }
      }
      applyProviderUI()
    }

    loadSettings()

    ;[sortEl, periodEl, nsfwEl, apiKeyEl, hfTokenEl, postIdEl, usernameEl].forEach((el) =>
      el.addEventListener("change", saveSettings)
    )

    modeEls.forEach((el) =>
      el.addEventListener("change", () => {
        applyModeUI()
        saveSettings()
      })
    )

    providerEl.addEventListener("change", () => {
      applyProviderUI()
      saveSettings()
      resultsEl.innerHTML = ""
      statusEl.textContent = currentProvider === "huggingface" ? "Hugging Face model mode." : "Civitai model mode."
      activeSearch = null
      currentPage = 1
      pageLabel.textContent = "Page 1"
      prevBtn.disabled = true
      nextBtn.disabled = true
    })

    clearBtn.addEventListener("click", () => {
      apiKeyEl.value = ""
      saveSettings()
    })

    clearHfBtn.addEventListener("click", () => {
      hfTokenEl.value = ""
      saveSettings()
    })

    function parseUrn(raw) {
      // urn:air:sdxl:lora:civitai:MODELID@VERSIONID
      const s = (raw || "").trim()
      if (!s.toLowerCase().startsWith("urn:")) return null
      // allow anything in the front; we only care about the tail civitai:<mid>@<vid?>
      const m = s.match(/civitai:(\d+)(?:@(\d+))?$/i)
      if (!m) return null
      return { modelId: parseInt(m[1], 10), versionId: m[2] ? parseInt(m[2], 10) : null }
    }

    function escapeHTML(value) {
      return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
      })[character])
    }

    function safeExternalUrl(value) {
      try {
        const parsed = new URL(String(value || ""))
        return parsed.protocol === "https:" ? parsed.href : ""
      } catch (_) {
        return ""
      }
    }

    function tagChip(text) {
      return `<span style="font-size:0.75em; padding:2px 6px; border-radius:999px; background:var(--background-color3); color:var(--text-color2);">${escapeHTML(text)}</span>`
    }

    async function runSearch(search, cursorOverride = undefined) {
      // Allow blank query: browse recent, filtered by sort/period/nsfw
      const q = search.query
      const mode = search.mode

      // A setting can change while an earlier request is still running. Abort
      // it and ignore stale responses so old cards cannot replace new results.
      activeSearchController?.abort()
      const controller = typeof AbortController === "function" ? new AbortController() : null
      activeSearchController = controller
      const generation = ++searchGeneration

      resultsEl.innerHTML = ""
      statusEl.textContent = search.provider === "huggingface"
        ? "Searching Hugging Face models..."
        : mode === "images" ? "Searching Civitai images..." : "Searching Civitai models..."
      pageLabel.textContent = `Page ${currentPage}`

      try {
        if (search.provider === "huggingface") {
          let url = `${HF_API_ROOT}/search?limit=20&page=${encodeURIComponent(String(currentPage))}`
          url += `&query=${encodeURIComponent(q || "")}`
          if (search.sort) url += `&sort=${encodeURIComponent(search.sort)}`
          const requestOptions = { headers: headersForRequest(search.apiKey, search.provider) }
          if (controller) requestOptions.signal = controller.signal
          const res = await fetch(url, requestOptions)
          const json = await safeJson(res)
          if (generation !== searchGeneration) return
          if (!res.ok || !json.ok) throw new Error(json.detail || json.error || "Search failed")
          renderModels(json.items || [])
          prevBtn.disabled = currentPage <= 1
          nextBtn.disabled = !json.metadata?.hasNext
          nextCursor = null
          statusEl.textContent = `${json.items?.length || 0} Hugging Face models`
          return
        }

        const urn = parseUrn(q)
        const endpoint = mode === "images" ? IMAGES_ENDPOINT : SEARCH_ENDPOINT

        // build base url
        let url = `${endpoint}?limit=${encodeURIComponent(mode === "images" ? 60 : 20)}`

        if (mode === "images") {
          url += `&page=${encodeURIComponent(String(currentPage))}`
        } else {
          // models:
          // - If query is blank => page-based browse is allowed
          // - If query is non-blank => CivitAI requires cursor-based pagination (NO page param)
          const qTrim = (q || "").trim()
          const isUrn = !!urn
          const isQuerySearch = !isUrn && qTrim.length > 0

          const useCursor = cursorOverride !== undefined ? cursorOverride : currentCursor

          if (isQuerySearch) {
            if (useCursor) url += `&cursor=${encodeURIComponent(useCursor)}`
            // else: first page of query search: omit both page and cursor
          } else {
            // browse (blank query) OR URN lookup
            if (useCursor) url += `&cursor=${encodeURIComponent(useCursor)}`
            else url += `&page=${encodeURIComponent(String(currentPage))}`
          }
        }

        if (search.nsfw) url += `&nsfw=true`
        if (search.sort) url += `&sort=${encodeURIComponent(search.sort)}`
        if (search.period) url += `&period=${encodeURIComponent(search.period)}`

        // If URN, prefer direct lookup in server (modelId/modelVersionId) for BOTH modes.
        if (urn) {
          url += `&modelId=${encodeURIComponent(String(urn.modelId))}`
          if (urn.versionId != null) {
            url += `&modelVersionId=${encodeURIComponent(String(urn.versionId))}`
          }
        } else {
          // normal query (can be blank => browse)
          url += `&query=${encodeURIComponent(q || "")}`
        }

        // Images extra filters (only meaningful in images mode)
        if (mode === "images") {
          if (search.postId) url += `&postId=${encodeURIComponent(search.postId)}`
          if (search.username) url += `&username=${encodeURIComponent(search.username)}`
        }

        const requestOptions = { headers: headersForRequest(search.apiKey, search.provider) }
        if (controller) requestOptions.signal = controller.signal
        const res = await fetch(url, requestOptions)
        const json = await safeJson(res)
        if (generation !== searchGeneration) return
        if (!json.ok) throw new Error(json.error || "Search failed")

        if (mode === "images") {
          renderImages(json.items || [])
          const totalPages = json.metadata?.pages || null
          prevBtn.disabled = currentPage <= 1
          nextBtn.disabled = totalPages ? currentPage >= totalPages : (json.items || []).length === 0
          statusEl.textContent = `${json.items?.length || 0} images`
          // cursor state irrelevant
          nextCursor = null
        } else {
          renderModels(json.items || [])
          nextCursor = extractCursor(json.metadata?.nextCursor) || null

          // If this is browse mode (blank query) and server provides total pages, allow page-based next.
          const qTrim = (q || "").trim()
          const isUrn = !!urn
          const isQuerySearch = !isUrn && qTrim.length > 0
          const totalPages = json.metadata?.pages || null

          prevBtn.disabled = currentPage <= 1 && prevCursorStack.length === 0

          if (isQuerySearch) {
            nextBtn.disabled = !nextCursor
          } else {
            // browse or URN: if cursor exists, use it; else fall back to pages if available
            nextBtn.disabled = nextCursor ? false : (totalPages ? currentPage >= totalPages : false)
          }

          statusEl.textContent = `${json.items?.length || 0} models`
        }
      } catch (err) {
        if (err?.name === "AbortError" || generation !== searchGeneration) return
        console.error(err)
        statusEl.textContent = "Search error: " + (err && err.message ? err.message : "Server Error")
        prevBtn.disabled = true
        nextBtn.disabled = true
      } finally {
        if (controller && activeSearchController === controller) activeSearchController = null
      }
    }

    function renderModels(items) {
      resultsEl.innerHTML = ""
      items.forEach((item) => {
        // IMPORTANT: server returns item.modelVersion (singular)
        const mv =
          item.modelVersion ||
          (Array.isArray(item.modelVersions) ? item.modelVersions[0] : {}) ||
          {}

        const previewImage = (mv.images && mv.images[0]) || {}
        const thumb = safeExternalUrl(previewImage.thumbnailUrl || previewImage.url || "")
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
              <div style="font-weight:600; font-size:0.95em;">${escapeHTML(item.name || "Untitled")}</div>
            </div>

            <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
              ${tagChip(`Type: ${item.type || "?"}`)}
              ${tagChip(`Base: ${mv.baseModel || "?"}`)}
              ${tagChip(`Creator: ${creatorName}`)}
              ${item.provider === "huggingface" ? tagChip("Hugging Face") : tagChip("Civitai")}
            </div>

            <div style="font-size:0.8em;color:var(--text-color2);">
              ${item.provider === "huggingface" ? "Repository" : "ModelID"}: ${escapeHTML(item.id || "?")}
              ${item.provider === "huggingface" ? ` • Downloads: ${Number(item.downloads || 0)} • Likes: ${Number(item.likes || 0)}` : ` • VersionID: ${escapeHTML(mv.id || "?")}`}
            </div>

            <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
              ${
                safeExternalUrl(previewImage.url)
                  ? `<a class="secondaryButton" href="${safeExternalUrl(previewImage.url)}" target="_blank" rel="noopener">Preview</a>`
                  : ""
              }
              ${
                item.provider !== "huggingface" && item.id && mv.id
                  ? `<button class="secondaryButton" data-action="copy-urn">Copy URN</button>`
                  : ""
              }
              ${
                safeExternalUrl(item.pageUrl)
                  ? `<a class="secondaryButton" href="${safeExternalUrl(item.pageUrl)}" target="_blank" rel="noopener">Model page</a>`
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
                              escapeHTML(f.name || `File ${idx + 1}`)
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
            const urn = `urn:air:${
              (mv.baseModel || "sd").toLowerCase().includes("xl") ? "sdxl" : "sd"
            }:${String(item.type || "model").toLowerCase()}:civitai:${item.id}@${mv.id}`
            try {
              await navigator.clipboard.writeText(urn)
              statusEl.textContent = "Copied URN to clipboard."
            } catch (_) {
              statusEl.textContent = urn
            }
          })
        }

        resultsEl.appendChild(card)
      })
    }

    function renderImages(items) {
      resultsEl.innerHTML = ""

      const grid = document.createElement("div")
      grid.style = "display:grid; grid-template-columns:repeat(auto-fill, minmax(120px, 1fr)); gap:8px;"
      resultsEl.appendChild(grid)

      items.forEach((img) => {
        const url = safeExternalUrl(img.url || img.thumbnailUrl || "")
        const card = document.createElement("div")
        card.style =
          "border:1px solid var(--background-color3); border-radius:8px; overflow:hidden; background:var(--background-color2); display:flex; flex-direction:column;"
        card.innerHTML = `
          <div style="width:100%; aspect-ratio:1/1; background:var(--background-color3); display:flex; align-items:center; justify-content:center; overflow:hidden;">
            ${
              url
                ? `<img src="${url}" style="width:100%;height:100%;object-fit:cover;" />`
                : `<span style="font-size:10px;">No image</span>`
            }
          </div>
          <div style="padding:6px; display:flex; flex-direction:column; gap:4px;">
            <div style="font-size:0.75em; color:var(--text-color2); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
              ID: ${escapeHTML(img.id || "?")}
            </div>
            <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
              ${img.width && img.height ? tagChip(`${img.width}x${img.height}`) : ""}
              ${img.username ? tagChip(`${img.username}`) : ""}
            </div>
            ${
              img.url
                ? `<div style="display:flex;gap:4px;flex-wrap:wrap;">
                    <a class="secondaryButton" href="${img.url}" target="_blank" rel="noopener" style="text-align:center;">Open</a>
                    <button class="primaryButton" data-action="import-image" type="button">Import + save</button>
                  </div>`
                : ""
            }
          </div>
        `
        const importButton = card.querySelector("[data-action=import-image]")
        if (importButton) {
          importButton.addEventListener("click", () => importCivitaiImage(img, importButton))
        }
        grid.appendChild(card)
      })
    }

    function metadataValue(metadata, ...names) {
      if (!metadata || typeof metadata !== "object") return undefined
      const entries = Object.entries(metadata)
      for (const name of names) {
        const wanted = String(name).toLowerCase().replaceAll("_", "").replaceAll(" ", "")
        const match = entries.find(([key]) =>
          String(key).toLowerCase().replaceAll("_", "").replaceAll(" ", "") === wanted
        )
        if (match && match[1] !== null && match[1] !== "") return match[1]
      }
      return undefined
    }

    function civitaiTaskBody(image, localUrl) {
      const metadata = image.meta && typeof image.meta === "object" ? image.meta : {}
      const body = { init_image: localUrl }
      const prompt = image.prompt || metadataValue(metadata, "prompt", "positivePrompt")
      const negativePrompt = metadataValue(metadata, "negativePrompt", "negative_prompt")
      const steps = metadataValue(metadata, "steps", "numInferenceSteps")
      const cfg = metadataValue(metadata, "cfgScale", "guidanceScale", "cfg")
      const seed = metadataValue(metadata, "seed")
      const sampler = metadataValue(metadata, "sampler", "samplerName")
      const scheduler = metadataValue(metadata, "scheduler", "schedulerName")
      const model = metadataValue(metadata, "model", "modelName")
      const size = String(metadataValue(metadata, "size", "resolution") || "").match(/(\d+)\s*[x×]\s*(\d+)/i)

      if (prompt !== undefined) {
        body.prompt = String(prompt)
        body.original_prompt = String(prompt)
      }
      if (negativePrompt !== undefined) body.negative_prompt = String(negativePrompt)
      if (Number.isFinite(Number(steps))) body.num_inference_steps = Number(steps)
      if (Number.isFinite(Number(cfg))) body.guidance_scale = Number(cfg)
      if (Number.isFinite(Number(seed))) body.seed = Number(seed)
      if (sampler !== undefined) body.sampler_name = String(sampler)
      if (scheduler !== undefined) body.scheduler_name = String(scheduler)
      if (model !== undefined) body.use_stable_diffusion_model = String(model)
      if (size) {
        body.width = Number(size[1])
        body.height = Number(size[2])
      } else {
        if (Number.isFinite(Number(image.width))) body.width = Number(image.width)
        if (Number.isFinite(Number(image.height))) body.height = Number(image.height)
      }
      return body
    }

    async function importCivitaiImage(image, button) {
      const previousText = button.textContent
      button.disabled = true
      button.textContent = "Importing…"
      statusEl.textContent = "Saving the CivitAI image into the local gallery…"
      try {
        const response = await fetch(`${API_ROOT}/import-image`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...headersForRequest(),
          },
          body: JSON.stringify({ image }),
        })
        const result = await safeJson(response)
        if (!response.ok || !result.ok) throw new Error(result.detail || result.error || "Import failed")
        const imported = civitaiTaskBody(image, result.editorUrl)
        Object.entries(imported).forEach(([key, value]) => {
          if (typeof TASK_MAPPING !== "undefined" && TASK_MAPPING[key]) TASK_MAPPING[key].setUI(value)
        })
        if (imported.prompt !== undefined) promptField.dispatchEvent(new Event("input", { bubbles: true }))
        statusEl.textContent = `Imported settings and saved ${result.relativePath} to the gallery.`
        selectTab("tab-main")
      } catch (error) {
        console.error(error)
        statusEl.textContent = "Import error: " + (error && error.message ? error.message : "Server Error")
      } finally {
        button.disabled = false
        button.textContent = previousText
      }
    }

    async function handleDownload(item, file) {
      statusEl.textContent = "Starting download..."
      try {
        const provider = item.provider === "huggingface" ? "huggingface" : "civitai"
        const endpoint = provider === "huggingface" ? `${HF_API_ROOT}/download` : DOWNLOAD_ENDPOINT
        const headers = {
          "Content-Type": "application/json",
          ...headersForRequest(undefined, provider),
        }

        const res = await fetch(endpoint, {
          method: "POST",
          headers,
          body: JSON.stringify({ model: item, file }),
        })
        const json = await safeJson(res)
        if (!json.ok) throw new Error(json.error || "Download failed")
        const downloadId = json.downloadId
        if (!downloadId) throw new Error("Download did not return a job ID")
        while (true) {
          await new Promise((resolve) => setTimeout(resolve, 750))
          const statusResponse = await fetch(`${endpoint}/${encodeURIComponent(downloadId)}`, {
            headers: headersForRequest(undefined, provider),
          })
          const job = await safeJson(statusResponse)
          if (!job.ok) throw new Error(job.error || "Download status failed")
          statusEl.textContent = `Downloading… ${Number(job.percent || 0).toFixed(1)}%`
          if (job.status === "failed") throw new Error(job.error || "Download failed")
          if (job.status === "completed") {
            statusEl.textContent = `Downloaded to ${job.path} (${(job.size / (1024 * 1024)).toFixed(1)} MB)`
            break
          }
        }
      } catch (err) {
        console.error(err)
        statusEl.textContent = "Download error: " + (err && err.message ? err.message : "Server Error")
      }
    }

    function captureSearch() {
      applyModeUI()
      return {
        query: (queryEl.value || "").trim(),
        mode: currentMode,
        sort: sortEl.value,
        period: periodEl.value,
        nsfw: nsfwEl.checked,
        provider: currentProvider,
        apiKey: (currentProvider === "huggingface" ? hfTokenEl.value : apiKeyEl.value).trim(),
        postId: (postIdEl.value || "").trim(),
        username: (usernameEl.value || "").trim(),
      }
    }

    function run() {
      // Search is the commit boundary. Draft controls may change without
      // changing the cards or the query used by pagination.
      activeSearch = captureSearch()
      currentPage = 1
      currentCursor = null
      nextCursor = null
      prevCursorStack.length = 0
      prevBtn.disabled = true
      nextBtn.disabled = true
      runSearch(activeSearch, undefined)
    }

    searchBtn.addEventListener("click", run)

    nextBtn.addEventListener("click", () => {
      if (!activeSearch) return
      if (activeSearch.provider === "huggingface") {
        currentPage++
        runSearch(activeSearch, undefined)
        return
      }
      if (activeSearch.mode === "images") {
        currentPage++
        runSearch(activeSearch, undefined)
        return
      }

      // models: move forward using nextCursor
      if (nextCursor) {
        // push current cursor for "Prev"
        prevCursorStack.push(currentCursor) // may be null for page 1
        currentCursor = nextCursor
        currentPage++
        runSearch(activeSearch, currentCursor)
      } else if (!activeSearch.query) {
        currentPage++
        runSearch(activeSearch, undefined)
      }
    })

    prevBtn.addEventListener("click", () => {
      if (!activeSearch) return
      if (activeSearch.provider === "huggingface") {
        currentPage = Math.max(1, currentPage - 1)
        runSearch(activeSearch, undefined)
        return
      }
      if (activeSearch.mode === "images") {
        currentPage = Math.max(1, currentPage - 1)
        runSearch(activeSearch, undefined)
        return
      }

      // models: pop back to previous cursor
      if (prevCursorStack.length > 0) {
        currentCursor = prevCursorStack.pop() || null
        currentPage = Math.max(1, currentPage - 1)
        runSearch(activeSearch, currentCursor)
      } else if (!activeSearch.query && currentPage > 1) {
        currentPage--
        runSearch(activeSearch, undefined)
      }
    })

    // initial mode UI text
    statusEl.textContent = currentProvider === "huggingface"
      ? "Hugging Face model mode."
      : currentMode === "images" ? "Civitai image mode." : "Civitai model mode."
    return true
  }

  const timer = setInterval(() => {
    if (ready()) clearInterval(timer)
  }, 500)
})()
