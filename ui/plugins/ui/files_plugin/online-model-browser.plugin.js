;(function () {
  "use strict"

  const PLUGIN_ID = "civitai-downloader"
  const API_ROOT = "/civitai-api/civitai"
  const SEARCH_ENDPOINT = `${API_ROOT}/search`
  const ENUMS_ENDPOINT = `${API_ROOT}/enums`
  const IMAGES_ENDPOINT = `${API_ROOT}/images`
  const DOWNLOAD_ENDPOINT = `${API_ROOT}/download`
  const HF_API_ROOT = "/huggingface-api"
  const LS_SETTINGS_KEY = "civitaiSettings"
  const LS_ENUMS_KEY = "civitaiEnumCache"
  const LS_IMAGE_TAGS_KEY = "civitaiImageTags"

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
      "Random",
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
    const baseModelEl = panel.querySelector(`#${PLUGIN_ID}-base-model`)
    const modelTypeEl = panel.querySelector(`#${PLUGIN_ID}-model-type`)
    const downloadQueueEl = panel.querySelector(`#${PLUGIN_ID}-queue`)
    const downloadQueueCountEl = panel.querySelector(`#${PLUGIN_ID}-queue-count`)
    const helpEl = panel.querySelector(`#${PLUGIN_ID}-help`)

    const imgFiltersRow = panel.querySelector(`#${PLUGIN_ID}-imgfilters`)
    const postIdEl = panel.querySelector(`#${PLUGIN_ID}-postid`)
    const usernameEl = panel.querySelector(`#${PLUGIN_ID}-username`)
    const imageSortEl = panel.querySelector(`#${PLUGIN_ID}-image-sort`)
    const tagSearchEl = panel.querySelector(`#${PLUGIN_ID}-tag-search`)
    const tagSuggestionsEl = panel.querySelector(`#${PLUGIN_ID}-tag-suggestions`)
    const selectedTagsEl = panel.querySelector(`#${PLUGIN_ID}-selected-tags`)
    const modeEls = panel.querySelectorAll(`input[name="${PLUGIN_ID}-mode"]`)
    const tagCatalog = new Map(), selectedTags = new Map()
    let suggestions = [], activeTagIndex = -1
    let warmingTagCatalog = false

    // Paging state
    let currentPage = 1
    let currentProvider = "civitai"
    let currentMode = "models"
    let activeSearch = null
    let activeSearchController = null
    let searchGeneration = 0
    let downloadQueueRefreshTimer = null

    // Cursor paging (models)
    let currentCursor = null
    let nextCursor = null
    const prevCursorStack = [] // stack of cursors for previous pages (cursor used to fetch that page)

    function applyProviderUI() {
      currentProvider = providerEl.value === "huggingface" ? "huggingface" : "civitai"
      const isHuggingFace = currentProvider === "huggingface"
      const imageRadio = panel.querySelector(`input[name="${PLUGIN_ID}-mode"][value="images"]`)
      imageRadio.disabled = isHuggingFace
      if (isHuggingFace && imageRadio.checked) panel.querySelector(`input[name="${PLUGIN_ID}-mode"][value="models"]`).checked = true
      currentMode = panel.querySelector(`input[name="${PLUGIN_ID}-mode"]:checked`).value
      const imageMode = currentMode === "images"
      imgFiltersRow.style.display = isHuggingFace ? "none" : "flex"
      periodEl.disabled = isHuggingFace
      baseModelEl.disabled = isHuggingFace || imageMode
      modelTypeEl.disabled = isHuggingFace || imageMode
      sortEl.disabled = imageMode
      sortEl.hidden = imageMode
      panel.querySelector(`#${PLUGIN_ID}-model-filters`).style.display = imageMode ? "none" : "flex"
      queryEl.hidden = imageMode
      panel.querySelector(`#${PLUGIN_ID}-queue-panel`).hidden = imageMode
      imageSortEl.querySelector('option[value="Random"]').disabled = !imageMode
      if (!imageMode && imageSortEl.value === "Random") imageSortEl.value = "Newest"
      queryEl.placeholder = isHuggingFace
        ? "Search Hugging Face models or enter owner/repository"
        : "Search term or Civitai URN (blank browses)"
      helpEl.textContent = isHuggingFace
        ? "Browse Hugging Face model repositories and download supported weight files into your configured models folder."
        : imageMode ? "Browse images directly through the image API using selected autocomplete tags. No model cards or custom text queries."
        : "Search models to see their bound images automatically. Model files and each model's image gallery stay separate."
      refreshSortOptions()
    }

    function refreshSortOptions() {
      const key = currentProvider === "huggingface" ? "huggingface" : "models"
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

    function addFilterOptions(select, values) {
      if (!Array.isArray(values)) return
      const existing = new Set(Array.from(select.options, (option) => option.value))
      for (const value of values) {
        if (typeof value !== "string" || !value.trim() || existing.has(value)) continue
        select.add(new Option(value, value))
        existing.add(value)
      }
    }

    async function loadFilterOptions() {
      function applyEnums(enums) {
        if (!enums || !Array.isArray(enums.modelTypes) || !Array.isArray(enums.baseModels)) {
          throw new Error("Invalid Civitai enum response")
        }
        addFilterOptions(modelTypeEl, enums.modelTypes)
        addFilterOptions(baseModelEl, enums.baseModels)
      }
      try {
        const cached = JSON.parse(localStorage.getItem(LS_ENUMS_KEY) || "null")
        if (cached) applyEnums(cached.enums)
      } catch (_) {
        console.warn("Cached Civitai enums unavailable; using built-in options until refresh")
      }
      try {
        // Refresh on every UI load, without discarding usable cached choices.
        const res = await fetch(ENUMS_ENDPOINT, {cache:"no-store"})
        const json = await safeJson(res)
        if (!res.ok || !json.ok) throw new Error("Could not load Civitai filter options")
        applyEnums(json)
        try {
          localStorage.setItem(LS_ENUMS_KEY, JSON.stringify({fetchedAt:Date.now(),enums:json}))
        } catch (_) {
          console.warn("Civitai enums refreshed, but browser storage is unavailable")
        }
      } catch (error) {
        console.warn("Using cached or built-in Civitai filter options", error)
      }
    }

    function validTag(tag) {
      return tag && Number.isSafeInteger(tag.id) && tag.id > 0 && typeof tag.name === "string" && tag.name.trim()
    }

    function rememberImageTags(items) {
      let changed = false
      for (const image of items) {
        for (const tag of Array.isArray(image.tags) ? image.tags : []) {
          if (!validTag(tag)) continue
          if (tagCatalog.get(tag.id)?.name !== tag.name) {
            tagCatalog.set(tag.id, {id:tag.id,name:tag.name})
            changed = true
          }
        }
      }
      if (!changed) return
      for (const id of tagCatalog.keys()) {
        if (tagCatalog.size <= 10000) break
        if (!selectedTags.has(id)) tagCatalog.delete(id)
      }
      try {localStorage.setItem(LS_IMAGE_TAGS_KEY, JSON.stringify([...tagCatalog.values()]))}
      catch (_) {console.warn("Image tag cache could not be saved; current-session suggestions remain available")}
      if (document.activeElement === tagSearchEl) showTagSuggestions()
    }

    async function warmTagCatalog() {
      if (warmingTagCatalog || currentProvider !== "civitai") return
      warmingTagCatalog = true
      try {
        const response = await fetch(`${IMAGES_ENDPOINT}?limit=20&sort=Most%20Reactions&nsfw=false`, {headers:headersForRequest(undefined,"civitai")})
        const result = await safeJson(response)
        if (!response.ok || !result.ok) throw new Error("Image tag discovery unavailable")
        rememberImageTags(result.items || [])
      } catch (_) {
        console.warn("Using cached image tags; browse images to discover additional suggestions")
      } finally {warmingTagCatalog = false}
    }

    function hideTagSuggestions() {
      tagSuggestionsEl.hidden = true
      tagSearchEl.setAttribute("aria-expanded", "false")
      tagSearchEl.removeAttribute("aria-activedescendant")
    }

    function highlightTag(index) {
      activeTagIndex = index
      const options = [...tagSuggestionsEl.querySelectorAll('[role="option"]')]
      options.forEach((option, i) => {
        option.setAttribute("aria-selected", String(i === index))
        option.style.background = i === index ? "var(--background-color3)" : "transparent"
      })
      if (options[index]) tagSearchEl.setAttribute("aria-activedescendant", options[index].id)
      else tagSearchEl.removeAttribute("aria-activedescendant")
    }

    function chooseTag(tag) {
      // Only catalog-backed suggestion objects can create a search filter.
      const known = tagCatalog.get(tag?.id)
      if (!known) return
      selectedTags.set(known.id, known)
      tagSearchEl.value = ""
      hideTagSuggestions()
      renderSelectedTags()
      saveSettings()
    }

    function renderSelectedTags() {
      selectedTagsEl.replaceChildren()
      for (const tag of selectedTags.values()) {
        const chip = document.createElement("button")
        chip.type = "button"
        chip.className = "secondaryButton"
        chip.dataset.selectedTag = tag.id
        chip.setAttribute("aria-label", `Remove ${tag.name}`)
        chip.textContent = `${tag.name} ×`
        chip.addEventListener("click", () => {selectedTags.delete(tag.id);renderSelectedTags();saveSettings()})
        selectedTagsEl.appendChild(chip)
      }
    }

    function showTagSuggestions() {
      const query = tagSearchEl.value.trim().toLowerCase()
      suggestions = [...tagCatalog.values()].filter(tag => !selectedTags.has(tag.id) && tag.name.toLowerCase().includes(query))
        .sort((a,b) => Number(b.name.toLowerCase().startsWith(query)) - Number(a.name.toLowerCase().startsWith(query)) || a.name.localeCompare(b.name))
        .slice(0,20)
      tagSuggestionsEl.replaceChildren()
      for (const [index,tag] of suggestions.entries()) {
        const option = document.createElement("button")
        option.type = "button"
        option.id = `${PLUGIN_ID}-tag-option-${tag.id}`
        option.dataset.tagId = tag.id
        option.setAttribute("role", "option")
        option.tabIndex = -1
        option.style.cssText = "display:block;width:100%;text-align:left;overflow-wrap:anywhere;border:0;padding:8px;"
        option.textContent = `${tag.name} · #${tag.id}`
        option.addEventListener("pointerdown", event => event.preventDefault())
        option.addEventListener("pointermove", () => highlightTag(index))
        option.addEventListener("click", () => chooseTag(tag))
        tagSuggestionsEl.appendChild(option)
      }
      if (!suggestions.length) {
        const empty = document.createElement("small")
        empty.textContent = "No matching cached image tags. Browse Images to discover more; custom terms are not accepted."
        tagSuggestionsEl.appendChild(empty)
      }
      tagSuggestionsEl.hidden = false
      tagSearchEl.setAttribute("aria-expanded", "true")
      highlightTag(suggestions.length ? 0 : -1)
    }

    tagSearchEl.addEventListener("input", showTagSuggestions)
    tagSearchEl.addEventListener("focus", () => {showTagSuggestions();if (!tagCatalog.size) warmTagCatalog()})
    tagSearchEl.addEventListener("keydown", event => {
      if (event.key === "Escape") {hideTagSuggestions();return}
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault()
        if (tagSuggestionsEl.hidden) showTagSuggestions()
        else if (suggestions.length) highlightTag((activeTagIndex + (event.key === "ArrowDown" ? 1 : -1) + suggestions.length) % suggestions.length)
      } else if (event.key === "Enter") {
        event.preventDefault()
        if (!tagSuggestionsEl.hidden && suggestions[activeTagIndex]) chooseTag(suggestions[activeTagIndex])
      }
    })
    document.addEventListener("click", event => {
      if (event.target !== tagSearchEl && !tagSuggestionsEl.contains(event.target)) hideTagSuggestions()
    })

    function saveSettings() {
      const settings = {
        sort: sortEl.value,
        period: periodEl.value,
        nsfw: nsfwEl.checked,
        apikey: apiKeyEl.value.trim(),
        huggingfaceToken: hfTokenEl.value.trim(),
        provider: currentProvider,
        imageSort: imageSortEl.value,
        selectedTags: [...selectedTags.values()],
        mode: currentMode,
        postId: (postIdEl.value || "").trim(),
        username: (usernameEl.value || "").trim(),
        baseModel: baseModelEl.value,
        modelType: modelTypeEl.value,
      }
      localStorage.setItem(LS_SETTINGS_KEY, JSON.stringify(settings))
    }

    function loadSettings() {
      try {
        for (const tag of JSON.parse(localStorage.getItem(LS_IMAGE_TAGS_KEY) || "[]")) {
          if (validTag(tag)) tagCatalog.set(tag.id, {id:tag.id,name:tag.name})
        }
      } catch (_) {console.warn("Saved image tag catalog unavailable")}
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
          for (const tag of Array.isArray(settings.selectedTags) ? settings.selectedTags : []) {
            if (validTag(tag) && tagCatalog.has(tag.id)) selectedTags.set(tag.id, tagCatalog.get(tag.id))
          }
          const mode = settings.mode === "images" ? "images" : "models"
          panel.querySelector(`input[name="${PLUGIN_ID}-mode"][value="${mode}"]`).checked = true
          // Restore selections from live enums even before the next enum fetch.
          for (const [select, value] of [[baseModelEl, settings.baseModel], [modelTypeEl, settings.modelType]]) {
            const savedValue = typeof value === "string" ? value : ""
            addFilterOptions(select, [savedValue])
            select.value = savedValue
          }

          refreshSortOptions()
          const sortKey = currentProvider === "huggingface" ? "huggingface" : "models"
          const allowed = SORT_OPTIONS[sortKey] || []
          sortEl.value = allowed.includes(settings.sort) ? settings.sort : allowed[0]
          const imageSort = settings.imageSort || settings.sort
          imageSortEl.value = SORT_OPTIONS.images.includes(imageSort) ? imageSort : "Newest"
        } catch (e) {
          console.warn("Failed to parse CivitAI settings", e)
        }
      }
      applyProviderUI()
      renderSelectedTags()
    }

    loadSettings()
    loadFilterOptions()
    warmTagCatalog()

    ;[sortEl, imageSortEl, periodEl, baseModelEl, modelTypeEl, nsfwEl, apiKeyEl, hfTokenEl, postIdEl, usernameEl].forEach((el) =>
      el.addEventListener("change", saveSettings)
    )

    providerEl.addEventListener("change", () => {
      applyProviderUI()
      saveSettings()
      resetSearchView()
    })
    modeEls.forEach(el => el.addEventListener("change", () => {
      applyProviderUI()
      saveSettings()
      resetSearchView()
    }))

    function resetSearchView() {
      activeSearchController?.abort()
      searchGeneration++
      resultsEl.innerHTML = ""
      statusEl.textContent = currentProvider === "huggingface" ? "Hugging Face models. Press Search to browse." : `Civitai ${currentMode}. Press Search to browse.`
      activeSearch = null
      currentPage = 1
      pageLabel.textContent = "Page 1"
      prevBtn.disabled = true
      nextBtn.disabled = true
    }

    clearBtn.addEventListener("click", () => {
      apiKeyEl.value = ""
      saveSettings()
    })

    clearHfBtn.addEventListener("click", () => {
      hfTokenEl.value = ""
      saveSettings()
    })

    function parseUrn(raw) {
      // urn:air:anima:diffusionmodel:civitai:MODELID@VERSIONID+FILEID
      const s = (raw || "").trim()
      if (!s.toLowerCase().startsWith("urn:")) return null
      // The file suffix is optional for backwards compatibility.
      const m = s.match(/civitai:(\d+)(?:@(\d+)(?:\+(\d+))?)?$/i)
      if (!m) return null
      return {
        modelId: parseInt(m[1], 10),
        versionId: m[2] ? parseInt(m[2], 10) : null,
        fileId: m[3] ? parseInt(m[3], 10) : null,
      }
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

      // A setting can change while an earlier request is still running. Abort
      // it and ignore stale responses so old cards cannot replace new results.
      activeSearchController?.abort()
      const controller = new AbortController()
      activeSearchController = controller
      const generation = ++searchGeneration

      resultsEl.innerHTML = ""
      prevBtn.disabled = true
      nextBtn.disabled = true
      statusEl.textContent = search.provider === "huggingface"
        ? "Searching Hugging Face models..."
        : "Searching Civitai models and images..."
      if (search.mode === "images") statusEl.textContent = "Searching Civitai images..."
      pageLabel.textContent = `Page ${currentPage}`

      try {
        if (search.mode === "images" && search.provider === "civitai") {
          let url = `${IMAGES_ENDPOINT}?limit=30&nsfw=${Boolean(search.nsfw)}&sort=${encodeURIComponent(search.imageSort)}`
          const cursor = cursorOverride !== undefined ? cursorOverride : currentCursor
          if (cursor) url += `&cursor=${encodeURIComponent(cursor)}`
          if (search.period) url += `&period=${encodeURIComponent(search.period)}`
          if (search.postId) url += `&postId=${encodeURIComponent(search.postId)}`
          if (search.username) url += `&username=${encodeURIComponent(search.username)}`
          if (search.tagIds) url += `&tags=${encodeURIComponent(search.tagIds)}`
          const response = await fetch(url, {headers:headersForRequest(search.apiKey,"civitai"),signal:controller.signal})
          const result = await safeJson(response)
          if (generation !== searchGeneration) return
          if (!response.ok || !result.ok) throw new Error(result.detail || result.error || "Image search failed")
          renderImages(result.items || [], resultsEl)
          nextCursor = extractCursor(result.metadata?.nextCursor || result.metadata?.nextPage) || null
          prevBtn.disabled = !prevCursorStack.length
          nextBtn.disabled = !nextCursor
          statusEl.textContent = `${result.items?.length || 0} images`
          return
        }
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
          renderModels(json.items || [], search, generation, controller)
          prevBtn.disabled = currentPage <= 1
          nextBtn.disabled = !json.metadata?.hasNext
          nextCursor = null
          statusEl.textContent = `${json.items?.length || 0} Hugging Face models`
          return
        }

        const urn = parseUrn(q)
        let url = `${SEARCH_ENDPOINT}?limit=20`
        const isQuerySearch = !urn && (q || "").trim().length > 0
        const useCursor = cursorOverride !== undefined ? cursorOverride : currentCursor
        if (useCursor) url += `&cursor=${encodeURIComponent(useCursor)}`
        else if (!isQuerySearch) url += `&page=${encodeURIComponent(String(currentPage))}`

        url += `&nsfw=${Boolean(search.nsfw)}`
        if (search.sort) url += `&sort=${encodeURIComponent(search.sort)}`
        if (search.period) url += `&period=${encodeURIComponent(search.period)}`
        if (search.baseModel) {
          url += `&baseModels=${encodeURIComponent(search.baseModel)}`
        }
        if (search.modelType) {
          url += `&types=${encodeURIComponent(search.modelType)}`
        }

        // Exact URN versions also scope the associated image gallery.
        if (urn) {
          url += `&modelId=${encodeURIComponent(String(urn.modelId))}`
          if (urn.versionId != null) {
            url += `&modelVersionId=${encodeURIComponent(String(urn.versionId))}`
          }
          if (urn.fileId != null) {
            url += `&fileId=${encodeURIComponent(String(urn.fileId))}`
          }
        } else {
          // normal query (can be blank => browse)
          url += `&query=${encodeURIComponent(q || "")}`
        }

        const requestOptions = { headers: headersForRequest(search.apiKey, search.provider) }
        if (controller) requestOptions.signal = controller.signal
        const res = await fetch(url, requestOptions)
        const json = await safeJson(res)
        if (generation !== searchGeneration) return
        if (!res.ok || !json.ok) throw new Error(json.detail || json.error || "Search failed")

        const galleries = renderModels(json.items || [], search, generation, controller)
        nextCursor = extractCursor(json.metadata?.nextCursor || json.metadata?.nextPage) || null
        const totalPages = json.metadata?.pages || null
        prevBtn.disabled = currentPage <= 1 && prevCursorStack.length === 0
        nextBtn.disabled = isQuerySearch ? !nextCursor
          : nextCursor ? false : (totalPages ? currentPage >= totalPages : (json.items || []).length === 0)
        statusEl.textContent = `${json.items?.length || 0} models · associated images appear below each model.`
        // Limit automatic image discovery to one in-flight request. A new
        // search cancels both the active request and the rest of this queue.
        for (const loadGallery of galleries) {
          if (generation !== searchGeneration || controller.signal.aborted) return
          await loadGallery()
        }
      } catch (err) {
        if (err?.name === "AbortError" || generation !== searchGeneration) return
        console.error(err)
        statusEl.textContent = "Search error: " + (err && err.message ? err.message : "Server Error")
        prevBtn.disabled = true
        nextBtn.disabled = true
      }
    }

    function renderModels(items, search, generation, controller) {
      const galleries = []
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
        card.dataset.modelId = item.id
        card.style =
          "border:1px solid var(--background-color3); border-radius:8px; padding:8px; display:grid; grid-template-columns:76px minmax(0,1fr); gap:8px; margin-bottom:4px;min-width:0;overflow-wrap:anywhere;"
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

            <details data-model-files><summary style="cursor:pointer">Model files (${files.length})</summary>
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
                            <span style="display:flex;gap:4px;align-items:center;">
                              ${
                                f.air
                                  ? `<button class="secondaryButton" style="padding:1px 6px;" data-action="copy-file-air" data-idx="${idx}" title="Copy exact file AIR">AIR</button>`
                                  : ""
                              }
                              <button class="primaryButton" style="padding:1px 6px;" data-action="dl" data-idx="${idx}">Get</button>
                            </span>
                            <span style="font-size:0.75em;color:var(--text-color2);margin-left:6px;">${sizeMB}</span>
                          </div>`
                      })
                      .join("")
                  : `<div style="font-size:0.85em;color:var(--text-color2);">No downloadable files</div>`
              }
            </div></details>
          </div>
        `

        card.querySelectorAll("[data-action=dl]").forEach((b) => {
          b.addEventListener("click", () => {
            const idx = parseInt(b.dataset.idx || "0", 10)
            handleDownload(item, files[idx])
          })
        })

        card.querySelectorAll("[data-action=copy-file-air]").forEach((button) => {
          button.addEventListener("click", async () => {
            const file = files[parseInt(button.dataset.idx || "0", 10)] || {}
            if (!file.air) return
            try {
              await navigator.clipboard.writeText(file.air)
              statusEl.textContent = "Copied exact file AIR to clipboard."
            } catch (_) {
              statusEl.textContent = file.air
            }
          })
        })

        const copyBtn = card.querySelector("[data-action=copy-urn]")
        if (copyBtn && item.id && mv.id) {
          copyBtn.addEventListener("click", async () => {
            const primaryFile = files.find((file) => file.primary) || files[0] || {}
            const urn = primaryFile.air || mv.air
            if (!urn) {
              statusEl.textContent = "Civitai did not provide an AIR for this version."
              return
            }
            try {
              await navigator.clipboard.writeText(urn)
              statusEl.textContent = "Copied URN to clipboard."
            } catch (_) {
              statusEl.textContent = urn
            }
          })
        }

        resultsEl.appendChild(card)
        if (search.provider === "civitai" && item.id) galleries.push(createModelGallery(card, item, search, generation, controller))
      })
      return galleries
    }

    function createModelGallery(card, item, search, generation, controller) {
      const section = document.createElement("section")
      section.style.cssText = "grid-column:1/-1;min-width:0;border-top:1px solid var(--background-color3);padding-top:10px;"
      section.innerHTML = `<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px">
        <strong>Model images</strong><small data-image-status role="status">Waiting to load images…</small>
        <button class="secondaryButton" data-action="image-retry" type="button" hidden>Retry images</button>
        </div><div data-image-gallery></div>
        <nav aria-label="Images for ${escapeHTML(item.name || item.id)}" style="display:flex;justify-content:space-between;gap:8px;margin-top:8px">
          <button class="secondaryButton" data-action="image-prev" type="button" disabled>Previous images</button>
          <button class="secondaryButton" data-action="image-next" type="button" disabled>Next images</button>
        </nav>`
      card.appendChild(section)
      const target = section.querySelector("[data-image-gallery]")
      const status = section.querySelector("[data-image-status]")
      const previous = section.querySelector("[data-action=image-prev]")
      const next = section.querySelector("[data-action=image-next]")
      const retry = section.querySelector("[data-action=image-retry]")
      let state = {cursor:null, previous:[], page:1, next:null}
      let retryPage = state, loading = false
      const stale = () => generation !== searchGeneration || controller.signal.aborted
      async function load(paging) {
        if (loading || stale()) return
        loading = true
        previous.disabled = next.disabled = true
        retry.hidden = true
        status.textContent = "Loading images…"
        try {
          let url = `${IMAGES_ENDPOINT}?limit=6&modelId=${encodeURIComponent(String(item.id))}`
          const urn = parseUrn(search.query)
          if (urn?.modelId === Number(item.id) && urn.versionId) url += `&modelVersionId=${urn.versionId}`
          if (paging.cursor) url += `&cursor=${encodeURIComponent(paging.cursor)}`
          url += `&nsfw=${Boolean(search.nsfw)}&sort=${encodeURIComponent(search.imageSort)}`
          if (search.period) url += `&period=${encodeURIComponent(search.period)}`
          if (search.postId) url += `&postId=${encodeURIComponent(search.postId)}`
          if (search.username) url += `&username=${encodeURIComponent(search.username)}`
          if (search.tagIds) url += `&tags=${encodeURIComponent(search.tagIds)}`
          const response = await fetch(url, {headers:headersForRequest(search.apiKey, "civitai"),signal:controller.signal})
          const result = await safeJson(response)
          if (stale()) return
          if (!response.ok || !result.ok) throw new Error(result.detail || result.error || "Image listing failed")
          renderImages(result.items || [], target)
          state = {...paging, next:extractCursor(result.metadata?.nextCursor || result.metadata?.nextPage) || null}
          status.textContent = `${result.items?.length || 0} images · Page ${state.page}`
        } catch (error) {
          if (error.name === "AbortError" || stale()) return
          status.textContent = `Image listing error: ${error.message || "Server Error"}`
          retryPage = paging
          retry.hidden = false
        } finally {
          loading = false
          if (!stale()) {
            previous.disabled = !state.previous.length
            next.disabled = !state.next
          }
        }
      }
      next.addEventListener("click", () => {
        if (state.next) load({...state, cursor:state.next, previous:[...state.previous,state.cursor], page:state.page + 1})
      })
      previous.addEventListener("click", () => {
        if (!state.previous.length) return
        const stack = [...state.previous]
        load({...state, cursor:stack.pop(), previous:stack, page:state.page - 1})
      })
      retry.addEventListener("click", () => load(retryPage))
      return () => load(state)
    }

    function renderImages(items, target) {
      rememberImageTags(items)
      target.innerHTML = ""
      const grid = document.createElement("div")
      grid.style = "display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%,160px), 1fr)); gap:8px;"
      target.appendChild(grid)
      if (!items.length) {
        const empty = document.createElement("p")
        empty.textContent = "No images returned for this model and the selected filters."
        grid.appendChild(empty)
      }

      items.forEach((img) => {
        const url = safeExternalUrl(img.url || img.thumbnailUrl || "")
        const card = document.createElement("div")
        card.style =
          "border:1px solid var(--background-color3); border-radius:8px; overflow:hidden; background:var(--background-color2); display:flex; flex-direction:column;"
        card.innerHTML = `
          <div style="width:100%; aspect-ratio:1/1; background:var(--background-color3); display:flex; align-items:center; justify-content:center; overflow:hidden;">
            ${
              url
                ? `<img src="${url}" loading="lazy" decoding="async" alt="Civitai image ${escapeHTML(img.id || "")}" style="width:100%;height:100%;object-fit:cover;" />`
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
              url
                ? `<div style="display:flex;gap:4px;flex-wrap:wrap;">
                    <a class="secondaryButton" href="${escapeHTML(url)}" target="_blank" rel="noopener" style="text-align:center;">Open</a>
                    <button class="secondaryButton" data-action="image-details" type="button">Prompt &amp; metadata</button>
                    <button class="primaryButton" data-action="import-image" type="button">Save + apply Civitai settings</button>
                  </div>`
                : ""
            }
          </div>
        `
        const importButton = card.querySelector("[data-action=import-image]")
        card.querySelector("[data-action=image-details]")?.addEventListener("click", () => showImageDetails(img))
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
      const body = localUrl ? { init_image: localUrl } : {}
      const prompt = image.prompt || metadataValue(metadata, "prompt", "positivePrompt")
      const negativePrompt = metadataValue(metadata, "negativePrompt", "negative_prompt")
      const steps = metadataValue(metadata, "steps", "numInferenceSteps", "num_inference_steps")
      const cfg = metadataValue(metadata, "cfgScale", "guidanceScale", "cfg")
      const seed = metadataValue(metadata, "seed")
      const sampler = metadataValue(metadata, "sampler", "samplerName")
      const scheduler = metadataValue(metadata, "scheduler", "schedulerName")
      const model = metadataValue(metadata, "model", "modelName", "use_stable_diffusion_model")
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

    function embeddedSettings(file) {
      const info = file.embedded?.info || {}, exif = file.embedded?.exif || {}
      const comment = exif.UserComment || exif.ExifIFD?.UserComment
      const raw = info.parameters || comment?.text || (typeof comment === "string" ? comment : "")
      let metadata = {...info}
      if (raw) {
        try {
          const parsed = JSON.parse(raw)
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) metadata = parsed
        } catch (_) {
          const marker = raw.lastIndexOf("\nSteps:")
          if (marker >= 0) {
            const prompts = raw.slice(0, marker).split("\nNegative prompt:")
            metadata = {prompt:prompts[0].trim(), negativePrompt:prompts.slice(1).join("\nNegative prompt:").trim()}
            for (const match of raw.slice(marker + 1).matchAll(/(?:^|,\s*)(Steps|Sampler|CFG scale|Seed|Size|Model|Scheduler):\s*([^,]*)/gi)) metadata[match[1]] = match[2].trim()
          }
        }
      }
      // A ComfyUI prompt/workflow graph is not a positive prompt string.
      if (typeof metadata.prompt !== "string" || /^\s*[\[{]/.test(metadata.prompt)) delete metadata.prompt
      return civitaiTaskBody({meta:metadata,width:metadata.width,height:metadata.height})
    }

    function applySettings(body, source) {
      const applied = [], skipped = []
      for (const [key, originalValue] of Object.entries(body)) {
        let value = originalValue
        const select = key === "sampler_name" ? document.getElementById("sampler_name")
          : key === "scheduler_name" ? document.getElementById("scheduler_name") : null
        if (select?.options) {
          const normalize = name => String(name).toLowerCase().replace(/[^a-z0-9+]/g, "")
          const option = [...select.options].find(option => normalize(option.value) === normalize(value) || normalize(option.textContent) === normalize(value))
          if (!option) {skipped.push(key);continue}
          value = option.value
        }
        if (value !== undefined && typeof TASK_MAPPING !== "undefined" && TASK_MAPPING[key]) {
          TASK_MAPPING[key].setUI(value)
          applied.push(key)
        }
      }
      if (applied.includes("prompt") && typeof promptField !== "undefined") promptField.dispatchEvent(new Event("input", {bubbles:true}))
      const result = applied.length ? `Applied ${applied.join(", ")} from ${source}.` : "No recognized settings to apply; copy the report to preserve other fields."
      return result + (skipped.length ? ` Unavailable choices left unchanged: ${skipped.join(", ")}.` : "")
    }

    function showImageDetails(image) {
      const dialog = document.createElement("dialog")
      dialog.style.cssText = "width:min(1050px,calc(100vw - 24px));max-width:calc(100vw - 24px);max-height:90vh;box-sizing:border-box;overflow:auto;"
      dialog.innerHTML = `<div style="display:flex;justify-content:space-between;gap:12px;align-items:center"><h2 style="margin:0;min-width:0;overflow-wrap:anywhere">Image ${escapeHTML(image.id)} · metadata sources</h2><button data-action="close-details" type="button" style="flex-shrink:0">Close</button></div>
        <p>Civitai reports and embedded file metadata may disagree. Each report stays independent. Applying settings is explicit; unknown fields remain in the report.</p>
        <div data-reports style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:16px;align-items:start"></div>
        <p data-report-status role="status" style="overflow-wrap:anywhere"></p>`
      const status = dialog.querySelector("[data-report-status]")
      let inspectionController
      dialog.querySelector("[data-action=close-details]").addEventListener("click", () => dialog.close())
      dialog.addEventListener("close", () => {inspectionController?.abort();dialog.remove()})
      function reportPanel(source, title, report, settings) {
        const section = document.createElement("section")
        section.dataset.reportSource = source
        section.style.minWidth = "0"
        section.innerHTML = `<h3>${title}</h3><textarea aria-label="${title}" readonly rows="16" spellcheck="false" style="width:100%;resize:vertical;box-sizing:border-box;font:12px/1.5 monospace"></textarea><div data-report-actions style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px"><button data-action="copy-report" type="button">Copy report</button><button data-action="copy-prompt" type="button">Copy prompt</button><button data-action="copy-negative" type="button">Copy negative</button><button data-action="apply-report" type="button">Apply settings</button></div>`
        const field = section.querySelector("textarea")
        let body = settings
        async function copy(value) {
          try {await navigator.clipboard.writeText(value);status.textContent = `Copied ${title}.`}
          catch (_) {field.focus();field.select();status.textContent = "Clipboard unavailable. The report is selected for manual copying."}
        }
        section.querySelector("[data-action=copy-report]").addEventListener("click", () => copy(field.value))
        section.querySelector("[data-action=copy-prompt]").addEventListener("click", () => copy(String(body.prompt ?? "")))
        section.querySelector("[data-action=copy-negative]").addEventListener("click", () => copy(String(body.negative_prompt ?? "")))
        section.querySelector("[data-action=apply-report]").addEventListener("click", () => {status.textContent = applySettings(body, title)})
        function update(value, newSettings) {
          body = newSettings
          field.value = typeof value === "string" ? value : JSON.stringify(value, null, 2)
          section.querySelector("[data-action=copy-prompt]").disabled = body.prompt === undefined
          section.querySelector("[data-action=copy-negative]").disabled = body.negative_prompt === undefined
          section.querySelector("[data-action=apply-report]").disabled = !Object.keys(body).length
        }
        update(report, settings)
        dialog.querySelector("[data-reports]").appendChild(section)
        return {section, update}
      }
      reportPanel("civitai", "Civitai-reported metadata", image, civitaiTaskBody(image))
      const file = reportPanel("file", "Downloaded-file metadata", "Not read yet. Inspect the delivered image bytes; Civitai may deliver a resized or metadata-stripped copy.", {})
      const inspect = document.createElement("button")
      inspect.type = "button"
      inspect.dataset.action = "inspect-file"
      inspect.textContent = "Inspect downloaded file"
      inspect.style.marginTop = "10px"
      file.section.appendChild(inspect)
      inspect.addEventListener("click", async () => {
        inspect.disabled = true
        status.textContent = "Reading the delivered image file (up to 100 MB); not saving it to the gallery…"
        inspectionController = new AbortController()
        try {
          const response = await fetch(`${API_ROOT}/image-metadata`, {method:"POST",headers:{"Content-Type":"application/json",...headersForRequest(undefined,"civitai")},body:JSON.stringify({url:image.url}),signal:inspectionController.signal})
          const result = await safeJson(response)
          if (!response.ok || !result.ok) throw new Error(result.detail || result.error || "Metadata inspection failed")
          file.update(result.file, embeddedSettings(result.file))
          status.textContent = `Read ${result.file.sizeBytes} bytes · ${result.file.format} · ${result.file.width}×${result.file.height}. File metadata is not replaced by Civitai metadata; empty sections mean no metadata was found there.`
          inspect.textContent = "Read file again"
        } catch (error) {
          if (error.name !== "AbortError") status.textContent = `File inspection error: ${error.message}`
        } finally {inspect.disabled = false}
      })
      document.body.appendChild(dialog)
      dialog.showModal()
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
        const applied = applySettings(imported, "Civitai-reported metadata")
        statusEl.textContent = `Saved ${result.relativePath} to the gallery. ${applied}`
        selectTab("tab-main")
      } catch (error) {
        console.error(error)
        statusEl.textContent = "Import error: " + (error && error.message ? error.message : "Server Error")
      } finally {
        button.disabled = false
        button.textContent = previousText
      }
    }

    const ACTIVE_DOWNLOAD_STATUSES = new Set(["queued", "preparing", "downloading"])

    function renderDownloadQueue(jobs) {
      const active = jobs.filter((job) => ACTIVE_DOWNLOAD_STATUSES.has(job.status))
      const recent = jobs.filter((job) => !ACTIVE_DOWNLOAD_STATUSES.has(job.status)).slice(0, 3)
      const visible = [...active, ...recent]
      downloadQueueCountEl.textContent = active.length ? `(${active.length} active)` : ""
      if (!visible.length) {
        downloadQueueEl.innerHTML = "<small>No downloads yet.</small>"
        return
      }
      downloadQueueEl.innerHTML = visible.map((job) => {
        const provider = job.provider === "huggingface" ? "Hugging Face" : "Civitai"
        let state = job.status || "unknown"
        if (job.status === "queued") state = `queued #${job.queuePosition || "?"}`
        if (job.status === "preparing") state = "preparing"
        if (job.status === "downloading") state = `${Number(job.percent || 0).toFixed(1)}%`
        if (job.status === "completed") state = "complete"
        if (job.status === "failed") state = `failed: ${job.error || "unknown error"}`
        return `<div style="display:flex;justify-content:space-between;gap:8px;font-size:0.8em;background:var(--background-color2);padding:3px 6px;border-radius:4px;">
          <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHTML(job.filename || job.downloadId)}">${escapeHTML(job.filename || "Model download")}</span>
          <span style="white-space:nowrap;color:var(--text-color2);">${provider} • ${escapeHTML(state)}</span>
        </div>`
      }).join("")
    }

    async function fetchDownloadQueue(endpoint, provider) {
      const response = await fetch(endpoint, { headers: headersForRequest(undefined, provider) })
      const result = await safeJson(response)
      if (!response.ok || !result.ok) throw new Error(result.detail || result.error || "Queue status failed")
      return result.jobs || []
    }

    async function refreshDownloadQueue() {
      let hasActive = false
      try {
        const results = await Promise.allSettled([
          fetchDownloadQueue(DOWNLOAD_ENDPOINT, "civitai"),
          fetchDownloadQueue(`${HF_API_ROOT}/download`, "huggingface"),
        ])
        const jobs = results.flatMap((result) => result.status === "fulfilled" ? result.value : [])
          .sort((left, right) => Number(right.createdAt || 0) - Number(left.createdAt || 0))
        hasActive = jobs.some((job) => ACTIVE_DOWNLOAD_STATUSES.has(job.status))
        renderDownloadQueue(jobs)
      } catch (error) {
        console.warn("Unable to refresh model download queue", error)
      } finally {
        if (downloadQueueRefreshTimer) clearTimeout(downloadQueueRefreshTimer)
        downloadQueueRefreshTimer = setTimeout(refreshDownloadQueue, hasActive ? 750 : 5000)
      }
    }

    async function handleDownload(item, file) {
      statusEl.textContent = "Adding download to queue..."
      try {
        const provider = item.provider === "huggingface" ? "huggingface" : "civitai"
        const endpoint = provider === "huggingface" ? `${HF_API_ROOT}/download` : DOWNLOAD_ENDPOINT
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...headersForRequest(undefined, provider),
          },
          body: JSON.stringify({ model: item, file }),
        })
        const job = await safeJson(response)
        if (!response.ok || !job.ok) throw new Error(job.detail || job.error || "Download failed")
        if (!job.downloadId) throw new Error("Download did not return a job ID")
        statusEl.textContent = job.status === "queued"
          ? `Queued ${job.filename || file.name || "model"} at position ${job.queuePosition || 1}.`
          : `Started ${job.filename || file.name || "model"}.`
        await refreshDownloadQueue()
      } catch (err) {
        console.error(err)
        statusEl.textContent = "Download error: " + (err && err.message ? err.message : "Server Error")
      }
    }

    function captureSearch() {
      if (currentProvider === "civitai" && tagSearchEl.value.trim()) {
        throw new Error("Select a tag from autocomplete or clear the unselected text before searching.")
      }
      return {
        query: currentMode === "images" ? "" : (queryEl.value || "").trim(),
        mode: currentMode,
        imageSort: imageSortEl.value,
        tagIds: [...selectedTags.keys()].join(","),
        sort: sortEl.value,
        period: periodEl.value,
        nsfw: nsfwEl.checked,
        provider: currentProvider,
        apiKey: (currentProvider === "huggingface" ? hfTokenEl.value : apiKeyEl.value).trim(),
        postId: (postIdEl.value || "").trim(),
        username: (usernameEl.value || "").trim(),
        baseModel: baseModelEl.value,
        modelType: modelTypeEl.value,
      }
    }

    function run() {
      // Search is the commit boundary. Draft controls may change without
      // changing the cards or the query used by pagination.
      try {activeSearch = captureSearch()}
      catch (error) {statusEl.textContent = `Search error: ${error.message}`;return}
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
      // Civitai cursors avoid repeated results from page-number paging, and
      // the API returns them for both model and image feeds.
      if (nextCursor) {
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
      // Return using the cursor saved when advancing, or page numbers when
      // the upstream response did not provide a cursor.
      if (prevCursorStack.length > 0) {
        currentCursor = prevCursorStack.pop() || null
        currentPage = Math.max(1, currentPage - 1)
        runSearch(activeSearch, currentCursor)
      } else if (!activeSearch.query && currentPage > 1) {
        currentPage--
        runSearch(activeSearch, undefined)
      }
    })

    statusEl.textContent = currentProvider === "huggingface"
      ? "Hugging Face models."
      : currentMode === "images" ? "Browse Civitai images using selected tags." : "Search Civitai models to see their bound images and downloadable files."
    refreshDownloadQueue()
    return true
  }

  const timer = setInterval(() => {
    if (ready()) clearInterval(timer)
  }, 500)
})()
