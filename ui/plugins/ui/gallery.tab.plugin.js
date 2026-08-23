(function () {

    "use strict"

    // Check if plugin is already running
    if (document.querySelector("#gallery-tab") !== null) {
        return
    }

    // Selection state
    const selectedImages = new Set()

    // Lightbox state
    let lightboxOpen = false
    let currentLightboxIndex = 0
    let lightboxImages = []

    // Add styles
    var styleSheet = document.createElement("style")
    styleSheet.textContent = `
        #gallery-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 8px;
            align-items: flex-start;
        }

        #gallery-container img {
            margin: 0;
            display: block;
            max-width: none;
        }

        .gallery-image-wrapper {
            flex: 0 0 auto;
            display: flex;
            align-items: flex-start;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }

        .gallery-image-wrapper img {
            height: auto;
            width: auto;
            border-radius: 8px;
            cursor: pointer;
        }

        /* Checkbox Styles */
        .gallery-checkbox {
            position: absolute;
            top: 5px;
            left: 5px;
            width: 26px;
            height: 26px;
            cursor: pointer;
            z-index: 10;
            opacity: 0;
            appearance: none;
            -webkit-appearance: none;
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid var(--text-color);
            border-radius: 50%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .gallery-checkbox::after {
            content: '✓';
            color: transparent;
            font-size: 18px;
            font-weight: bold;
        }

        .gallery-checkbox:checked {
            background: #4CAF50;
            border-color: white;
        }

        .gallery-checkbox:checked::after {
            color: white;
        }

        .gallery-checkbox:hover {
            background: rgba(76, 175, 80, 0.7);
            border-color: white;
        }

        .gallery-image-wrapper:hover .gallery-checkbox {
            opacity: 1;
        }

        .gallery-image-wrapper.selected {
            outline: 3px solid #4CAF50;
            outline-offset: -3px;
        }

        .gallery-image-wrapper.selected .gallery-checkbox {
            opacity: 1;
        }

        /* Delete Button Styles */
        .gallery-delete-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(0, 0, 0, 0.5);
            color: #ff4444;
            border: 2px solid var(--text-color);
            border-radius: 50%;
            width: 26px;
            height: 26px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: bold;
            opacity: 0;
            z-index: 10;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        
        .gallery-delete-btn:hover {
            background: #ff4444;
            color: white;
            border-color: white;
        }

        /* Download Button Styles */
        .gallery-download-btn {
            position: absolute;
            bottom: 8px;
            right: 8px;
            background: rgba(0, 0, 0, 0.5);
            color: #4444ff;
            border: 2px solid var(--text-color);
            border-radius: 50%;
            width: 26px;
            height: 26px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: bold;
            opacity: 0;
            z-index: 10;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        
        .gallery-download-btn:hover {
            background: #4444ff;
            color: white;
            border-color: white;
        }

        .gallery-image-wrapper:hover .gallery-delete-btn,
        .gallery-image-wrapper:hover .gallery-download-btn {
            opacity: 1;
        }

        #gallery-empty {
            text-align: center;
            padding: 40px;
            color: rgb(153, 153, 153);
            font-style: italic;
        }

        #gallery-controls {
            padding: 4px;
            display: flex;
            align-items: center;
            background: var(--background-color2);
            margin: 8px 16px;
            border-radius: 4px;
            gap: 4px;
            flex-wrap: wrap;
            position: sticky;
            top: 16px;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        }

        #gallery-zoom-control {
            margin-left: auto;
            margin-right: 0;
        }

        .gallery-action-btn {
            padding: 6px 12px;
            border-radius: 4px;
            border: none;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
        }

        .gallery-action-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        #gallery-selection-count {
            font-size: 13px;
            color: rgb(153, 153, 153);
            margin-right: 8px;
        }

/* Lightbox Styles */
#gallery-lightbox {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.75);
    z-index: 10000;
    display: none;
    align-items: center;
    justify-content: center;
}

#gallery-lightbox.active {
    display: flex;
}

#gallery-lightbox-content {
    position: relative;
    width: 100vw;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

#gallery-lightbox-image {
    width: auto;
    height: auto;
    object-fit: contain;
    user-select: none;
    cursor: pointer;
    max-width: none;
    max-height: none;
}

.gallery-lightbox-btn {
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: bold;
    background: none;
    border: none;
    padding: 0;
    margin: 0 4px;
}

.gallery-lightbox-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
}

#gallery-lightbox-counter {
    font-weight: bold;
    position: fixed;
    top: 8px;
    left: 50%;
    transform: translateX(-50%);
    color: white;
    background: rgba(0, 0, 0, 0.5);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    z-index: 10011;  /* Match button z-index */
    border: 2px solid var(--text-color);
    box-shadow: 0 1px 3px rgba(0,0,0,0.5);
}

#gallery-lightbox-controls {
    position: fixed;
    bottom: 8px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 8px;
    align-items: center;
    background: rgba(0, 0, 0, 0.5);
    padding: 0px 8px;
    border-radius: 13px;
    border: 2px solid var(--text-color);
    box-shadow: 0 1px 3px rgba(0,0,0,0.5);
    z-index: 10011;
}

#gallery-delete-confirm {
    position: fixed;
    inset: 0;
    z-index: 12000;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 16px;
    background: rgba(0, 0, 0, 0.78);
}

#gallery-delete-confirm.active {
    display: flex;
}

#gallery-delete-confirm-dialog {
    box-sizing: border-box;
    width: min(460px, 100%);
    max-height: calc(100vh - 32px);
    overflow-y: auto;
    padding: 18px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 10px;
    background: var(--background-color2, #242526);
    color: var(--text-color, white);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.55);
}

#gallery-delete-confirm-message {
    margin: 12px 0 18px;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}

#gallery-delete-confirm-actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 10px;
}

#gallery-delete-confirm-actions button {
    min-height: 42px;
    padding: 8px 14px;
}

#gallery-delete-confirm-continue {
    background: #b3261e;
    color: white;
}

@media screen and (max-width: 700px) {
    #gallery-controls {
        position: static;
        margin: 4px;
        padding: 6px;
    }

    #gallery-controls button {
        flex: 1 1 calc(50% - 6px);
        min-height: 40px;
    }

    #gallery-zoom-control {
        width: 100%;
        margin-left: 0;
    }

    #gallery-zoom-slider {
        width: 100%;
    }

    #gallery-container {
        box-sizing: border-box;
        width: 100%;
        padding: 4px;
        gap: 4px;
    }

    .gallery-image-wrapper {
        max-width: 100%;
    }

    .gallery-image-wrapper img {
        max-width: calc(100vw - 16px) !important;
        height: auto !important;
    }

    .gallery-checkbox,
    .gallery-delete-btn,
    .gallery-download-btn {
        opacity: 1;
    }

    .gallery-checkbox,
    .gallery-delete-btn,
    .gallery-download-btn {
        width: 36px;
        height: 36px;
        min-height: 36px;
    }

    #gallery-delete-selected:not(:disabled) {
        background: #b3261e;
        color: white;
    }

    #gallery-delete-confirm-actions button {
        flex: 1 1 100%;
        width: 100%;
    }
}

@media (hover: none), (pointer: coarse) {
    .gallery-checkbox,
    .gallery-delete-btn,
    .gallery-download-btn {
        opacity: 1;
    }
}
    `
    document.head.appendChild(styleSheet)

    // Add gallery tab
    document.querySelector('.tab-container')?.insertAdjacentHTML('beforeend', `
        <span id="tab-gallery" class="tab">
            <span><i class="fa fa-images icon"></i> Gallery</span>
        </span>
    `)

    // Add gallery content area
    document.querySelector('#tab-content-wrapper')?.insertAdjacentHTML('beforeend', `
        <div id="tab-content-gallery" class="tab-content">
            <div id="gallery">
                <div id="gallery-controls">
                    <span style="color: rgb(96, 96, 96);">
                        Total <span id="gallery-count">0</span> images
                    </span>
                    <button id="gallery-select-all" class="tertiaryButton" style="white-space: nowrap;">
                        Select All
                    </button>
                    <button id="gallery-deselect-all" class="tertiaryButton" style="white-space: nowrap;">
                        Deselect All
                    </button>
                    <button id="gallery-refresh" class="tertiaryButton" style="white-space: nowrap;">
                        Refresh Directory
                    </button>
                    <button id="gallery-prev-page" class="tertiaryButton" disabled style="white-space: nowrap;">
                        ← Previous
                    </button>
                    <span id="gallery-page-status" style="white-space: nowrap;">Page 1 of 1</span>
                    <button id="gallery-next-page" class="tertiaryButton" disabled style="white-space: nowrap;">
                        Next →
                    </button>
                    <button id="gallery-collage-horizontal" class="primaryButton" disabled style="white-space: nowrap;">
                        Horizontal Collage
                    </button>
                    <button id="gallery-collage-vertical" class="primaryButton" disabled style="white-space: nowrap;">
                        Vertical Collage
                    </button>
                    <button id="gallery-collage-grid" class="primaryButton" disabled style="white-space: nowrap;">
                        Grid Collage
                    </button>
                    <button id="gallery-delete-selected" class="tertiaryButton" disabled style="white-space: nowrap;">
                        Delete Selected
                    </button>
                    <span id="gallery-selection-count" style="display: none;">
                        Selected: <span id="gallery-selected-count">0</span>
                    </span>
                    <div id="gallery-zoom-control">
                        <input type="range" id="gallery-zoom-slider" min="25" max="200" value="100" step="1">
                    </div>
                    <span id="gallery-source" style="width: 100%; color: rgb(153, 153, 153); overflow-wrap: anywhere;">
                        Loading configured gallery directory...
                    </span>
                </div>
                <div id="gallery-container"></div>
                <div id="gallery-empty">No supported images were found in the configured gallery directory.</div>
            </div>
        </div>
    `)

    // Add the gallery's own directory setting to the Settings page.
    const systemSettingsTable = document.querySelector('#system-settings-table')
    if (systemSettingsTable && !document.querySelector('#gallery-directory-setting')) {
        const settingRow = document.createElement('div')
        settingRow.id = 'gallery-directory-setting'
        settingRow.innerHTML = `
            <div><i class="fa fa-images"></i></div>
            <div>
                <label for="gallery-directory-input">Gallery Image Directory</label>
                <small>Gallery scans this server directory and its subdirectories for images.</small>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center;">
                <input id="gallery-directory-input" type="text" style="flex: 1 1 260px; min-width: 0;"
                    placeholder="/path/to/image/directory">
                <button id="gallery-directory-save" type="button" class="primaryButton">Apply</button>
                <small id="gallery-directory-status" style="width: 100%; overflow-wrap: anywhere;"></small>
            </div>
        `
        systemSettingsTable.appendChild(settingRow)
    }

    // Add lightbox HTML
    document.body.insertAdjacentHTML('beforeend', `
        <div id="gallery-lightbox">
            <div id="gallery-lightbox-content">
                <img id="gallery-lightbox-image" src="" alt="Full size image">
                <div id="gallery-lightbox-counter"></div>
                <div id="gallery-lightbox-controls">
                    <button class="gallery-lightbox-btn" id="gallery-lightbox-prev" title="Previous (Left Arrow)">⇠</button>
                    <button class="gallery-lightbox-btn" id="gallery-lightbox-close" title="Close (Escape)"> × </button>
                    <button class="gallery-lightbox-btn" id="gallery-lightbox-next" title="Next (Right Arrow)">⇢</button>

                </div>
            </div>
        </div>
    `)

    // Use an in-page confirmation so older mobile browsers always show both actions.
    document.body.insertAdjacentHTML('beforeend', `
        <div id="gallery-delete-confirm" role="dialog" aria-modal="true"
            aria-labelledby="gallery-delete-confirm-title">
            <div id="gallery-delete-confirm-dialog">
                <h3 id="gallery-delete-confirm-title">Confirm file deletion</h3>
                <div id="gallery-delete-confirm-message"></div>
                <div id="gallery-delete-confirm-actions">
                    <button id="gallery-delete-confirm-cancel" type="button" class="tertiaryButton">
                        Cancel
                    </button>
                    <button id="gallery-delete-confirm-continue" type="button" class="primaryButton">
                        Continue deletion
                    </button>
                </div>
            </div>
        </div>
    `)

    // Cache DOM nodes used repeatedly
    const galleryContainer = document.querySelector('#gallery-container')
    const galleryEmpty = document.querySelector('#gallery-empty')
    const galleryCountSpan = document.querySelector('#gallery-count')
    const selectionCountSpan = document.querySelector('#gallery-selected-count')
    const selectionCountContainer = document.querySelector('#gallery-selection-count')
    const collageHorizontalBtn = document.querySelector('#gallery-collage-horizontal')
    const collageVerticalBtn = document.querySelector('#gallery-collage-vertical')
    const collageGridBtn = document.querySelector('#gallery-collage-grid')
    const deleteSelectedBtn = document.querySelector('#gallery-delete-selected')
    const zoomSlider = document.querySelector('#gallery-zoom-slider')
    const selectAllBtn = document.querySelector('#gallery-select-all')
    const deselectAllBtn = document.querySelector('#gallery-deselect-all')
    const refreshBtn = document.querySelector('#gallery-refresh')
    const previousPageBtn = document.querySelector('#gallery-prev-page')
    const nextPageBtn = document.querySelector('#gallery-next-page')
    const pageStatus = document.querySelector('#gallery-page-status')
    const gallerySource = document.querySelector('#gallery-source')
    const galleryDirectoryInput = document.querySelector('#gallery-directory-input')
    const galleryDirectorySave = document.querySelector('#gallery-directory-save')
    const galleryDirectoryStatus = document.querySelector('#gallery-directory-status')

    // Lightbox elements
    const lightbox = document.querySelector('#gallery-lightbox')
    const lightboxImage = document.querySelector('#gallery-lightbox-image')
    const lightboxCounter = document.querySelector('#gallery-lightbox-counter')
    const lightboxPrev = document.querySelector('#gallery-lightbox-prev')
    const lightboxNext = document.querySelector('#gallery-lightbox-next')
    const lightboxClose = document.querySelector('#gallery-lightbox-close')
    const lightboxContent = document.querySelector('#gallery-lightbox-content')
    const lightboxControls = document.querySelector('#gallery-lightbox-controls')
    const deleteConfirm = document.querySelector('#gallery-delete-confirm')
    const deleteConfirmMessage = document.querySelector('#gallery-delete-confirm-message')
    const deleteConfirmCancel = document.querySelector('#gallery-delete-confirm-cancel')
    const deleteConfirmContinue = document.querySelector('#gallery-delete-confirm-continue')

    // Link the tab
    const tabGallery = document.querySelector('#tab-gallery')
    if (tabGallery && typeof linkTabContents === 'function') {
        linkTabContents(tabGallery)
    }

    // Gallery state and constants
    let imageObserver = null
    let updateTimeout = null
    let isDeletingImage = false
    let refreshIntervalId = null
    let lastGalleryIds = new Set()
    let galleryTotal = 0
    let currentPage = 1
    let totalPages = 1
    const galleryPageSize = 60

    // Filesystem-backed gallery data from the directory configured on Settings.
    const imageCache = new Map()
    let deleteConfirmationResolver = null

    function finishDeleteConfirmation(confirmed) {
        if (deleteConfirm) {
            deleteConfirm.classList.remove('active')
        }
        const resolve = deleteConfirmationResolver
        deleteConfirmationResolver = null
        if (resolve) resolve(confirmed)
    }

    function confirmGalleryDeletion(message) {
        if (
            !deleteConfirm ||
            !deleteConfirmMessage ||
            !deleteConfirmCancel ||
            !deleteConfirmContinue
        ) {
            return Promise.resolve(false)
        }
        if (deleteConfirmationResolver) {
            return Promise.resolve(false)
        }

        deleteConfirmMessage.textContent = message
        deleteConfirm.classList.add('active')
        setTimeout(() => deleteConfirmContinue.focus(), 0)
        return new Promise((resolve) => {
            deleteConfirmationResolver = resolve
        })
    }

    deleteConfirmCancel?.addEventListener('click', () => {
        finishDeleteConfirmation(false)
    })
    deleteConfirmContinue?.addEventListener('click', () => {
        finishDeleteConfirmation(true)
    })
    deleteConfirm?.addEventListener('click', (event) => {
        if (event.target === deleteConfirm) {
            finishDeleteConfirmation(false)
        }
    })
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && deleteConfirmationResolver) {
            finishDeleteConfirmation(false)
        }
    })

    async function collectImages() {
        const query = new URLSearchParams({
            page: String(currentPage),
            page_size: String(galleryPageSize)
        })
        const response = await fetch(`/gallery-plugin/images?${query}`, { cache: 'no-store' })
        if (!response.ok) {
            const text = await response.text()
            throw new Error(`Gallery directory request failed (${response.status}): ${text}`)
        }
        const data = await response.json()
        const records = Array.isArray(data.images) ? data.images : []
        galleryTotal = Number(data.total || records.length)
        currentPage = Number(data.page || 1)
        totalPages = Number(data.total_pages || 1)

        if (pageStatus) {
            pageStatus.textContent = `Page ${currentPage} of ${totalPages}`
        }
        if (previousPageBtn) {
            previousPageBtn.disabled = !data.has_previous
        }
        if (nextPageBtn) {
            nextPageBtn.disabled = !data.has_next
        }

        if (gallerySource) {
            const range = galleryTotal
                ? `images ${data.start_index}–${data.end_index} of ${galleryTotal}`
                : '0 images'
            gallerySource.textContent = `${data.directory || 'Gallery directory'} — ${range}`
        }
        if (galleryDirectoryInput && !galleryDirectoryInput.value) {
            galleryDirectoryInput.value = data.directory || ''
        }

        imageCache.clear()
        const images = records.map((record) => {
            const imgData = {
                src: record.url,
                thumbnailSrc: record.thumbnail_url || record.url,
                taskId: 'gallery-directory',
                uniqueId: record.id,
                element: null,
                timestamp: Number(record.mtime || 0),
                relativePath: record.relative_path,
                filename: record.filename
            }
            imageCache.set(imgData.uniqueId, imgData)
            return imgData
        })
        return images
    }

    function setDirectoryStatus(message, isError = false) {
        if (!galleryDirectoryStatus) return
        galleryDirectoryStatus.textContent = message
        galleryDirectoryStatus.style.color = isError ? 'rgb(255, 120, 120)' : ''
    }

    async function loadGallerySettings() {
        if (!galleryDirectoryInput) return
        setDirectoryStatus('Loading gallery setting...')
        try {
            const response = await fetch('/gallery-plugin/settings', { cache: 'no-store' })
            if (!response.ok) {
                throw new Error(`Settings request failed (${response.status})`)
            }
            const settings = await response.json()
            galleryDirectoryInput.value = settings.gallery_directory || ''
            setDirectoryStatus(
                settings.exists
                    ? 'Directory is available.'
                    : 'Directory does not exist.',
                !settings.exists
            )
        } catch (error) {
            setDirectoryStatus(error.message, true)
        }
    }

    async function saveGallerySettings() {
        if (!galleryDirectoryInput || !galleryDirectorySave) return
        const directory = galleryDirectoryInput.value.trim()
        galleryDirectorySave.disabled = true
        setDirectoryStatus('Saving and scanning directory...')
        try {
            const response = await fetch('/gallery-plugin/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ gallery_directory: directory })
            })
            const result = await response.json()
            if (!response.ok) {
                throw new Error(result.detail || `Could not save directory (${response.status})`)
            }
            galleryDirectoryInput.value = result.gallery_directory
            setDirectoryStatus('Gallery directory saved.')
            selectedImages.clear()
            imageCache.clear()
            lastGalleryIds.clear()
            currentPage = 1
            await updateGallery()
        } catch (error) {
            setDirectoryStatus(error.message, true)
        } finally {
            galleryDirectorySave.disabled = false
        }
    }

    galleryDirectorySave?.addEventListener('click', saveGallerySettings)
    galleryDirectoryInput?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault()
            saveGallerySettings()
        }
    })
    refreshBtn?.addEventListener('click', () => {
        imageCache.clear()
        updateGallery()
    })

    async function changeGalleryPage(page) {
        if (page < 1 || page > totalPages || page === currentPage) return
        currentPage = page
        selectedImages.clear()
        imageCache.clear()
        lastGalleryIds.clear()
        updateSelectionUI()
        await updateGallery()
        document.querySelector('#gallery-controls')?.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        })
    }

    previousPageBtn?.addEventListener('click', () => {
        changeGalleryPage(currentPage - 1)
    })
    nextPageBtn?.addEventListener('click', () => {
        changeGalleryPage(currentPage + 1)
    })

    // Function to update selection UI
    function updateSelectionUI() {
        const count = selectedImages.size

        if (selectionCountSpan) selectionCountSpan.textContent = count
        if (selectionCountContainer) selectionCountContainer.style.display = count > 0 ? 'inline' : 'none'

        const enabled = count > 0
        if (collageHorizontalBtn) collageHorizontalBtn.disabled = !enabled
        if (collageVerticalBtn) collageVerticalBtn.disabled = !enabled
        if (collageGridBtn) collageGridBtn.disabled = !enabled
        if (deleteSelectedBtn) deleteSelectedBtn.disabled = !enabled
    }

    // Function to download an image
    function downloadImage(imgSrc, imgData) {
        // Try to find the original download button/link in the source element
        if (imgData.element) {
            const downloadLink = imgData.element.querySelector('a[download]') ||
                imgData.element.querySelector('[title*="Download"]')

            if (downloadLink) {
                downloadLink.click()
                return
            }
        }

        // Fallback: manual download
        fetch(imgSrc)
            .then(response => response.blob())
            .then(blob => {
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url

                // Try to extract filename from src or create a timestamp-based name
                const filename = imgData.filename ||
                    decodeURIComponent(imgSrc.split('/').pop()) ||
                    `image_${Date.now()}.png`
                a.download = filename

                document.body.appendChild(a)
                a.click()
                document.body.removeChild(a)
                URL.revokeObjectURL(url)
            })
            .catch(err => {
                console.error('Download failed:', err)
            })
    }

    // Lightbox Functions
    function openLightbox(index) {
        if (lightboxImages.length === 0) return

        currentLightboxIndex = index
        lightboxOpen = true

        updateLightboxImage()

        if (lightbox) {
            lightbox.classList.add('active')
        }

        // Prevent body scrolling
        document.body.style.overflow = 'hidden'
    }

    function closeLightbox() {
        lightboxOpen = false

        if (lightbox) {
            lightbox.classList.remove('active')
        }

        // Restore body scrolling
        document.body.style.overflow = ''
    }

    function navigateLightbox(direction) {
        if (lightboxImages.length === 0) return

        currentLightboxIndex += direction

        // Loop around
        if (currentLightboxIndex < 0) {
            currentLightboxIndex = lightboxImages.length - 1
        } else if (currentLightboxIndex >= lightboxImages.length) {
            currentLightboxIndex = 0
        }

        updateLightboxImage()
    }

    function updateLightboxImage() {
        if (!lightboxImage || !lightboxCounter) return
        if (currentLightboxIndex < 0 || currentLightboxIndex >= lightboxImages.length) return

        const imgData = lightboxImages[currentLightboxIndex]
        lightboxImage.src = imgData.src
        
        // Reverse the counter: show latest as total, oldest as 1
        const reversedIndex = lightboxImages.length - currentLightboxIndex
        lightboxCounter.textContent = `${reversedIndex} / ${lightboxImages.length}`
        
        // Load the image to get its dimensions for proper sizing
        const tempImg = new Image()
        tempImg.onload = function() {
            const imageAspectRatio = this.width / this.height
            const viewportAspectRatio = window.innerWidth / window.innerHeight
            
            // Set dimensions based on comparison between image and viewport aspect ratios
            if (imageAspectRatio > viewportAspectRatio) {
                // Image is wider relative to viewport - constrain by width
                lightboxImage.style.width = '100vw'
                lightboxImage.style.height = 'auto'
                lightboxImage.style.maxHeight = '100vh'
            } else {
                // Image is taller relative to viewport - constrain by height
                lightboxImage.style.width = 'auto'
                lightboxImage.style.height = '100vh'
                lightboxImage.style.maxWidth = '100vw'
            }
        }
        tempImg.src = imgData.src
    }

    function jumpToOriginal() {
        if (currentLightboxIndex < 0 || currentLightboxIndex >= lightboxImages.length) return

        const imgData = lightboxImages[currentLightboxIndex]

        // Close lightbox first
        closeLightbox()

        // Directory images do not have a corresponding preview element.
        if (!imgData.element) return

        // Switch to main tab
        const mainTab = document.querySelector('#tab-main')
        if (mainTab) {
            mainTab.click()
        }

        // Scroll to original
        setTimeout(() => {
            if (imgData.element) {
                imgData.element.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                })
            }
        }, 300)
    }

    // Lightbox Event Listeners
    lightboxPrev?.addEventListener('click', (e) => {
        e.stopPropagation()
        navigateLightbox(-1)
    })

    lightboxNext?.addEventListener('click', (e) => {
        e.stopPropagation()
        navigateLightbox(1)
    })

    lightboxClose?.addEventListener('click', (e) => {
        e.stopPropagation()
        closeLightbox()
    })

    // Click on image to jump to original (only the actual image element)
    lightboxImage?.addEventListener('click', (e) => {
        e.stopPropagation()
        jumpToOriginal()
    })

    // Click on content background to close (anywhere except the image and controls)
    lightboxContent?.addEventListener('click', (e) => {
        // Only close if clicking on the content background, not on the image or controls
        if (e.target === lightboxContent) {
            closeLightbox()
        }
    })

    // Click on lightbox background to close
    lightbox?.addEventListener('click', (e) => {
        if (e.target === lightbox) {
            closeLightbox()
        }
    })

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (!lightboxOpen) return

        if (e.key === 'Escape') {
            closeLightbox()
        } else if (e.key === 'ArrowLeft') {
            navigateLightbox(-1)
        } else if (e.key === 'ArrowRight') {
            navigateLightbox(1)
        }
    })

    // Function to load an image
    function loadImage(src) {
        return new Promise((res, rej) => {
            const img = new Image()
            img.crossOrigin = "Anonymous"
            img.onload = () => res(img)
            img.onerror = rej
            img.src = src
        })
    }

    // Function to create collage
    async function createCollage(mode) {
        if (selectedImages.size === 0) return

        if (!galleryContainer) return

        // Get selected image sources in order
        const selectedWrappers = Array.from(galleryContainer.querySelectorAll('.gallery-image-wrapper.selected'))
        const imageSources = selectedWrappers
            .map(wrapper => imageCache.get(wrapper.dataset.uniqueId)?.src)
            .filter(Boolean)

        // Load all images
        const images = await Promise.all(imageSources.map(src => loadImage(src)))

        let canvas, ctx
        const gap = 0

        if (mode === 'horizontal') {
            // Horizontal: same height, variable width
            const targetHeight = Math.min(...images.map(img => img.height))
            const scaledImages = images.map(img => ({
                img,
                width: Math.round((img.width / img.height) * targetHeight),
                height: targetHeight
            }))

            const totalWidth = scaledImages.reduce((sum, si) => sum + si.width, 0)

            canvas = document.createElement('canvas')
            canvas.width = totalWidth
            canvas.height = targetHeight
            ctx = canvas.getContext('2d')

            let xPos = 0
            for (const si of scaledImages) {
                ctx.drawImage(si.img, xPos, 0, si.width, si.height)
                xPos += si.width + gap
            }

        } else if (mode === 'vertical') {
            // Vertical: same width, variable height
            const targetWidth = Math.min(...images.map(img => img.width))
            const scaledImages = images.map(img => ({
                img,
                width: targetWidth,
                height: Math.round((img.height / img.width) * targetWidth)
            }))

            const totalHeight = scaledImages.reduce((sum, si) => sum + si.height, 0)

            canvas = document.createElement('canvas')
            canvas.width = targetWidth
            canvas.height = totalHeight
            ctx = canvas.getContext('2d')

            let yPos = 0
            for (const si of scaledImages) {
                ctx.drawImage(si.img, 0, yPos, si.width, si.height)
                yPos += si.height + gap
            }

        } else if (mode === 'grid') {
            // Grid: calculate optimal grid layout
            const count = images.length
            const cols = Math.ceil(Math.sqrt(count))
            const rows = Math.ceil(count / cols)

            // Base grid cell on largest dimensions so we never squash
            const refWidth = Math.max(...images.map(img => img.width))
            const refHeight = Math.max(...images.map(img => img.height))

            canvas = document.createElement('canvas')
            canvas.width = cols * refWidth
            canvas.height = rows * refHeight
            ctx = canvas.getContext('2d')

            // Fill background
            ctx.fillStyle = '#191A1B'
            ctx.fillRect(0, 0, canvas.width, canvas.height)

            let imgIndex = 0
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    if (imgIndex >= images.length) break

                    const img = images[imgIndex]
                    const cellX = c * refWidth
                    const cellY = r * refHeight

                    // Fit image inside cell without stretching up
                    const scale = Math.min(refWidth / img.width, refHeight / img.height, 1)
                    const drawW = Math.round(img.width * scale)
                    const drawH = Math.round(img.height * scale)
                    const xPos = cellX + Math.floor((refWidth - drawW) / 2)
                    const yPos = cellY + Math.floor((refHeight - drawH) / 2)

                    ctx.drawImage(img, xPos, yPos, drawW, drawH)
                    imgIndex++
                }
            }
        }

        // Download collage
        const dataUrl = canvas.toDataURL('image/png')
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
        const filename = `gallery-collage-${mode}-${timestamp}.png`

        const a = document.createElement('a')
        a.href = dataUrl
        a.download = filename
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)

        // Clean up
        canvas.width = 0
        canvas.height = 0
    }

    async function deleteGalleryImage(imgData) {
        if (!imgData?.src || !imgData.relativePath) {
            throw new Error('This image is not a directory-backed gallery file.')
        }
        const response = await fetch(imgData.src, { method: 'DELETE' })
        if (!response.ok) {
            let message = `Delete failed (${response.status})`
            try {
                const result = await response.json()
                message = result.detail || message
            } catch (_) {
                // Keep the status-based error.
            }
            throw new Error(message)
        }
    }

    // Function to delete selected images
    async function deleteSelectedImages() {
        if (selectedImages.size === 0) return

        if (!galleryContainer) return
        const confirmed = await confirmGalleryDeletion(
            `Permanently delete ${selectedImages.size} selected image` +
            `${selectedImages.size === 1 ? '' : 's'} from the gallery directory?\n\n` +
            `This cannot be undone.`
        )
        if (!confirmed) return

        isDeletingImage = true
        if (deleteSelectedBtn) deleteSelectedBtn.disabled = true
        const files = Array.from(selectedImages)
            .map(uniqueId => imageCache.get(uniqueId))
            .filter(Boolean)

        try {
            await Promise.all(files.map(deleteGalleryImage))
            selectedImages.clear()
            imageCache.clear()
            await updateGallery()
        } catch (error) {
            console.error('Could not delete selected gallery images:', error)
            alert(error.message)
        } finally {
            isDeletingImage = false
            updateSelectionUI()
        }
    }

    // Function to update gallery display
    async function updateGallery(precollectedImages = null) {
        if (!galleryContainer || !galleryEmpty) {
            return
        }

        let galleryImages
        try {
            galleryImages = precollectedImages
                ? [...precollectedImages]
                : await collectImages()
        } catch (error) {
            console.error('Could not update directory gallery:', error)
            galleryContainer.style.display = 'none'
            galleryEmpty.style.display = 'block'
            galleryEmpty.textContent = error.message
            if (gallerySource) gallerySource.textContent = 'Gallery directory could not be loaded.'
            return
        }

        // Remove selections that no longer exist
        const currentIds = new Set(galleryImages.map(img => img.uniqueId))
        for (const id of Array.from(selectedImages)) {
            if (!currentIds.has(id)) {
                selectedImages.delete(id)
            }
        }
        lastGalleryIds = currentIds

        // Sort by timestamp (newest first)
        galleryImages.sort((a, b) => b.timestamp - a.timestamp)

        // Update lightbox images array
        lightboxImages = galleryImages

        // Update display
        if (galleryImages.length === 0) {
            galleryEmpty.style.display = 'block'
            galleryContainer.style.display = 'none'
        } else {
            galleryEmpty.style.display = 'none'
            galleryContainer.style.display = 'flex'

            // Get existing wrappers to check what needs to be updated
            const existingWrappers = new Map()
            galleryContainer.querySelectorAll('.gallery-image-wrapper').forEach(wrapper => {
                const id = wrapper.dataset.uniqueId
                if (id) existingWrappers.set(id, wrapper)
            })

            // Check if we need to rebuild (order changed or items added/removed)
            let needsRebuild = existingWrappers.size !== galleryImages.length
            if (!needsRebuild) {
                const existingOrder = Array.from(galleryContainer.querySelectorAll('.gallery-image-wrapper'))
                    .map(w => w.dataset.uniqueId)
                const newOrder = galleryImages.map(img => img.uniqueId)
                needsRebuild = existingOrder.join(',') !== newOrder.join(',')
            }

            if (needsRebuild) {
                // Hide container during rebuild
                galleryContainer.style.visibility = 'hidden'

                // Build the entire gallery in a document fragment first
                const fragment = document.createDocumentFragment()
                let loadedImages = 0
                const totalImages = galleryImages.length

                const checkAllLoaded = () => {
                    loadedImages++
                    if (loadedImages === totalImages) {
                        // All images loaded, now show the container
                        requestAnimationFrame(() => {
                            galleryContainer.style.visibility = 'visible'
                        })
                    }
                }

                galleryImages.forEach((imgData, index) => {
                    const wrapper = document.createElement('div')
                    wrapper.className = 'gallery-image-wrapper'
                    wrapper.dataset.uniqueId = imgData.uniqueId

                    // Restore selection state
                    if (selectedImages.has(imgData.uniqueId)) {
                        wrapper.classList.add('selected')
                    }

                    // --- CREATE CHECKBOX ---
                    const checkbox = document.createElement('input')
                    checkbox.type = 'checkbox'
                    checkbox.className = 'gallery-checkbox'
                    checkbox.checked = selectedImages.has(imgData.uniqueId)

                    checkbox.addEventListener('change', (e) => {
                        e.stopPropagation()

                        if (e.target.checked) {
                            selectedImages.add(imgData.uniqueId)
                            wrapper.classList.add('selected')
                        } else {
                            selectedImages.delete(imgData.uniqueId)
                            wrapper.classList.remove('selected')
                        }

                        updateSelectionUI()
                    })

                    wrapper.appendChild(checkbox)

                    // --- CREATE DELETE BUTTON ---
                    const deleteBtn = document.createElement('div')
                    deleteBtn.className = 'gallery-delete-btn'
                    deleteBtn.innerHTML = '&times;'
                    deleteBtn.title = 'Delete image file'
                    deleteBtn.setAttribute('role', 'button')
                    deleteBtn.setAttribute('aria-label', `Delete ${imgData.filename || 'image'}`)
                    deleteBtn.tabIndex = 0

                    deleteBtn.addEventListener('click', async (e) => {
                        e.stopPropagation()
                        const confirmed = await confirmGalleryDeletion(
                            `Permanently delete this file?\n\n` +
                            `${imgData.relativePath || imgData.filename || 'Selected image'}\n\n` +
                            `This cannot be undone.`
                        )
                        if (!confirmed) {
                            return
                        }
                        isDeletingImage = true
                        deleteBtn.style.pointerEvents = 'none'
                        try {
                            await deleteGalleryImage(imgData)
                            selectedImages.delete(imgData.uniqueId)
                            imageCache.clear()
                            await updateGallery()
                        } catch (error) {
                            console.error('Could not delete gallery image:', error)
                            alert(error.message)
                        } finally {
                            isDeletingImage = false
                            deleteBtn.style.pointerEvents = ''
                            updateSelectionUI()
                        }
                    })
                    deleteBtn.addEventListener('keydown', (event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault()
                            deleteBtn.click()
                        }
                    })
                    wrapper.appendChild(deleteBtn)

                    // --- CREATE DOWNLOAD BUTTON ---
                    const downloadBtn = document.createElement('div')
                    downloadBtn.className = 'gallery-download-btn'
                    downloadBtn.innerHTML = '↓'
                    downloadBtn.title = 'Download Image'

                    downloadBtn.addEventListener('click', (e) => {
                        e.stopPropagation()
                        downloadImage(imgData.src, imgData)
                    })
                    wrapper.appendChild(downloadBtn)

                    const img = document.createElement('img')
                    img.src = imgData.thumbnailSrc

                    // Store zoom scale on wrapper
                    wrapper.dataset.zoomPercent = zoomSlider ? zoomSlider.value : 100

                    // Apply zoom after image loads
                    let loadRecorded = false
                    const recordLoaded = () => {
                        if (loadRecorded) return
                        loadRecorded = true
                        checkAllLoaded()
                    }

                    img.onload = function () {
                        const naturalWidth = this.naturalWidth
                        const zoomPercent = parseFloat(wrapper.dataset.zoomPercent)
                        const scaleFactor = zoomPercent / 100
                        const newWidth = Math.max(naturalWidth * scaleFactor, 50)
                        this.style.width = `${newWidth}px`
                        recordLoaded()
                    }
                    img.onerror = recordLoaded

                    // Apply zoom immediately if already cached
                    if (img.complete) {
                        const naturalWidth = img.naturalWidth
                        const zoomPercent = parseFloat(wrapper.dataset.zoomPercent)
                        const scaleFactor = zoomPercent / 100
                        const newWidth = Math.max(naturalWidth * scaleFactor, 50)
                        img.style.width = `${newWidth}px`
                        recordLoaded()
                    }

                    // Click handler to open lightbox
                    img.addEventListener('click', () => {
                        openLightbox(index)
                    })

                    wrapper.appendChild(img)
                    fragment.appendChild(wrapper)
                })

                // Clear container and append everything at once
                galleryContainer.innerHTML = ''
                galleryContainer.appendChild(fragment)

                // Fallback: show container after 1 second even if images haven't loaded
                setTimeout(() => {
                    galleryContainer.style.visibility = 'visible'
                }, 1000)
            }
        }

        // Update count
        if (galleryCountSpan) {
            galleryCountSpan.textContent = galleryTotal
        }

        updateSelectionUI()
    }

    // Debounced update function
    function scheduleUpdate() {
        if (updateTimeout) {
            clearTimeout(updateTimeout)
        }
        updateTimeout = setTimeout(() => {
            updateGallery()
            updateTimeout = null
        }, 300)
    }

    // Set up mutation observer
    function startObserving() {
        const previewArea = document.querySelector('#preview')
        if (!previewArea) {
            setTimeout(startObserving, 1000)
            return
        }

        imageObserver = new MutationObserver((mutations) => {
            // Skip updates if we're in the middle of deleting an image
            if (isDeletingImage) {
                return
            }

            let shouldUpdate = false

            mutations.forEach((mutation) => {
                if (mutation.addedNodes.length > 0 || mutation.removedNodes.length > 0) {
                    mutation.addedNodes.forEach(node => {
                        if (node.nodeType === 1) {
                            if (node.tagName === 'IMG' || node.querySelector('img')) {
                                shouldUpdate = true
                            }
                        }
                    })

                    mutation.removedNodes.forEach(node => {
                        if (node.nodeType === 1) {
                            if (node.tagName === 'IMG' || node.querySelector('img')) {
                                shouldUpdate = true
                            }
                        }
                    })
                }
                else if (mutation.type === 'attributes' &&
                    mutation.attributeName === 'src' &&
                    mutation.target.tagName === 'IMG') {
                    shouldUpdate = true
                }
            })

            if (shouldUpdate) {
                scheduleUpdate()
            }
        })

        imageObserver.observe(previewArea, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['src']
        })

        // Periodic check as fallback
        refreshIntervalId = setInterval(async () => {
            try {
                const newImages = await collectImages()
                const newIds = new Set(newImages.map(img => img.uniqueId))
                let changed = newIds.size !== lastGalleryIds.size
                if (!changed) {
                    for (const id of newIds) {
                        if (!lastGalleryIds.has(id)) {
                            changed = true
                            break
                        }
                    }
                }
                if (changed) {
                    await updateGallery(newImages)
                } else {
                    lastGalleryIds = newIds
                }
            } catch (error) {
                console.warn('Gallery refresh failed:', error)
            }
        }, 10000)
    }

    // Clean up on unload
    window.addEventListener('beforeunload', () => {
        if (imageObserver) {
            imageObserver.disconnect()
        }
        if (updateTimeout) {
            clearTimeout(updateTimeout)
        }
        if (refreshIntervalId) {
            clearInterval(refreshIntervalId)
        }
    })

    // Zoom slider handler
    zoomSlider?.addEventListener('input', (e) => {
        const zoomPercent = e.target.value
        const scaleFactor = zoomPercent / 100

        const wrappers = document.querySelectorAll('.gallery-image-wrapper')
        wrappers.forEach(wrapper => {
            wrapper.dataset.zoomPercent = zoomPercent
            const img = wrapper.querySelector('img')
            if (img && img.complete) {
                const naturalWidth = img.naturalWidth
                const newWidth = Math.max(naturalWidth * scaleFactor, 50)
                img.style.width = `${newWidth}px`
            }
        })
    })

    // Collage button handlers
    collageHorizontalBtn?.addEventListener('click', () => {
        createCollage('horizontal')
    })

    collageVerticalBtn?.addEventListener('click', () => {
        createCollage('vertical')
    })

    collageGridBtn?.addEventListener('click', () => {
        createCollage('grid')
    })

    deleteSelectedBtn?.addEventListener('click', () => {
        deleteSelectedImages()
    })

    // Select All / Deselect All button handlers
    selectAllBtn?.addEventListener('click', () => {
        if (!galleryContainer) return

        const wrappers = galleryContainer.querySelectorAll('.gallery-image-wrapper')
        wrappers.forEach(wrapper => {
            const uniqueId = wrapper.dataset.uniqueId
            const checkbox = wrapper.querySelector('.gallery-checkbox')

            if (uniqueId && checkbox) {
                selectedImages.add(uniqueId)
                wrapper.classList.add('selected')
                checkbox.checked = true
            }
        })

        updateSelectionUI()
    })

    deselectAllBtn?.addEventListener('click', () => {
        if (!galleryContainer) return

        const wrappers = galleryContainer.querySelectorAll('.gallery-image-wrapper')
        wrappers.forEach(wrapper => {
            const checkbox = wrapper.querySelector('.gallery-checkbox')

            wrapper.classList.remove('selected')
            if (checkbox) checkbox.checked = false
        })

        selectedImages.clear()
        updateSelectionUI()
    })

    // Auto-refresh when switching to gallery tab
    document.addEventListener('tabClick', (e) => {
        if (e.detail.name === 'gallery') {
            updateGallery()

            if (zoomSlider) {
                const zoomPercent = zoomSlider.value
                const scaleFactor = zoomPercent / 100
                setTimeout(() => {
                    const wrappers = galleryContainer ? galleryContainer.querySelectorAll('.gallery-image-wrapper') : []
                    wrappers.forEach(wrapper => {
                        const img = wrapper.querySelector('img')
                        if (img && img.complete) {
                            const naturalWidth = img.naturalWidth
                            const newWidth = Math.max(naturalWidth * scaleFactor, 50)
                            img.style.width = `${newWidth}px`
                        }
                    })
                }, 50)
            }
        }
    })

    // Initialize
    setTimeout(async () => {
        // Clear any leftover selections
        selectedImages.clear()
        startObserving()
        await loadGallerySettings()
        await updateGallery()
    }, 1000)

})();
