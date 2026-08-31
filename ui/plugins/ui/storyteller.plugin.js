(function () {
    "use strict"
    const ID_PREFIX = "storyteller-plugin"
    if (window.__easyDiffusionStorytellerLoaded) return
    window.__easyDiffusionStorytellerLoaded = true

    const style = document.createElement("style")
    style.textContent = `
        #tab-content-storyteller .storyteller-toolbar {
            display: flex;
            gap: .5rem;
            margin-bottom: 1rem;
        }
        #storyteller-items-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 1rem;
        }
        .storyteller-item {
            position: relative;
            padding: .5rem;
            border: 1px solid var(--background-color3);
            border-radius: 7px;
            background: var(--background-color2);
        }
        .storyteller-item canvas {
            width: 100%;
            height: auto;
            display: block;
        }
        .storyteller-item-actions {
            display: flex;
            justify-content: flex-end;
            gap: .4rem;
            margin-top: .5rem;
        }
    `
    document.head.appendChild(style)

    const tabContainer = document.getElementById("tab-container") || document.querySelector(".tab-container")
    const contentWrapper = document.getElementById("tab-content-wrapper")
    if (!tabContainer || !contentWrapper) {
        console.error(`${ID_PREFIX}: tab container is unavailable`)
        return
    }

    const tab = document.createElement("span")
    tab.id = "tab-storyteller"
    tab.className = "tab"
    tab.innerHTML = '<span><i class="fa-solid fa-book-open icon"></i> Storyteller</span>'
    tabContainer.appendChild(tab)

    const tabContent = document.createElement("div")
    tabContent.id = "tab-content-storyteller"
    tabContent.className = "tab-content"
    tabContent.innerHTML = `
        <div class="tab-content-inner">
            <h1>Storyteller</h1>
            <div class="storyteller-toolbar">
                <button type="button" class="storyteller-save-all-button secondaryButton">Save All</button>
                <button type="button" class="storyteller-remove-all-button secondaryButton">Remove All</button>
            </div>
            <div id="storyteller-items-container"></div>
        </div>`
    contentWrapper.appendChild(tabContent)

    if (typeof linkTabContents === "function") {
        linkTabContents(tab)
    } else {
        // Compatibility with the older 3.5 UI tab implementation.
        tab.addEventListener("click", () => {
            document.querySelectorAll(".tab-content").forEach((content) => content.style.display = "none")
            document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"))
            tabContent.style.display = "block"
            tab.classList.add("active")
        })
    }

    const items = tabContent.querySelector("#storyteller-items-container")

    function downloadImage(item) {
        const canvas = item.querySelector("canvas")
        const link = document.createElement("a")
        link.download = `storyteller-${Array.from(items.children).indexOf(item) + 1}.png`
        link.href = canvas.toDataURL("image/png")
        link.click()
    }

    tabContent.querySelector(".storyteller-save-all-button").addEventListener("click", () => {
        items.querySelectorAll(".storyteller-item").forEach(downloadImage)
    })
    tabContent.querySelector(".storyteller-remove-all-button").addEventListener("click", () => {
        items.replaceChildren()
    })

    PLUGINS.IMAGE_INFO_BUTTONS.push({
        text: "Add to Storyteller",
        on_click: function (origRequest, image) {
            const width = Number(origRequest.width) || image.naturalWidth || image.width
            const height = Number(origRequest.height) || image.naturalHeight || image.height
            const item = document.createElement("div")
            item.className = "storyteller-item"
            const canvas = document.createElement("canvas")
            canvas.width = width
            canvas.height = height
            canvas.getContext("2d").drawImage(image, 0, 0, width, height)
            const actions = document.createElement("div")
            actions.className = "storyteller-item-actions"
            const saveButton = document.createElement("button")
            saveButton.type = "button"
            saveButton.className = "secondaryButton"
            saveButton.textContent = "Save"
            saveButton.addEventListener("click", () => downloadImage(item))
            const removeButton = document.createElement("button")
            removeButton.type = "button"
            removeButton.className = "secondaryButton"
            removeButton.textContent = "Remove"
            removeButton.addEventListener("click", () => item.remove())
            actions.append(saveButton, removeButton)
            item.append(canvas, actions)
            items.appendChild(item)
        },
    })
})()
