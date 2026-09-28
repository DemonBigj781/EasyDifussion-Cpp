// Release retained generation models from the native backend.
;(function () {
    "use strict"
    const table = document.getElementById("system-settings-table")
    if (!table || document.getElementById("clear-vram-settings")) return

    const row = document.createElement("div")
    row.id = "clear-vram-settings"
    row.dataset.settingId = "clear_vram"
    row.innerHTML = `
        <div><i class="fa fa-memory"></i></div>
        <div>
            <label for="clear-vram-button">Clear VRAM</label>
            <small>Unload generation models from GPU memory while the queue is idle. Models reload on the next generation.</small>
            <small id="clear-vram-status" role="status" aria-live="polite"></small>
        </div>
        <div><button id="clear-vram-button" type="button" class="secondaryButton">Clear VRAM</button></div>`
    const routingRow = document.getElementById("native-device-routing-settings")
    table.insertBefore(row, routingRow || null)
    const button = row.querySelector("button")
    const status = row.querySelector("[role=status]")
    button.addEventListener("click", async () => {
        button.disabled = true
        button.textContent = "Clearing VRAM…"
        status.textContent = "Unloading generation models…"
        try {
            const response = await fetch("/clear-vram", { method: "POST" })
            const result = await response.json()
            if (!response.ok) throw new Error(result.detail || result.message || `HTTP ${response.status}`)
            status.textContent = result.message
        } catch (error) {
            status.textContent = error.message || "Could not clear VRAM. Please try again."
        } finally {
            button.disabled = false
            button.textContent = "Clear VRAM"
        }
    })
})()
