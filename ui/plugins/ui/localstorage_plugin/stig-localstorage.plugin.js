/**
 * LocalStorage Manager
 * v.1.0, last updated: 28/08/2026
 * By The Stig
 *
 * Change Log
 * 28/08/2026 Initial Build
 * 28/08/2026 Integrated dynamic key scanning
 * 28/08/2026 Algorithmic signature detection
 * 28/08/2026 Real-time byte calculation
 * 28/08/2026 Dynamic dropdown filtering by source
 * 28/08/2026 Live source sub-totals
 * 28/08/2026 Hidden double-click delete protection.
 */

/**
 * Block 1
 * Fast-loading storage layout skeleton with master visibility toggle controls,
 * isolated source filtering dropdowns, and individual filter sub-total tracking frames.
 */
(function() {
    "use strict";
    if (window.__storageManagerPluginLoaded) return;
    window.__storageManagerPluginLoaded = true;

    // --- master visibility toggle controls ---
    const showDeleteButtons = false; // Set to true to reveal 'X' delete buttons, or false to hide them completely!
    // -----------------------------------------

    window.ED_StorageManager_ShowDelete = showDeleteButtons;
    window.ED_StorageManager_LastFilter = "ALL";

    const old = document.getElementById('storage-manager-grid-panel');
    if (old) old.remove();

    const panel = document.createElement('div');
    panel.id = 'storage-manager-grid-panel';
    panel.classList.add('settings-box', 'panel-box');
    panel.innerHTML = window.loadRequiredPluginHTML('/plugins/core/localstorage_plugin/stig-localstorage.plugin.html');

    var editorSettings = document.getElementById('editor-settings') || document.getElementById('editor-inputs');
    if (editorSettings && editorSettings.parentNode) {
        editorSettings.parentNode.insertBefore(panel, editorSettings.nextSibling);
    } else {
        (document.getElementById('editor-inputs') || document.querySelector('.left-panel') || document.body).appendChild(panel);
    }

    if (typeof createCollapsibles === 'function') { createCollapsibles(panel); }

    const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (character) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    })[character]);


/**
 * Block 2
 * Fast data loop mapping system with sorted byte calculations, algorithmic signature matches,
 * dynamic dropdown generation, and isolated active source sub-total calculations.
 */
    const renderStorageItems = () => {
        const listContainer = document.getElementById('storage-item-list');
        const totalSizeLabel = document.getElementById('storage-total-size');
        const filterSelect = document.getElementById('storage-filter-select');
        const subtotalLabel = document.getElementById('storage-filter-subtotal');
        if (!listContainer) return;

        let grandTotalBytes = 0;
        let filteredSourceBytes = 0; // NEW TRACKER: Accumulates bytes of just the highlighted category
        const itemsArray = [];
        const storageLength = localStorage.length;
        const discoveredCategories = new Set();

        for (let i = 0; i < storageLength; i++) {
            const keyName = localStorage.key(i);
            if (!keyName) continue;

            const valueString = localStorage.getItem(keyName) || "";
            const itemBytes = (keyName.length + valueString.length) * 2;
            grandTotalBytes += itemBytes;

            itemsArray.push({ key: keyName, val: valueString, bytes: itemBytes });
        }

        itemsArray.sort((a, b) => b.bytes - a.bytes);

        const processedItems = itemsArray.map(item => {
            let creatorTag = "[Unknown Plugin]";
            let tagColor = "#9e9e9e";
            const lowerKey = item.key.toLowerCase();
            const lowerVal = item.val.toLowerCase();

            if (lowerKey.startsWith("ed_grid_")) {
                creatorTag = "[Matrix: Sampler]";
                tagColor = "#4fc3f7";
            } else if (lowerKey.startsWith("ed_promptgrid_")) {
                creatorTag = "[Matrix: Prompt]";
                tagColor = "#ba68c8";
            } else if (lowerKey.includes("quota") || lowerKey.includes("test")) {
                creatorTag = "[System Test Junk]";
                tagColor = "#e57373";
            } else if (lowerKey.startsWith("theme") || lowerKey.includes("dark-mode") || lowerKey.includes("ui-") || lowerKey.startsWith("style:")) {
                creatorTag = "[UI Core Theme]";
                tagColor = "#a1887f";
            } else if (lowerKey.includes("button") || lowerKey.includes("hover") || lowerKey.includes("infobtn")) {
                creatorTag = "[Hover Info Buttons]";
                tagColor = "#ffb74d";
            } else if (lowerKey.includes("history") || lowerKey.includes("recent") || lowerKey.includes("last_prompt")) {
                creatorTag = "[Prompt History Log]";
                tagColor = "#ffd54f";
            } else if (lowerKey.startsWith("ed_") || lowerKey.includes("easydiffusion") || lowerKey.startsWith("sd-ui")) {
                creatorTag = "[Easy Diffusion Core]";
                tagColor = "#81c784";
            }

            if (creatorTag === "[Unknown Plugin]") {
                if (lowerVal.includes("model_name") || lowerVal.includes("vae_name") || lowerVal.includes("hypernetwork")) {
                    creatorTag = "[Model / VAE Preset]";
                    tagColor = "#26a69a";
                } else if (lowerVal.includes("lora_") || lowerVal.includes("loraname")) {
                    creatorTag = "[LoRA Manager Plugin]";
                    tagColor = "#00bcd4";
                } else if (lowerVal.includes("plugin") || lowerVal.includes("author") || lowerVal.includes("version")) {
                    creatorTag = "[Extension Config]";
                    tagColor = "#78909c";
                } else if (lowerVal.includes("controlnet") || lowerVal.includes("preprocessor")) {
                    creatorTag = "[ControlNet Extension]";
                    tagColor = "#ab47bc";
                }
            }

            discoveredCategories.add(creatorTag);
            return { ...item, creatorTag, tagColor };
        });

        if (filterSelect) {
            const currentSelection = window.ED_StorageManager_LastFilter;
            let dropdownHTML = `<option value="ALL">All Sources (Show Everything)</option>`;

            const tagLabels = {
                "[Matrix: Sampler]": "Sampler Matrix Plugin",
                "[Matrix: Prompt]": "Prompt Modifier Plugin",
                "[Hover Info Buttons]": "Hover Info Buttons Plugin",
                "[Prompt History Log]": "Prompt History Logs",
                "[Easy Diffusion Core]": "Easy Diffusion Core System",
                "[UI Core Theme]": "UI Core Themes / Styles",
                "[Model / VAE Preset]": "Model & VAE Presets",
                "[LoRA Manager Plugin]": "LoRA Plugins",
                "[ControlNet Extension]": "ControlNet Extensions",
                "[System Test Junk]": "System Test Junk / Quota Buffers",
                "[Unknown Plugin]": "Unknown Plugin / Possible Stale Data" // UPDATED LABEL PER REQUEST
            };

            Object.keys(tagLabels).forEach(tagKey => {
                if (discoveredCategories.has(tagKey)) {
                    dropdownHTML += `<option value="${tagKey}">${tagLabels[tagKey]}</option>`;
                }
            });

            filterSelect.innerHTML = dropdownHTML;
            if (discoveredCategories.has(currentSelection) || currentSelection === "ALL") {
                filterSelect.value = currentSelection;
            } else {
                filterSelect.value = "ALL";
                window.ED_StorageManager_LastFilter = "ALL";
            }
        }

        let HTMLBuffer = "";
        let visibleCount = 0;
        const activeFilter = window.ED_StorageManager_LastFilter;

        processedItems.forEach(item => {
            // Check byte totals for specific source allocation tracking independently of display filtering
            if (item.creatorTag === activeFilter) {
                filteredSourceBytes += item.bytes;
            }

            if (activeFilter !== "ALL" && item.creatorTag !== activeFilter) {
                return;
            }

            visibleCount++;
            const kbSize = (item.bytes / 1024).toFixed(2);
            const shortVal = item.val.length > 35 ? item.val.substring(0, 32) + "..." : item.val;
            const safeKey = escapeHTML(item.key);
            const safeShortVal = escapeHTML(shortVal);
            const safeCreatorTag = escapeHTML(item.creatorTag);
            const deleteButtonHtml = window.ED_StorageManager_ShowDelete ?
                `<button class="delete-storage-item-btn" data-key="${safeKey}" style="background:#4a151b; color:#ff8585; border:1px solid #721c24; border-radius:3px; padding:2px 6px; font-size:10px; font-weight:bold; cursor:pointer;">X</button>` : '';

            HTMLBuffer += `
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #333; padding-bottom:6px; margin-bottom:4px; gap:8px;">
                    <div style="flex:1; min-width:0;">
                        <div style="font-size:11px; font-weight:bold; color:#fff; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${safeKey}">${safeKey}</div>
                        <div style="display:flex; gap:6px; align-items:center; margin-top:2px;">
                            <span style="font-size:9px; font-weight:bold; color:${item.tagColor}; background:rgba(0,0,0,0.2); padding:1px 4px; border-radius:2px; border:1px solid ${item.tagColor}44;">${safeCreatorTag}</span>
                            <span style="font-size:10px; color:#aaa; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${safeShortVal}</span>
                        </div>
                    </div>
                    <span style="font-size:10px; color:#ffc107; font-weight:bold; white-space:nowrap; margin-right:${window.ED_StorageManager_ShowDelete ? '0' : '5px'};">${kbSize} KB</span>
                    ${deleteButtonHtml}
                </div>
            `;
        });

        if (visibleCount === 0 && activeFilter !== "ALL") {
            listContainer.style.display = "none";
        } else {
            listContainer.style.display = "flex";
            listContainer.innerHTML = HTMLBuffer;
        }

        // DYNAMIC SUB-TOTAL CARD DRAW LAYER
        if (subtotalLabel) {
            if (activeFilter === "ALL" || filteredSourceBytes === 0) {
                subtotalLabel.style.display = "none";
            } else {
                subtotalLabel.style.display = "block";
                subtotalLabel.innerHTML = `Selected Source Uses: <span style="color:#ffc107; font-weight:bold;">${(filteredSourceBytes / 1024).toFixed(2)} KB</span>`;
            }
        }

        totalSizeLabel.innerText = `${(grandTotalBytes / 1024).toFixed(1)} KB`;
    };


/**
 * Block 3
 * Locked delete verification listeners, dropdown change handlers,
 * flash indicators, and delayed execution boots.
 */
    document.getElementById('storage-refresh-btn').addEventListener('click', (e) => {
        renderStorageItems();

        const refreshBtn = e.currentTarget;
        const originalText = "Refresh Storage List";

        refreshBtn.innerText = "✓ List Refreshed!";
        refreshBtn.style.background = "#2e7d32";
        refreshBtn.style.color = "#ffffff";
        refreshBtn.style.borderColor = "#81c784";
        refreshBtn.disabled = true;

        setTimeout(() => {
            refreshBtn.innerText = originalText;
            refreshBtn.style.background = "";
            refreshBtn.style.color = "";
            refreshBtn.style.borderColor = "";
            refreshBtn.disabled = false;
        }, 2000);
    });

    const filterSelect = document.getElementById('storage-filter-select');
    if (filterSelect) {
        filterSelect.addEventListener('change', (e) => {
            window.ED_StorageManager_LastFilter = e.target.value;
            renderStorageItems();
        });
    }

    document.getElementById('storage-item-list').addEventListener('click', (e) => {
        if (!window.ED_StorageManager_ShowDelete) return;

        const btn = e.target;
        if (btn.classList.contains('delete-storage-item-btn')) {
            const isConfirmedState = btn.getAttribute('data-confirm') === 'true';

            if (!isConfirmedState) {
                btn.setAttribute('data-confirm', 'true');
                btn.innerText = "SURE?";
                btn.style.background = "#d32f2f";
                btn.style.color = "#ffffff";
                btn.style.border = "1px solid #ffb3b3";

                setTimeout(() => {
                    if (btn && btn.getAttribute('data-confirm') === 'true') {
                        btn.setAttribute('data-confirm', 'false');
                        btn.innerText = "X";
                        btn.style.background = "#4a151b";
                        btn.style.color = "#ff8585";
                        btn.style.border = "1px solid #721c24";
                    }
                }, 4000);

            } else {
                const keyTarget = btn.getAttribute('data-key');
                localStorage.removeItem(keyTarget);
                renderStorageItems();
            }
        }
    });

    renderStorageItems();
})();
