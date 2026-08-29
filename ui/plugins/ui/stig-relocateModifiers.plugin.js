/**
 * Relocate Modifiers Plugin
 * v.1.1, last updated: 29/08/2026
 * By The Stig
 *
 * Change Log
 * 26/08/2026 Initial Beta Build
 * 27/08/2026 Rewrote code to fix unknown bugs
 * 29/08/2026 Added Thumbnail Size slider
*/



(function() {
    'use strict';
    if (window.__relocateModifiersPluginLoaded) return;
    window.__relocateModifiersPluginLoaded = true;

    function addOptions() {
        const anchor = document.getElementById('editor-settings');
        if (!anchor || document.getElementById('modifier-mirror-panel')) return;

        const styleSheet = document.createElement("style");
        styleSheet.textContent = `
            #modifier-mirror-panel { --dup-card-pixel-size: 70px; }
            #modifier-mirror-panel::before, #modifier-mirror-panel::after,
            #modifier-mirror-panel *::before, #modifier-mirror-panel *::after,
            #modifier-mirror-panel *::marker, #modifier-mirror-panel li::marker {
                content: none !important; display: none !important; background: none !important; background-image: none !important;
            }

            #mirrored-icons-container { max-width:100% !important; box-sizing:border-box !important; padding:0 !important; margin:0 !important; }
            .clean-category-div-wrapper { display:block !important; margin-bottom:12px !important; width:100% !important; padding:0 !important; text-indent: 0 !important; }

            .category-header-strip {
                background: var(--background-color2, #21252b) !important; padding:8px 12px !important;
                font-size:12px !important; font-weight:bold !important; cursor:pointer !important;
                display:flex !important; align-items:center !important;
                border-radius:4px !important; color:var(--text-color,#fff) !important; margin:0 0 6px 0 !important; user-select:none !important;
            }
            #modifier-mirror-panel .category-header-strip::before {
                content: '+ ' !important; display: inline-block !important; font-weight: bold !important; color: var(--accent-color, #a855f7) !important; margin-right: 6px !important;
            }
            #modifier-mirror-panel .expanded-strip::before { content: '− ' !important; }

            .clean-cards-grid-tray {
                display:none !important;
                grid-template-columns: repeat(auto-fill, minmax(var(--dup-card-pixel-size), 1fr)) !important;
                gap:6px !important; padding:4px 0 !important; width:100% !important; box-sizing:border-box !important;
            }

            /* 🛠️ CARD BLOCK ALIGNMENT REWRITE: Forces flex centering on the cards themselves to lock custom icons in place */
            #mirrored-icons-container .modifier-card, #mirrored-icons-container .modifier-item {
                position:relative !important;
                width: 100% !important;
                max-width: 100% !important;
                height: calc(var(--dup-card-pixel-size) + 24px) !important;
                box-sizing:border-box !important;
                background:#13151b !important;
                border:1px solid #2d3139 !important;
                border-radius:6px !important;
                overflow:hidden !important;
                cursor:pointer !important;

                /* Layout Shift Fixes */
                display: flex !important;
                flex-direction: column !important;
                justify-content: space-between !important;
                align-items: center !important;
                margin:0 !important; padding:0 !important;
            }

            #mirrored-icons-container .modifier-card *,
            #mirrored-icons-container .modifier-item * {
                max-width: 100% !important;
                text-align: center !important;
            }

            /* Proportional Image scaling bounds */
            #mirrored-icons-container .modifier-card img, #mirrored-icons-container .modifier-item img {
                width:100% !important;
                height: var(--dup-card-pixel-size) !important;
                object-fit: contain !important;
                object-position: center !important;
                background-color: #13151b !important;
                display:block !important;
                margin: 0 auto !important;
                position: relative !important;
                left: 0 !important;
                right: 0 !important;
            }

            #mirrored-icons-container .modifier-card-label, #mirrored-icons-container label {
                padding:4px 2px !important; font-size:9px !important; font-weight:500 !important; color:#fff !important; text-align:center !important;
                width: 100% !important; box-sizing:border-box !important; text-overflow:ellipsis !important; overflow:hidden !important; white-space:nowrap !important;
                background:rgba(0,0,0,0.5) !important;
                margin: auto 0 0 0 !important; /* Locks text perfectly to the card footer layer floor */
            }
        `;
        document.head.appendChild(styleSheet);

        const panel = document.createElement('div');
        panel.id = 'modifier-mirror-panel';
        panel.className = 'sidebar-section panel-box';
        panel.innerHTML = window.loadRequiredPluginHTML('/plugins/core/stig-relocateModifiers.plugin.html');

        if (anchor.parentNode) anchor.parentNode.insertBefore(panel, anchor.nextSibling);
        if (typeof window.createCollapsibles === 'function') window.createCollapsibles(panel);

        const slider = panel.querySelector('#dup-size-slider');
        if (slider) {
            slider.addEventListener('input', (e) => {
                panel.style.setProperty('--dup-card-pixel-size', `${e.target.value}px`);
            });
        }

        rebuildCleanStructure();
    }

    function rebuildCleanStructure() {
        const box = document.getElementById('mirrored-icons-container');
        if (!box) return;

        const source = document.getElementById('editor-modifiers-entries');
        if (!source) { setTimeout(rebuildCleanStructure, 300); return; }

        const nativeCategories = source.querySelectorAll('.modifier-category');
        if (nativeCategories.length === 0) { setTimeout(rebuildCleanStructure, 300); return; }

        box.innerHTML = '';

        nativeCategories.forEach((origCat) => {
            const nativeHead = origCat.querySelector('h4, h5, .category-title, .collapsible, label, .category-header');
            const nativeTray = origCat.querySelector('div:not(h4):not(h5)');
            if (!nativeHead || !nativeTray) return;

            const cleanWrapper = document.createElement('div');
            cleanWrapper.className = 'clean-category-div-wrapper';

            let titleText = nativeHead.textContent.trim().replace(/^[\u2022\u25CF\u25AA\u25FE\u25A0\u2023\u25B6\s•●▪▸\+\-−\s]+/g, '');

            const cleanHeaderRow = document.createElement('div');
            cleanHeaderRow.className = 'category-header-strip';
            cleanHeaderRow.textContent = titleText;

            const cleanCardsTray = nativeTray.cloneNode(true);
            cleanCardsTray.className = 'clean-cards-grid-tray';
            cleanCardsTray.style.setProperty('display', 'none', 'important');

            cleanHeaderRow.addEventListener('click', (e) => {
                e.stopPropagation();
                const isHidden = cleanCardsTray.style.getPropertyValue('display') === 'none';
                cleanCardsTray.style.setProperty('display', isHidden ? 'grid' : 'none', 'important');
                cleanHeaderRow.classList.toggle('expanded-strip', isHidden);
            });

            const originalCards = origCat.querySelectorAll('.modifier-card, .modifier-item');
            const clonedCards = cleanCardsTray.querySelectorAll('.modifier-card, .modifier-item');

            clonedCards.forEach((clonedCard, cardIndex) => {
                const correspondingOriginalCard = originalCards[cardIndex];
                if (!correspondingOriginalCard) return;

                clonedCard.addEventListener('click', (e) => {
                    e.stopPropagation();
                    correspondingOriginalCard.click();
                    clonedCard.classList.toggle('active');
                    clonedCard.style.setProperty('border-color', (clonedCard.classList.contains('active') || correspondingOriginalCard.classList.contains('active')) ? 'var(--accent-color, #a855f7)' : '#2d3139', 'important');
                });
            });

            cleanWrapper.appendChild(cleanHeaderRow);
            cleanWrapper.appendChild(cleanCardsTray);
            box.appendChild(cleanWrapper);
        });
    }

    if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", () => setTimeout(addOptions, 1200)); }
    else { setTimeout(addOptions, 1200); }
})();
