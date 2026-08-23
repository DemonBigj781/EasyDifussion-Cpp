/*
 * Searchable CSV Style Injector for Easy Diffusion
 *
 * Loads styles.csv from this plugin's directory and adds a searchable Styles
 * tab directly after the main Generate tab. Selecting a style applies its
 * positive template to #prompt and prepends its negative prompt to
 * #negative_prompt.
 */
(function () {
    "use strict";

    const ID_PREFIX = "style-inject";
    const TAB_ID = `tab-${ID_PREFIX}`;
    const TAB_CONTENT_ID = `tab-content-${ID_PREFIX}`;
    const SCRIPT_URL = document.currentScript && document.currentScript.src
        ? document.currentScript.src
        : window.location.href;
    const CSV_URL = new URL("styles.csv", SCRIPT_URL);

    function parseCsv(text) {
        const rows = [];
        let row = [];
        let field = "";
        let quoted = false;

        for (let i = 0; i < text.length; i += 1) {
            const char = text[i];

            if (quoted) {
                if (char === '"') {
                    if (text[i + 1] === '"') {
                        field += '"';
                        i += 1;
                    } else {
                        quoted = false;
                    }
                } else {
                    field += char;
                }
                continue;
            }

            if (char === '"') {
                quoted = true;
            } else if (char === ",") {
                row.push(field);
                field = "";
            } else if (char === "\n" || char === "\r") {
                if (char === "\r" && text[i + 1] === "\n") {
                    i += 1;
                }
                row.push(field);
                if (row.some((value) => value.length > 0)) {
                    rows.push(row);
                }
                row = [];
                field = "";
            } else {
                field += char;
            }
        }

        if (field.length > 0 || row.length > 0) {
            row.push(field);
            if (row.some((value) => value.length > 0)) {
                rows.push(row);
            }
        }

        if (quoted) {
            throw new Error("styles.csv contains an unterminated quoted field");
        }

        return rows;
    }

    function normalizeHeader(value) {
        return String(value || "")
            .replace(/^\uFEFF/, "")
            .trim()
            .toLowerCase()
            .replace(/[\s-]+/g, "_");
    }

    function normalizeSearchText(value) {
        return String(value || "")
            .normalize("NFKD")
            .replace(/[\u0300-\u036f]/g, "")
            .toLocaleLowerCase();
    }

    function findColumn(headers, aliases) {
        return aliases
            .map((alias) => headers.indexOf(alias))
            .find((index) => index >= 0);
    }

    function readStyles(csvText) {
        const rows = parseCsv(csvText);
        if (rows.length < 2) {
            throw new Error("styles.csv has no style rows");
        }

        const headers = rows[0].map(normalizeHeader);
        const nameIndex = findColumn(headers, ["name", "style", "style_name"]);
        const positiveIndex = findColumn(headers, ["prompt", "positive", "positive_prompt"]);
        const negativeIndex = findColumn(headers, [
            "negative_prompt",
            "negative",
            "negitive_prompt",
            "negitive",
        ]);

        if (nameIndex === undefined || positiveIndex === undefined || negativeIndex === undefined) {
            throw new Error(
                "styles.csv must contain name, prompt/positive, and negative_prompt/negative columns"
            );
        }

        return rows
            .slice(1)
            .map((values) => {
                const style = {
                    name: String(values[nameIndex] || "").trim(),
                    positive: String(values[positiveIndex] || "").trim(),
                    negative: String(values[negativeIndex] || "").trim(),
                };
                style.searchText = normalizeSearchText(
                    `${style.name} ${style.positive} ${style.negative}`
                );
                return style;
            })
            .filter((style) => style.name.length > 0);
    }

    function joinPrompts(first, second) {
        const leading = String(first || "").trim();
        const trailing = String(second || "").trim();

        if (!leading) return trailing;
        if (!trailing) return leading;

        const separator = /[,.;:]\s*$/.test(leading) ? " " : ", ";
        return `${leading}${separator}${trailing}`;
    }

    function applyPositiveTemplate(template, currentPrompt) {
        const positive = String(template || "").trim();
        const current = String(currentPrompt || "").trim();

        if (/\{prompt\}/i.test(positive)) {
            return positive
                .replace(/\{prompt\}/gi, current)
                .replace(/[ \t]+([,.;:])/g, "$1")
                .replace(/[ \t]{2,}/g, " ")
                .trim();
        }

        return joinPrompts(positive, current);
    }

    function dispatchFieldEvents(field) {
        field.dispatchEvent(new Event("input", { bubbles: true }));
        field.dispatchEvent(new Event("change", { bubbles: true }));
    }

    function applyStyle(style) {
        const promptField = document.querySelector("#prompt");
        const negativePromptField = document.querySelector("#negative_prompt");

        if (!promptField || !negativePromptField) {
            throw new Error("The main Easy Diffusion prompt fields were not found");
        }

        promptField.value = applyPositiveTemplate(style.positive, promptField.value);
        negativePromptField.value = joinPrompts(style.negative, negativePromptField.value);
        dispatchFieldEvents(promptField);
        dispatchFieldEvents(negativePromptField);

        return promptField;
    }

    function makeElement(tagName, options = {}) {
        const element = document.createElement(tagName);

        if (options.id) element.id = options.id;
        if (options.className) element.className = options.className;
        if (options.text !== undefined) element.textContent = options.text;
        if (options.title) element.title = options.title;
        if (options.type) element.type = options.type;

        Object.entries(options.attributes || {}).forEach(([name, value]) => {
            element.setAttribute(name, value);
        });

        return element;
    }

    function addPluginStyles() {
        if (document.getElementById(`${ID_PREFIX}-css`)) return;

        const style = document.createElement("style");
        style.id = `${ID_PREFIX}-css`;
        style.textContent = `
            #${ID_PREFIX}-nav-divider {
                align-self: center;
                width: 1px;
                height: 1.8rem;
                margin-left: 0.5rem;
                background: var(--background-color3);
                flex: 0 0 auto;
                opacity: 0.9;
            }

            #${ID_PREFIX}-page {
                box-sizing: border-box;
                width: min(1200px, 100%);
                max-width: 1200px;
                padding: 1.25rem;
                text-align: left;
            }

            #${ID_PREFIX}-heading {
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 0.5rem 1rem;
                margin-bottom: 1rem;
            }

            #${ID_PREFIX}-heading h2,
            #${ID_PREFIX}-heading p {
                margin: 0;
            }

            #${ID_PREFIX}-heading p {
                opacity: 0.72;
            }

            #${ID_PREFIX}-search-row {
                position: sticky;
                top: 0;
                z-index: 5;
                display: grid;
                grid-template-columns: minmax(12rem, 1fr) auto;
                gap: 0.6rem;
                align-items: end;
                padding: 0.8rem;
                margin-bottom: 0.75rem;
                border: 1px solid var(--background-color3);
                border-radius: 7px;
                background: var(--background-color2);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
            }

            #${ID_PREFIX}-search-wrap {
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
                min-width: 0;
            }

            #${ID_PREFIX}-search {
                box-sizing: border-box;
                width: 100%;
                min-width: 0;
                height: 2.5rem;
                padding: 0.45rem 0.7rem;
                font-size: 1rem;
            }

            #${ID_PREFIX}-search-meta {
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 0.35rem 1rem;
                min-height: 1.2rem;
                font-size: 0.85rem;
                opacity: 0.78;
            }

            #${ID_PREFIX}-status {
                text-align: right;
            }

            #${ID_PREFIX}-results {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                gap: 0.65rem;
            }

            .${ID_PREFIX}-card {
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                gap: 0.45rem;
                width: 100%;
                min-height: 6.4rem;
                padding: 0.8rem;
                border: 1px solid var(--background-color3);
                border-radius: 7px;
                background: var(--background-color2);
                color: inherit;
                text-align: left;
                cursor: pointer;
                transition: border-color 0.15s, box-shadow 0.15s, transform 0.15s;
            }

            .${ID_PREFIX}-card[hidden],
            #${ID_PREFIX}-results[hidden],
            #${ID_PREFIX}-empty[hidden] {
                display: none !important;
            }

            .${ID_PREFIX}-card:hover,
            .${ID_PREFIX}-card:focus-visible {
                border-color: var(--accent-color, #7e57c2);
                box-shadow: 0 5px 14px rgba(0, 0, 0, 0.18);
                transform: translateY(-1px);
                outline: none;
            }

            .${ID_PREFIX}-card.${ID_PREFIX}-last-applied {
                border-color: var(--accent-color, #7e57c2);
            }

            .${ID_PREFIX}-card-name {
                display: block;
                font-weight: 700;
                line-height: 1.25;
            }

            .${ID_PREFIX}-card-prompt {
                display: -webkit-box;
                overflow: hidden;
                opacity: 0.72;
                font-size: 0.82rem;
                font-weight: 400;
                line-height: 1.35;
                -webkit-box-orient: vertical;
                -webkit-line-clamp: 3;
            }

            #${ID_PREFIX}-empty,
            #${ID_PREFIX}-error {
                padding: 2rem 1rem;
                border: 1px dashed var(--background-color3);
                border-radius: 7px;
                text-align: center;
                opacity: 0.78;
            }

            #${ID_PREFIX}-error {
                color: var(--status-red, #b3261e);
            }

            @media (max-width: 700px) {
                #${ID_PREFIX}-page {
                    padding: 0.65rem;
                }

                #${ID_PREFIX}-search-row {
                    grid-template-columns: 1fr;
                }

                #${ID_PREFIX}-clear {
                    width: 100%;
                }

                #${ID_PREFIX}-results {
                    grid-template-columns: 1fr;
                }
            }
        `;
        document.head.appendChild(style);
    }

    async function loadStyles() {
        const url = new URL(CSV_URL);
        url.searchParams.set("t", String(Date.now()));
        const response = await fetch(url.toString(), { cache: "no-store" });

        if (!response.ok) {
            throw new Error(`Could not load styles.csv (HTTP ${response.status})`);
        }

        const styles = readStyles(await response.text());
        if (styles.length === 0) {
            throw new Error("styles.csv contains no named styles");
        }
        return styles;
    }

    function createTabShell() {
        const mainTab = document.getElementById("tab-main");
        const mainContent = document.getElementById("tab-content-main");
        const tabContentWrapper = document.getElementById("tab-content-wrapper");

        if (!mainTab || !mainContent || !tabContentWrapper) {
            return null;
        }

        const divider = makeElement("span", {
            id: `${ID_PREFIX}-nav-divider`,
            attributes: { "aria-hidden": "true" },
        });

        const tab = makeElement("span", {
            id: TAB_ID,
            className: "tab",
            attributes: { role: "button", tabindex: "0" },
        });
        const tabLabel = makeElement("span");
        const tabIcon = makeElement("i", {
            className: "fa-solid fa-palette icon",
            attributes: { "aria-hidden": "true" },
        });
        tabLabel.append(tabIcon, document.createTextNode(" Styles"));
        tab.appendChild(tabLabel);

        const content = makeElement("div", {
            id: TAB_CONTENT_ID,
            className: "tab-content",
        });
        const page = makeElement("div", {
            id: `${ID_PREFIX}-page`,
            className: "tab-content-inner",
        });
        content.appendChild(page);

        mainTab.insertAdjacentElement("afterend", divider);
        divider.insertAdjacentElement("afterend", tab);
        mainContent.insertAdjacentElement("afterend", content);
        linkTabContents(tab);

        tab.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                tab.click();
            }
        });

        return { tab, page };
    }

    function createStyleBrowser(page, tab, styles) {
        const heading = makeElement("div", { id: `${ID_PREFIX}-heading` });
        const headingText = makeElement("div");
        const title = makeElement("h2", { text: "Prompt Styles" });
        const description = makeElement("p", {
            text: "Search by name or prompt text, then select a style to apply it.",
        });
        headingText.append(title, description);
        heading.appendChild(headingText);

        const searchRow = makeElement("div", { id: `${ID_PREFIX}-search-row` });
        const searchWrap = makeElement("div", { id: `${ID_PREFIX}-search-wrap` });
        const searchLabel = makeElement("label", {
            text: "Search styles",
            attributes: { for: `${ID_PREFIX}-search` },
        });
        const search = makeElement("input", {
            id: `${ID_PREFIX}-search`,
            type: "search",
            attributes: {
                placeholder: `Search ${styles.length.toLocaleString()} styles...`,
                autocomplete: "off",
                spellcheck: "false",
            },
        });
        const searchMeta = makeElement("div", { id: `${ID_PREFIX}-search-meta` });
        const resultCount = makeElement("span", { id: `${ID_PREFIX}-count` });
        const status = makeElement("span", {
            id: `${ID_PREFIX}-status`,
            attributes: { "aria-live": "polite" },
        });
        searchMeta.append(resultCount, status);
        searchWrap.append(searchLabel, search, searchMeta);

        const clearButton = makeElement("button", {
            id: `${ID_PREFIX}-clear`,
            className: "tertiaryButton",
            text: "Clear search",
            type: "button",
        });
        searchRow.append(searchWrap, clearButton);

        const results = makeElement("div", {
            id: `${ID_PREFIX}-results`,
        });
        const empty = makeElement("div", {
            id: `${ID_PREFIX}-empty`,
            text: "No styles match this search.",
        });
        empty.hidden = true;

        const cards = [];
        let lastAppliedCard = null;
        const fragment = document.createDocumentFragment();

        styles.forEach((style) => {
            const card = makeElement("button", {
                className: `${ID_PREFIX}-card`,
                type: "button",
                title: `Apply ${style.name}`,
                attributes: {
                    "aria-label": `Apply style: ${style.name}`,
                },
            });
            const name = makeElement("span", {
                className: `${ID_PREFIX}-card-name`,
                text: style.name,
            });
            const promptPreview = makeElement("span", {
                className: `${ID_PREFIX}-card-prompt`,
                text: style.positive || "(negative prompt only)",
            });
            card.append(name, promptPreview);

            card.addEventListener("click", () => {
                try {
                    const promptField = applyStyle(style);
                    if (lastAppliedCard) {
                        lastAppliedCard.classList.remove(`${ID_PREFIX}-last-applied`);
                    }
                    lastAppliedCard = card;
                    card.classList.add(`${ID_PREFIX}-last-applied`);
                    status.textContent = `Applied: ${style.name}`;

                    const mainTab = document.getElementById("tab-main");
                    if (mainTab) mainTab.click();

                    window.setTimeout(() => {
                        promptField.scrollIntoView({ behavior: "smooth", block: "center" });
                        promptField.focus();
                    }, 80);
                } catch (error) {
                    console.error("[Style Injector]", error);
                    status.textContent = error.message;
                }
            });

            cards.push({ element: card, style });
            fragment.appendChild(card);
        });

        results.appendChild(fragment);
        page.append(heading, searchRow, results, empty);

        let filterFrame = null;
        function filterStyles() {
            filterFrame = null;
            const terms = normalizeSearchText(search.value)
                .trim()
                .split(/\s+/)
                .filter(Boolean);
            let visibleCount = 0;
            let firstVisibleCard = null;

            cards.forEach(({ element, style }) => {
                const matches = terms.every((term) => style.searchText.includes(term));
                element.hidden = !matches;
                if (matches) {
                    visibleCount += 1;
                    if (!firstVisibleCard) firstVisibleCard = element;
                }
            });

            resultCount.textContent = terms.length > 0
                ? `${visibleCount.toLocaleString()} of ${styles.length.toLocaleString()} styles`
                : `${styles.length.toLocaleString()} styles`;
            empty.hidden = visibleCount !== 0;
            results.hidden = visibleCount === 0;
            search.dataset.firstVisibleCard = firstVisibleCard ? "available" : "";
            search._firstVisibleStyleCard = firstVisibleCard;
        }

        function scheduleFilter() {
            if (filterFrame !== null) {
                window.cancelAnimationFrame(filterFrame);
            }
            filterFrame = window.requestAnimationFrame(filterStyles);
        }

        search.addEventListener("input", scheduleFilter);
        search.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                if (filterFrame !== null) {
                    window.cancelAnimationFrame(filterFrame);
                    filterStyles();
                }
                if (!search._firstVisibleStyleCard) return;
                event.preventDefault();
                search._firstVisibleStyleCard.click();
            } else if (event.key === "Escape" && search.value) {
                search.value = "";
                filterStyles();
            }
        });

        clearButton.addEventListener("click", () => {
            search.value = "";
            filterStyles();
            search.focus();
        });

        tab.addEventListener("click", () => {
            window.setTimeout(() => search.focus(), 50);
        });

        filterStyles();
    }

    async function init() {
        if (document.getElementById(TAB_ID)) return;

        const shellReady = document.getElementById("tab-main")
            && document.getElementById("tab-content-main")
            && document.getElementById("tab-content-wrapper")
            && typeof linkTabContents === "function";

        if (!shellReady) {
            window.setTimeout(init, 300);
            return;
        }

        document.getElementById(`${ID_PREFIX}-control`)?.remove();
        addPluginStyles();

        const shell = createTabShell();
        if (!shell) {
            window.setTimeout(init, 300);
            return;
        }

        shell.page.textContent = "Loading styles...";

        try {
            const styles = await loadStyles();
            shell.page.replaceChildren();
            createStyleBrowser(shell.page, shell.tab, styles);
        } catch (error) {
            console.error("[Style Injector]", error);
            shell.page.replaceChildren(
                makeElement("div", {
                    id: `${ID_PREFIX}-error`,
                    text: error.message,
                })
            );
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init, { once: true });
    } else {
        init();
    }
})();
