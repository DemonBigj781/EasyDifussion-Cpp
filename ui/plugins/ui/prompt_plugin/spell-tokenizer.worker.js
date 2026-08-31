/*
 * Background index for the merged Danbooru/e621 spell-tokenizer CSV.
 * The source is headerless: tag, category, post count, quoted aliases.
 */
"use strict"

const CSV_URL = "/plugins/core/prompt_plugin/merged_2024-12-22_pt2-ia-dd-ed.csv"
const buckets = new Map()
let ready = false

function normalize(value) {
    return String(value || "").trim().toLowerCase().replaceAll(" ", "_")
}

function prefix(value) {
    return normalize(value).slice(0, 2)
}

function parseCsvLine(line) {
    const fields = []
    let value = ""
    let quoted = false
    for (let index = 0; index < line.length; index += 1) {
        const character = line[index]
        if (character === '"') {
            if (quoted && line[index + 1] === '"') {
                value += '"'
                index += 1
            } else {
                quoted = !quoted
            }
        } else if (character === "," && !quoted) {
            fields.push(value)
            value = ""
        } else {
            value += character
        }
    }
    fields.push(value)
    return fields
}

function addToBucket(key, entry) {
    if (key.length < 2) return
    let entries = buckets.get(key)
    if (!entries) {
        entries = []
        buckets.set(key, entries)
    }
    entries.push(entry)
}

async function buildIndex() {
    try {
        const response = await fetch(CSV_URL)
        if (!response.ok) throw new Error(`HTTP ${response.status} loading ${CSV_URL}`)
        const text = await response.text()
        let count = 0
        for (const line of text.split(/\r?\n/)) {
            if (!line) continue
            const [tagValue, categoryValue, countValue, aliasesValue = ""] = parseCsvLine(line)
            const tag = normalize(tagValue)
            const postCount = Number.parseInt(countValue, 10)
            const category = Number.parseInt(categoryValue, 10)
            if (!tag || !Number.isFinite(postCount) || !Number.isFinite(category)) continue
            const entry = {
                tag,
                count: postCount.toLocaleString("en-US"),
                category,
                aliases: aliasesValue ? aliasesValue.split(",").map(normalize).filter(Boolean) : [],
            }
            addToBucket(prefix(tag), entry)
            // Aliases are useful for common misspellings, but indexing every
            // alias on a million-row CSV is wasteful. Keep them for established
            // tags and retain all canonical tag names.
            if (postCount >= 1000) {
                new Set(entry.aliases.map(prefix)).forEach((key) => addToBucket(key, entry))
            }
            count += 1
        }
        ready = true
        postMessage({ type: "ready", count })
    } catch (error) {
        postMessage({ type: "error", message: error?.message || String(error) })
    }
}

function queryIndex(queryValue, limitValue) {
    const query = normalize(queryValue)
    const limit = Math.max(1, Math.min(50, Number(limitValue) || 10))
    const results = []
    const seen = new Set()
    for (const entry of buckets.get(prefix(query)) || []) {
        if (seen.has(entry.tag)) continue
        if (entry.tag.startsWith(query) || entry.aliases.some((alias) => alias.startsWith(query))) {
            results.push({ tag: entry.tag, count: entry.count, category: entry.category })
            seen.add(entry.tag)
            if (results.length >= limit) break
        }
    }
    return results
}

self.addEventListener("message", (event) => {
    if (event.data?.type !== "query") return
    const results = ready ? queryIndex(event.data.query, event.data.limit) : []
    postMessage({
        type: "result",
        requestId: event.data.requestId,
        query: event.data.query,
        results,
    })
})

buildIndex()
