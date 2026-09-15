;(function (root, factory) {
    const api = factory()
    if (typeof module === "object" && module.exports) {
        module.exports = api
    }
    root.PromptGroups = api
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict"

    const BREAK_AT_END = /(?:^|[^A-Za-z0-9_])BREAK\s*$/i
    const BREAK_AT_START = /^\s*BREAK(?:$|[^A-Za-z0-9_])/i
    const BREAK_IN_GROUP = /(?:^|[^A-Za-z0-9_])BREAK(?:$|[^A-Za-z0-9_])/i

    /**
     * Put the selected concept into its own text-encoder chunk.
     *
     * The returned selection covers only the concept text, not the BREAK
     * boundaries, so a caller can immediately replace or refine the group.
     */
    function isolateConceptSelection(prompt, selectionStart, selectionEnd) {
        if (typeof prompt !== "string") {
            throw new TypeError("prompt must be a string")
        }
        if (!Number.isInteger(selectionStart) || !Number.isInteger(selectionEnd)) {
            throw new TypeError("selection offsets must be integers")
        }
        if (selectionStart < 0 || selectionEnd < selectionStart || selectionEnd > prompt.length) {
            throw new RangeError("selection offsets are outside the prompt")
        }

        const selected = prompt.slice(selectionStart, selectionEnd)
        const leadingWhitespace = selected.length - selected.trimStart().length
        const trailingWhitespace = selected.length - selected.trimEnd().length
        const groupStart = selectionStart + leadingWhitespace
        const groupEnd = selectionEnd - trailingWhitespace
        const group = prompt.slice(groupStart, groupEnd)

        if (group === "") {
            return { error: "empty-selection" }
        }
        if (/\r|\n/.test(group)) {
            return { error: "multi-line-selection" }
        }
        if (BREAK_IN_GROUP.test(group)) {
            return { error: "contains-boundary" }
        }

        const before = prompt.slice(0, groupStart)
        const after = prompt.slice(groupEnd)
        const prefix = BREAK_AT_END.test(before) ? "" : "BREAK "
        const suffix = BREAK_AT_START.test(after) ? "" : " BREAK"
        const value = before + prefix + group + suffix + after
        const isolatedStart = before.length + prefix.length

        return {
            value,
            selectionStart: isolatedStart,
            selectionEnd: isolatedStart + group.length,
        }
    }

    return { isolateConceptSelection }
})
