/*
 * AT's Custom Modifier Prompts for Easy Diffusion
 * Version 1.0
 * Last Modified: 09/26/2026
 *
 * Extends Easy Diffusion's custom Image Modifiers so the prompt text can be
 * stored in the thumbnail image's Comments metadata instead of being limited
 * to the image filename.
 *
 * Supported formats:
 *
 *   JPG / JPEG
 *     Windows Explorer: Properties > Details > Comments
 *     Stored as EXIF XPComment (UTF-16LE)
 *
 *   PNG
 *     Windows Explorer: Properties > Details > Comments
 *     Stored as PNG iTXt -> XMP -> exif:UserComment
 *
 *     Conventional PNG text fields named Comment or UserComment are also
 *     supported.
 *
 * Behavior:
 *
 *   - Built-in Easy Diffusion modifiers are never changed.
 *   - Existing filename-based custom modifiers continue working unchanged.
 *   - If no supported comment metadata exists, the filename remains the prompt.
 *   - Metadata prompt text is passed through without escaping, sanitizing,
 *     splitting, or rewriting Easy Diffusion prompt syntax.
 *   - Portrait/landscape preview pairing continues to use Easy Diffusion's
 *     existing behavior.
 *
 * Performance:
 *
 *   - Parsed metadata is cached in localStorage.
 *   - HTTP ETags are used to detect changed files automatically.
 *   - Unchanged files do not need their metadata reparsed.
 *   - Changed files are reparsed and the cache is updated.
 *   - Renamed or deleted files are removed from the cache.
 */

(() => {
    "use strict";

    const PLUGIN_NAME = "AT's Custom Modifier Prompts";
    const PLUGIN_VERSION = "1.0";
    const LOG_PREFIX = "[Modifier Metadata]";

    /*
     * Cache schema version 1. Only bump this key if the stored cache format
     * becomes incompatible in a future release.
     */
    const CACHE_KEY =
        "atsCustomModifierPromptsCache_v1";

    const MAX_CONCURRENT_READS = 12;

    const MODIFIER_THUMBNAIL_BASE =
        "media/modifier-thumbnails/";

    const DEBUG = true;

    /*
     * Capture fetch before replacing window.fetch.
     */
    const originalFetch =
        window.fetch.bind(window);


    /* ================================================================
     * Logging
     * ================================================================ */

    function debug(...args) {
        if (DEBUG) {
            console.log(LOG_PREFIX, ...args);
        }
    }

    function warn(...args) {
        console.warn(LOG_PREFIX, ...args);
    }


    /* ================================================================
     * File/path helpers
     * ================================================================ */

    function normalizePreviewPath(path) {
        return String(path)
            .replace(/\\/g, "/");
    }


    function getExtension(path) {
        const match =
            String(path).match(
                /\.([^.?#]+)(?:[?#].*)?$/i
            );

        return match
            ? match[1].toLowerCase()
            : "";
    }


    function isJpeg(path) {
        const extension =
            getExtension(path);

        return (
            extension === "jpg" ||
            extension === "jpeg"
        );
    }


    function isPng(path) {
        return (
            getExtension(path) === "png"
        );
    }


    function isSupportedImage(path) {
        return (
            isJpeg(path) ||
            isPng(path)
        );
    }


    function previewPathToUrl(path) {
        return new URL(
            MODIFIER_THUMBNAIL_BASE + path,
            window.location.href
        ).href;
    }


    /* ================================================================
     * Cache
     * ================================================================ */

    function parseStoredCache(key) {
        try {
            const raw =
                localStorage.getItem(key);

            if (!raw) {
                return null;
            }

            const parsed =
                JSON.parse(raw);

            if (
                !parsed ||
                typeof parsed !== "object" ||
                Array.isArray(parsed)
            ) {
                return null;
            }

            return parsed;

        } catch (error) {
            warn(
                `Unable to read cache "${key}".`,
                error
            );

            return null;
        }
    }


    function loadCache() {
        return (
            parseStoredCache(CACHE_KEY) ||
            {}
        );
    }


    let metadataCache =
        loadCache();


    function saveCache() {
        try {
            localStorage.setItem(
                CACHE_KEY,
                JSON.stringify(metadataCache)
            );

        } catch (error) {
            warn(
                "Unable to save metadata cache. " +
                "Metadata will still work, but caching may not persist.",
                error
            );
        }
    }


    /*
     * Diagnostic helpers available from F12 > Console.
     */
    window.ModifierMetadata = {

        name: PLUGIN_NAME,
        version: PLUGIN_VERSION,

        clearCache() {

            metadataCache = {};

            localStorage.removeItem(
                CACHE_KEY
            );

            console.log(
                LOG_PREFIX,
                "Metadata cache cleared. Reload the page to rescan modifier images."
            );
        },


        showCache() {

            console.log(
                LOG_PREFIX,
                metadataCache
            );

            return metadataCache;
        }
    };


    /* ================================================================
     * Easy Diffusion request detection
     * ================================================================ */

    function isModifiersRequest(input) {
        try {
            const rawUrl =
                input instanceof Request
                    ? input.url
                    : String(input);

            const url =
                new URL(
                    rawUrl,
                    window.location.href
                );

            return (
                url.pathname === "/get/modifiers" ||
                url.pathname.endsWith(
                    "/get/modifiers"
                )
            );

        } catch (error) {
            return false;
        }
    }


    /* ================================================================
     * Modifier preview detection
     * ================================================================ */

    function isCustomPreview(preview) {
        if (
            !preview ||
            typeof preview.path !== "string"
        ) {
            return false;
        }

        return normalizePreviewPath(
            preview.path
        ).startsWith("custom/");
    }


    function getCustomPreviewPaths(
        modifier
    ) {
        if (
            !modifier ||
            !Array.isArray(
                modifier.previews
            )
        ) {
            return [];
        }


        return modifier.previews
            .filter(isCustomPreview)
            .map(
                (preview) =>
                    normalizePreviewPath(
                        preview.path
                    )
            )
            .filter(
                isSupportedImage
            );
    }


    /* ================================================================
     * General text helpers
     * ================================================================ */

    function cleanMetadataText(text) {

        if (
            typeof text !== "string"
        ) {
            return null;
        }


        /*
         * Remove encoding artifacts only.
         *
         * Do NOT trim or otherwise rewrite valid prompt content.
         */
        const cleaned =
            text
                .replace(/\u0000+$/g, "")
                .replace(/^\uFEFF/, "");


        return (
            cleaned.trim().length > 0
        )
            ? cleaned
            : null;
    }


    function normalizedTagName(name) {

        /*
         * Handles:
         *
         *   UserComment
         *   exif:UserComment
         *   user_comment
         *   user-comment
         */

        const localName =
            String(name)
                .split(":")
                .pop();


        return localName
            .replace(
                /[\s_-]/g,
                ""
            )
            .toLowerCase();
    }


    /* ================================================================
     * JPEG XPComment decoding
     * ================================================================ */

    function decodeUtf16LE(bytes) {

        if (
            !bytes ||
            bytes.length === 0
        ) {
            return null;
        }


        try {

            const decoded =
                new TextDecoder(
                    "utf-16le"
                ).decode(bytes);


            return cleanMetadataText(
                decoded
            );

        } catch (error) {

            return null;
        }
    }


    /*
     * Handles XPComment values exposed as comma-separated UTF-16LE bytes:
     *
     *   83, 0, 116, 0, 121, 0, 108, 0, 101, 0...
     *
     * which is UTF-16LE for:
     *
     *   Style...
     */
    function numericStringToBytes(value) {

        if (
            typeof value !== "string"
        ) {
            return null;
        }


        let candidate =
            value.trim();


        if (
            candidate.startsWith("[") &&
            candidate.endsWith("]")
        ) {
            candidate =
                candidate
                    .slice(1, -1)
                    .trim();
        }


        if (
            !/^\d{1,3}(?:\s*,\s*\d{1,3})+$/.test(
                candidate
            )
        ) {
            return null;
        }


        const values =
            candidate
                .split(",")
                .map(
                    (item) =>
                        Number(
                            item.trim()
                        )
                );


        if (
            values.length === 0 ||
            values.some(
                (value) =>
                    !Number.isInteger(value) ||
                    value < 0 ||
                    value > 255
            )
        ) {
            return null;
        }


        return Uint8Array.from(
            values
        );
    }


    function valueToBytes(value) {

        if (
            value instanceof Uint8Array
        ) {
            return value;
        }


        if (
            value instanceof ArrayBuffer
        ) {
            return new Uint8Array(
                value
            );
        }


        if (
            ArrayBuffer.isView(value)
        ) {
            return new Uint8Array(
                value.buffer,
                value.byteOffset,
                value.byteLength
            );
        }


        if (
            Array.isArray(value)
        ) {

            if (
                value.length > 0 &&
                value.every(
                    (item) =>
                        Number.isInteger(
                            Number(item)
                        ) &&
                        Number(item) >= 0 &&
                        Number(item) <= 255
                )
            ) {
                return Uint8Array.from(
                    value.map(Number)
                );
            }


            if (
                value.length === 1 &&
                typeof value[0] === "string"
            ) {
                return numericStringToBytes(
                    value[0]
                );
            }
        }


        return numericStringToBytes(
            value
        );
    }


    function decodeXpCommentCandidate(
        value
    ) {

        if (
            value === undefined ||
            value === null
        ) {
            return null;
        }


        const bytes =
            valueToBytes(value);


        if (bytes) {
            return decodeUtf16LE(
                bytes
            );
        }


        if (
            typeof value === "string"
        ) {
            return cleanMetadataText(
                value
            );
        }


        return null;
    }


    function findXpCommentTag(tags) {

        if (
            !tags ||
            typeof tags !== "object"
        ) {
            return null;
        }


        if (tags.XPComment) {
            return tags.XPComment;
        }


        const key =
            Object.keys(tags).find(
                (name) =>
                    normalizedTagName(
                        name
                    ) === "xpcomment"
            );


        return key
            ? tags[key]
            : null;
    }


    function getJpegComment(tags) {

        const tag =
            findXpCommentTag(tags);


        if (!tag) {
            return null;
        }


        /*
         * Prefer the raw value.
         */
        const fromValue =
            decodeXpCommentCandidate(
                tag.value
            );


        if (fromValue !== null) {
            return fromValue;
        }


        /*
         * Older/bundled ExifReader versions may expose the byte list
         * through description instead.
         */
        return decodeXpCommentCandidate(
            tag.description
        );
    }


    /* ================================================================
     * PNG chunk parsing
     * ================================================================ */

    const UTF8_DECODER =
        new TextDecoder("utf-8");

    const LATIN1_DECODER =
        new TextDecoder("latin1");


    function bytesToAscii(
        bytes,
        start,
        length
    ) {
        let result = "";

        for (
            let i = 0;
            i < length;
            i++
        ) {
            result += String.fromCharCode(
                bytes[start + i]
            );
        }

        return result;
    }


    function findNullByte(
        bytes,
        start
    ) {
        for (
            let i = start;
            i < bytes.length;
            i++
        ) {
            if (
                bytes[i] === 0
            ) {
                return i;
            }
        }

        return -1;
    }


    function isPngSignature(bytes) {

        const signature = [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A
        ];


        if (
            bytes.length <
            signature.length
        ) {
            return false;
        }


        return signature.every(
            (value, index) =>
                bytes[index] === value
        );
    }


    /*
     * Extract Windows Explorer's PNG Comments value from XMP UserComment.
     * Explorer commonly stores it as an rdf:Alt value with x-default text.
     */
    function extractUserCommentFromXmp(
        xmlText
    ) {

        try {

            const parser =
                new DOMParser();


            const document =
                parser.parseFromString(
                    xmlText,
                    "application/xml"
                );


            if (
                document.querySelector(
                    "parsererror"
                )
            ) {
                return null;
            }


            const elements =
                document.getElementsByTagName(
                    "*"
                );


            for (
                const element
                of elements
            ) {

                if (
                    normalizedTagName(
                        element.localName ||
                        element.nodeName
                    ) !==
                    "usercomment"
                ) {
                    continue;
                }


                /*
                 * Prefer rdf:li xml:lang="x-default".
                 */
                const descendants =
                    element.getElementsByTagName(
                        "*"
                    );


                let firstListItem =
                    null;


                for (
                    const child
                    of descendants
                ) {

                    if (
                        normalizedTagName(
                            child.localName ||
                            child.nodeName
                        ) !== "li"
                    ) {
                        continue;
                    }


                    if (!firstListItem) {
                        firstListItem =
                            child;
                    }


                    const language =
                        child.getAttribute(
                            "xml:lang"
                        );


                    if (
                        language &&
                        language.toLowerCase() ===
                            "x-default"
                    ) {
                        return cleanMetadataText(
                            child.textContent
                        );
                    }
                }


                /*
                 * If there was no x-default item, use the first rdf:li.
                 */
                if (firstListItem) {
                    return cleanMetadataText(
                        firstListItem.textContent
                    );
                }


                /*
                 * Final fallback: text directly inside UserComment.
                 */
                return cleanMetadataText(
                    element.textContent
                );
            }


            return null;

        } catch (error) {

            return null;
        }
    }


    /*
     * Directly inspect PNG tEXt / iTXt chunks.
     *
     * This avoids depending entirely on the exact ExifReader version
     * bundled with Easy Diffusion.
     */
    function getPngCommentFromChunks(
        bytes
    ) {

        if (
            !isPngSignature(bytes)
        ) {
            return null;
        }


        const view =
            new DataView(
                bytes.buffer,
                bytes.byteOffset,
                bytes.byteLength
            );


        let offset = 8;


        while (
            offset + 12 <=
            bytes.length
        ) {

            const length =
                view.getUint32(
                    offset,
                    false
                );


            const type =
                bytesToAscii(
                    bytes,
                    offset + 4,
                    4
                );


            const dataStart =
                offset + 8;

            const dataEnd =
                dataStart + length;


            if (
                dataEnd + 4 >
                bytes.length
            ) {
                break;
            }


            const data =
                bytes.subarray(
                    dataStart,
                    dataEnd
                );


            /*
             * --------------------------------------------------------
             * Standard PNG tEXt
             *
             * keyword\0text
             * --------------------------------------------------------
             */
            if (
                type === "tEXt"
            ) {

                const separator =
                    findNullByte(
                        data,
                        0
                    );


                if (
                    separator > 0
                ) {

                    const keyword =
                        LATIN1_DECODER.decode(
                            data.subarray(
                                0,
                                separator
                            )
                        );


                    const normalized =
                        normalizedTagName(
                            keyword
                        );


                    if (
                        normalized === "comment" ||
                        normalized === "usercomment"
                    ) {

                        const value =
                            LATIN1_DECODER.decode(
                                data.subarray(
                                    separator + 1
                                )
                            );


                        const result =
                            cleanMetadataText(
                                value
                            );


                        if (
                            result !== null
                        ) {
                            return result;
                        }
                    }
                }
            }


            /*
             * --------------------------------------------------------
             * PNG iTXt
             *
             * keyword
             * null
             * compression flag
             * compression method
             * language tag
             * null
             * translated keyword
             * null
             * UTF-8 text
             *
             * Windows 11 Explorer uses:
             *
             *   keyword:
             *       XML:com.adobe.xmp
             *
             *   compression:
             *       0 (uncompressed)
             *
             *   text:
             *       XMP XML containing exif:UserComment
             * --------------------------------------------------------
             */
            if (
                type === "iTXt"
            ) {

                const keywordEnd =
                    findNullByte(
                        data,
                        0
                    );


                if (
                    keywordEnd > 0 &&
                    keywordEnd + 2 <
                        data.length
                ) {

                    const keyword =
                        LATIN1_DECODER.decode(
                            data.subarray(
                                0,
                                keywordEnd
                            )
                        );


                    let position =
                        keywordEnd + 1;


                    const compressionFlag =
                        data[position++];

                    /*
                     * Compression method.
                     */
                    position++;


                    const languageEnd =
                        findNullByte(
                            data,
                            position
                        );


                    if (
                        languageEnd === -1
                    ) {
                        offset =
                            dataEnd + 4;

                        continue;
                    }


                    position =
                        languageEnd + 1;


                    const translatedEnd =
                        findNullByte(
                            data,
                            position
                        );


                    if (
                        translatedEnd === -1
                    ) {
                        offset =
                            dataEnd + 4;

                        continue;
                    }


                    position =
                        translatedEnd + 1;


                    /*
                     * Windows' XMP iTXt is uncompressed.
                     *
                     * If compressed iTXt is encountered, let ExifReader
                     * handle it later as a fallback.
                     */
                    if (
                        compressionFlag === 0 &&
                        position <=
                            data.length
                    ) {

                        const text =
                            UTF8_DECODER.decode(
                                data.subarray(
                                    position
                                )
                            );


                        const normalized =
                            normalizedTagName(
                                keyword
                            );


                        /*
                         * Conventional direct PNG Comment field.
                         */
                        if (
                            normalized === "comment" ||
                            normalized === "usercomment"
                        ) {

                            const result =
                                cleanMetadataText(
                                    text
                                );


                            if (
                                result !== null
                            ) {
                                return result;
                            }
                        }


                        /*
                         * Windows 11 Explorer XMP packet.
                         */
                        if (
                            keyword
                                .toLowerCase() ===
                            "xml:com.adobe.xmp"
                        ) {

                            const result =
                                extractUserCommentFromXmp(
                                    text
                                );


                            if (
                                result !== null
                            ) {
                                return result;
                            }
                        }
                    }
                }
            }


            offset =
                dataEnd + 4;
        }


        return null;
    }


    /* ================================================================
     * PNG ExifReader fallback
     * ================================================================ */

    function extractSimpleTagText(
        value,
        depth = 0
    ) {

        /*
         * Avoid pathological recursive structures.
         */
        if (depth > 5) {
            return null;
        }


        if (
            value === undefined ||
            value === null
        ) {
            return null;
        }


        if (
            typeof value === "string"
        ) {
            return cleanMetadataText(
                value
            );
        }


        if (
            Array.isArray(value)
        ) {

            for (
                const item
                of value
            ) {

                const result =
                    extractSimpleTagText(
                        item,
                        depth + 1
                    );


                if (
                    result !== null
                ) {
                    return result;
                }
            }


            return null;
        }


        if (
            typeof value === "object"
        ) {

            /*
             * Common ExifReader tag representations.
             */
            if (
                Object.prototype.hasOwnProperty.call(
                    value,
                    "value"
                )
            ) {

                const result =
                    extractSimpleTagText(
                        value.value,
                        depth + 1
                    );


                if (
                    result !== null
                ) {
                    return result;
                }
            }


            if (
                Object.prototype.hasOwnProperty.call(
                    value,
                    "description"
                )
            ) {

                const result =
                    extractSimpleTagText(
                        value.description,
                        depth + 1
                    );


                if (
                    result !== null
                ) {
                    return result;
                }
            }


            /*
             * XMP alternative-language representation.
             */
            if (
                Object.prototype.hasOwnProperty.call(
                    value,
                    "x-default"
                )
            ) {

                const result =
                    extractSimpleTagText(
                        value["x-default"],
                        depth + 1
                    );


                if (
                    result !== null
                ) {
                    return result;
                }
            }
        }


        return null;
    }


    function findPngCommentTag(
        tags
    ) {

        if (
            !tags ||
            typeof tags !== "object"
        ) {
            return null;
        }


        /*
         * First preference:
         *
         * Windows/XMP UserComment.
         */
        if (tags.UserComment) {
            return tags.UserComment;
        }


        /*
         * Second preference:
         *
         * Conventional PNG Comment.
         */
        if (tags.Comment) {
            return tags.Comment;
        }


        /*
         * Allow namespace/capitalization differences.
         */
        const directKey =
            Object.keys(tags).find(
                (key) => {

                    const normalized =
                        normalizedTagName(
                            key
                        );


                    return (
                        normalized ===
                            "usercomment" ||
                        normalized ===
                            "comment"
                    );
                }
            );


        if (directKey) {
            return tags[
                directKey
            ];
        }


        /*
         * Some ExifReader configurations may group XMP/PNG tags.
         */
        const groups = [
            "xmp",
            "XMP",
            "pngText",
            "PNG Text",
            "PngText",
            "png"
        ];


        for (
            const groupName
            of groups
        ) {

            const group =
                tags[groupName];


            if (
                !group ||
                typeof group !==
                    "object"
            ) {
                continue;
            }


            const key =
                Object.keys(group).find(
                    (name) => {

                        const normalized =
                            normalizedTagName(
                                name
                            );


                        return (
                            normalized ===
                                "usercomment" ||
                            normalized ===
                                "comment"
                        );
                    }
                );


            if (key) {
                return group[
                    key
                ];
            }
        }


        return null;
    }


    function getPngCommentFromExifReader(
        tags
    ) {

        const tag =
            findPngCommentTag(
                tags
            );


        if (!tag) {
            return null;
        }


        return extractSimpleTagText(
            tag
        );
    }


    /* ================================================================
     * Parse image metadata
     * ================================================================ */

    async function parseCommentFromResponse(
        response,
        path
    ) {

        const arrayBuffer =
            await response.arrayBuffer();


        const bytes =
            new Uint8Array(
                arrayBuffer
            );


        /*
         * ------------------------------------------------------------
         * JPEG
         * ------------------------------------------------------------
         */
        if (isJpeg(path)) {

            if (
                typeof ExifReader ===
                    "undefined" ||
                typeof ExifReader.load !==
                    "function"
            ) {
                throw new Error(
                    "Easy Diffusion's bundled ExifReader library is unavailable."
                );
            }


            const tags =
                ExifReader.load(
                    arrayBuffer
                );


            return getJpegComment(
                tags
            );
        }


        /*
         * ------------------------------------------------------------
         * PNG
         * ------------------------------------------------------------
         */
        if (isPng(path)) {

            /*
             * First inspect the actual PNG text/iTXt chunks ourselves.
             *
             * This directly supports the XMP UserComment format written by
             * Windows Explorer for PNG Comments metadata.
             */
            const directComment =
                getPngCommentFromChunks(
                    bytes
                );


            if (
                directComment !== null
            ) {
                return directComment;
            }


            /*
             * Then fall back to Easy Diffusion's bundled ExifReader.
             */
            if (
                typeof ExifReader !==
                    "undefined" &&
                typeof ExifReader.load ===
                    "function"
            ) {

                try {

                    const tags =
                        ExifReader.load(
                            arrayBuffer
                        );


                    const fallback =
                        getPngCommentFromExifReader(
                            tags
                        );


                    if (
                        fallback !== null
                    ) {
                        return fallback;
                    }

                } catch (error) {

                    debug(
                        `ExifReader PNG fallback could not parse ${path}:`,
                        error
                    );
                }
            }


            return null;
        }


        return null;
    }


    /* ================================================================
     * ETag-aware metadata loading
     * ================================================================ */

    async function getMetadataForPreview(
        path
    ) {

        const cached =
            metadataCache[path] ||
            null;


        const imageUrl =
            previewPathToUrl(path);


        const headers =
            new Headers();


        /*
         * Ask Easy Diffusion whether the cached file is still current.
         */
        if (
            cached &&
            typeof cached.etag ===
                "string" &&
            cached.etag.length > 0
        ) {

            headers.set(
                "If-None-Match",
                cached.etag
            );
        }


        const response =
            await originalFetch(
                imageUrl,
                {
                    method: "GET",
                    headers,

                    /*
                     * Force validation rather than blindly accepting a
                     * stale browser cache entry.
                     */
                    cache: "no-cache"
                }
            );


        /*
         * Some browser/server combinations expose the 304 directly.
         */
        if (
            response.status === 304 &&
            cached
        ) {

            return {
                comment:
                    cached.comment ??
                    null,

                result:
                    "not-modified"
            };
        }


        if (!response.ok) {

            throw new Error(
                `HTTP ${response.status} while checking ${path}`
            );
        }


        const currentEtag =
            response.headers.get(
                "etag"
            );


        const currentLastModified =
            response.headers.get(
                "last-modified"
            );


        /*
         * Chromium may internally satisfy a 304 and expose a normal
         * 200 response backed by cache.
         *
         * Matching ETags still prove the file is unchanged.
         */
        if (
            cached &&
            cached.etag &&
            currentEtag &&
            cached.etag ===
                currentEtag
        ) {

            return {
                comment:
                    cached.comment ??
                    null,

                result:
                    "etag-match"
            };
        }


        /*
         * Last-Modified fallback when no ETag is available.
         */
        if (
            cached &&
            !currentEtag &&
            cached.lastModified &&
            currentLastModified &&
            cached.lastModified ===
                currentLastModified
        ) {

            return {
                comment:
                    cached.comment ??
                    null,

                result:
                    "last-modified-match"
            };
        }


        /*
         * New or changed file.
         */
        const comment =
            await parseCommentFromResponse(
                response,
                path
            );


        metadataCache[path] = {

            etag:
                currentEtag ||
                null,

            lastModified:
                currentLastModified ||
                null,

            comment:
                comment
        };


        return {

            comment,

            result:
                cached
                    ? "changed"
                    : "new"
        };
    }


    /* ================================================================
     * Concurrency
     * ================================================================ */

    async function runWithConcurrency(
        items,
        limit,
        worker
    ) {

        if (
            items.length === 0
        ) {
            return;
        }


        let nextIndex = 0;


        const workerCount =
            Math.min(
                limit,
                items.length
            );


        const workers =
            Array.from(
                {
                    length:
                        workerCount
                },

                async () => {

                    while (true) {

                        const index =
                            nextIndex++;


                        if (
                            index >=
                            items.length
                        ) {
                            break;
                        }


                        await worker(
                            items[index]
                        );
                    }
                }
            );


        await Promise.all(
            workers
        );
    }


    /* ================================================================
     * Modifier processing
     * ================================================================ */

    async function applyMetadataModifiers(
        modifierGroups
    ) {

        if (
            !Array.isArray(
                modifierGroups
            )
        ) {
            return;
        }


        const customModifiers =
            [];

        const activePaths =
            new Set();


        /*
         * Collect ONLY filesystem custom JPG/JPEG/PNG modifiers.
         *
         * Built-in modifiers never enter this list.
         */
        for (
            const group
            of modifierGroups
        ) {

            if (
                !group ||
                !Array.isArray(
                    group.modifiers
                )
            ) {
                continue;
            }


            for (
                const modifier
                of group.modifiers
            ) {

                const previewPaths =
                    getCustomPreviewPaths(
                        modifier
                    );


                if (
                    previewPaths.length ===
                    0
                ) {
                    continue;
                }


                for (
                    const path
                    of previewPaths
                ) {
                    activePaths.add(
                        path
                    );
                }


                customModifiers.push({
                    modifier,
                    previewPaths
                });
            }
        }


        /*
         * Remove deleted/renamed files from the cache.
         */
        let cacheChanged =
            false;


        for (
            const cachedPath
            of Object.keys(
                metadataCache
            )
        ) {

            if (
                !activePaths.has(
                    cachedPath
                )
            ) {

                delete metadataCache[
                    cachedPath
                ];

                cacheChanged =
                    true;
            }
        }


        const stats = {

            customModifiers:
                customModifiers.length,

            jpegPreviews:
                0,

            pngPreviews:
                0,

            metadataApplied:
                0,

            unchanged304:
                0,

            unchangedETag:
                0,

            unchangedLastModified:
                0,

            newFiles:
                0,

            changedFiles:
                0,

            filenameFallback:
                0,

            errors:
                0
        };


        for (
            const item
            of customModifiers
        ) {

            for (
                const path
                of item.previewPaths
            ) {

                if (isJpeg(path)) {
                    stats.jpegPreviews++;
                }


                if (isPng(path)) {
                    stats.pngPreviews++;
                }
            }
        }


        debug(
            `Checking ${stats.customModifiers} custom modifier(s): ` +
            `${stats.jpegPreviews} JPEG preview(s), ` +
            `${stats.pngPreviews} PNG preview(s).`
        );


        await runWithConcurrency(
            customModifiers,
            MAX_CONCURRENT_READS,

            async ({
                modifier,
                previewPaths
            }) => {

                const originalName =
                    modifier.modifier;


                let foundMetadata =
                    false;


                /*
                 * Portrait/landscape pairs can contain multiple previews.
                 *
                 * The first preview containing usable metadata wins.
                 */
                for (
                    const path
                    of previewPaths
                ) {

                    try {

                        const result =
                            await getMetadataForPreview(
                                path
                            );


                        switch (
                            result.result
                        ) {

                            case "not-modified":

                                stats.unchanged304++;

                                break;


                            case "etag-match":

                                stats.unchangedETag++;

                                break;


                            case "last-modified-match":

                                stats.unchangedLastModified++;

                                break;


                            case "new":

                                stats.newFiles++;

                                cacheChanged =
                                    true;

                                break;


                            case "changed":

                                stats.changedFiles++;

                                cacheChanged =
                                    true;

                                break;
                        }


                        if (
                            typeof result.comment ===
                                "string" &&
                            result.comment.trim().length >
                                0
                        ) {

                            /*
                             * IMPORTANT:
                             *
                             * Metadata becomes the modifier prompt exactly
                             * as written.
                             *
                             * No escaping.
                             * No filename sanitization.
                             * No Easy Diffusion syntax processing here.
                             */
                            modifier.modifier =
                                result.comment;


                            stats.metadataApplied++;


                            foundMetadata =
                                true;


                            debug(
                                `${result.result}: ` +
                                `${isPng(path) ? "PNG Comment" : "XPComment"} ` +
                                `for "${originalName}" from ${path}:`,
                                result.comment
                            );


                            break;
                        }

                    } catch (error) {

                        stats.errors++;


                        warn(
                            `Unable to check metadata for ${path}. ` +
                            `Keeping Easy Diffusion's normal filename-derived modifier.`,
                            error
                        );
                    }
                }


                if (
                    !foundMetadata
                ) {

                    /*
                     * Do nothing.
                     *
                     * modifier.modifier therefore remains exactly what
                     * Easy Diffusion originally derived from the filename.
                     */
                    stats.filenameFallback++;
                }
            }
        );


        if (cacheChanged) {
            saveCache();
        }


        debug(
            "Metadata check complete:",
            stats
        );
    }


    /* ================================================================
     * Fetch interception
     * ================================================================ */

    window.fetch =
        async function modifierMetadataFetch(
            input,
            init
        ) {

            const response =
                await originalFetch(
                    input,
                    init
                );


            /*
             * Every request except /get/modifiers behaves normally.
             */
            if (
                !isModifiersRequest(
                    input
                ) ||
                !response.ok
            ) {
                return response;
            }


            try {

                /*
                 * Work on a clone so the untouched ED response remains
                 * available if anything unexpected occurs.
                 */
                const modifierGroups =
                    await response
                        .clone()
                        .json();


                await applyMetadataModifiers(
                    modifierGroups
                );


                return new Response(
                    JSON.stringify(
                        modifierGroups
                    ),

                    {
                        status:
                            response.status,

                        statusText:
                            response.statusText,

                        headers: {
                            "Content-Type":
                                "application/json; charset=utf-8"
                        }
                    }
                );

            } catch (error) {

                warn(
                    "Modifier metadata processing failed. " +
                    "Returning Easy Diffusion's original modifier data unchanged.",
                    error
                );


                return response;
            }
        };


    debug(
        `${PLUGIN_NAME} v${PLUGIN_VERSION} loaded. ` +
        "Waiting for Easy Diffusion to request /get/modifiers."
    );


    /* ================================================================
     * Modifier panel cache/rescan button
     * ================================================================ */

    function showMetadataRescanConfirmation() {

        /*
         * Return a Promise<boolean>:
         *
         * true  = user chose RESCAN & RELOAD
         * false = user cancelled
         */
        return new Promise((resolve) => {

            /*
             * Don't allow duplicate confirmation dialogs.
             */
            const existing =
                document.getElementById(
                    "modifier-metadata-confirm-overlay"
                );

            if (existing) {
                existing.remove();
            }


            /*
             * Full-screen dimmed overlay.
             */
            const overlay =
                document.createElement("div");

            overlay.id =
                "modifier-metadata-confirm-overlay";

            /*
             * Prevent Easy Diffusion's "click outside the modifier panel"
             * handler from closing the Image Modifiers flyout when interacting
             * with this confirmation dialog.
             */
            overlay.addEventListener(
                "mousedown",
                (event) => {
                    event.stopPropagation();
                }
            );

            Object.assign(
                overlay.style,
                {
                    position: "fixed",
                    left: "0",
                    top: "0",
                    width: "100%",
                    height: "100%",
                    background: "rgba(0, 0, 0, 0.65)",
                    zIndex: "100000",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "20px",
                    boxSizing: "border-box"
                }
            );


            /*
             * Dialog itself.
             */
            const dialog =
                document.createElement("div");


            Object.assign(
                dialog.style,
                {
                    width: "min(650px, 95vw)",
                    background: "#252b37",
                    color: "#e8e8e8",
                    border: "1px solid #4b5363",
                    borderRadius: "8px",
                    boxShadow: "0 12px 40px rgba(0, 0, 0, 0.55)",
                    padding: "26px 30px 24px 30px",
                    boxSizing: "border-box",
                    textAlign: "center"
                }
            );


            /*
             * Prominent warning/title.
             *
             * This is intentionally much more obvious than the old
             * confirmation wording because clicking the button reloads
             * the entire Easy Diffusion page.
             */
            const title =
                document.createElement("div");


            title.innerHTML =
                "<strong>Reload page and rescan all custom modifier metadata?</strong>";


            Object.assign(
                title.style,
                {
                    color: "#ff6464",
                    fontSize: "20px",
                    lineHeight: "1.35",
                    marginBottom: "18px"
                }
            );


            /*
             * Explanation.
             */
            const message =
                document.createElement("div");


            message.innerHTML =
                "This will clear AT's Custom Modifier Prompts cache and perform " +
                "a full JPG/PNG metadata scan when Easy Diffusion reloads." +
                "<br><br>" +
                "<strong>The current Easy Diffusion page will reload immediately.</strong>" +
                "<br>" +
                "Any unsaved prompt/settings changes on the current page may be lost.";


            Object.assign(
                message.style,
                {
                    fontSize: "15px",
                    lineHeight: "1.55",
                    marginBottom: "26px"
                }
            );


            /*
             * Button container.
             */
            const buttons =
                document.createElement("div");


            Object.assign(
                buttons.style,
                {
                    display: "flex",
                    justifyContent: "center",
                    gap: "14px",
                    flexWrap: "wrap"
                }
            );


            /*
             * Confirm button.
             */
            const confirmButton =
                document.createElement("button");


            confirmButton.type =
                "button";


            confirmButton.textContent =
                "RESCAN & RELOAD";


            /*
             * Use ED's button class as a base, but give this destructive/
             * disruptive action a clear warning appearance.
             */
            confirmButton.className =
                "primaryButton";


            Object.assign(
                confirmButton.style,
                {
                    background: "#b33a3a",
                    borderColor: "#d65252",
                    color: "#ffffff",
                    fontWeight: "bold",
                    minWidth: "165px",
                    padding: "10px 18px",
                    cursor: "pointer"
                }
            );


            /*
             * Cancel button.
             */
            const cancelButton =
                document.createElement("button");


            cancelButton.type =
                "button";


            cancelButton.textContent =
                "CANCEL";


            cancelButton.className =
                "tertiaryButton";


            Object.assign(
                cancelButton.style,
                {
                    minWidth: "120px",
                    padding: "10px 18px",
                    cursor: "pointer"
                }
            );


            function close(result) {

                document.removeEventListener(
                    "keydown",
                    keyHandler
                );


                overlay.remove();


                resolve(result);
            }


            /*
             * Confirm.
             */
            confirmButton.addEventListener(
                "click",
                (event) => {

                    event.preventDefault();
                    event.stopPropagation();

                    close(true);
                }
            );


            /*
             * Cancel.
             */
            cancelButton.addEventListener(
                "click",
                (event) => {

                    event.preventDefault();
                    event.stopPropagation();

                    close(false);
                }
            );


            /*
             * Escape = Cancel.
             */
            function keyHandler(event) {

                if (event.key === "Escape") {
                    event.preventDefault();
                    close(false);
                }
            }


            document.addEventListener(
                "keydown",
                keyHandler
            );


            /*
             * Clicking the dark background also cancels.
             * Clicking inside the dialog does not.
             */
            overlay.addEventListener(
                "click",
                (event) => {

                    if (event.target === overlay) {
                        close(false);
                    }
                }
            );


            buttons.appendChild(
                confirmButton
            );


            buttons.appendChild(
                cancelButton
            );


            dialog.appendChild(
                title
            );


            dialog.appendChild(
                message
            );


            dialog.appendChild(
                buttons
            );


            overlay.appendChild(
                dialog
            );


            document.body.appendChild(
                overlay
            );


            /*
             * Put keyboard focus on Cancel by default.
             *
             * This prevents somebody from accidentally confirming simply
             * by hitting Enter immediately after opening the dialog.
             */
            cancelButton.focus();
        });
    }



    function addMetadataRescanButton() {

        /*
         * Don't accidentally add it twice.
         */
        if (
            document.getElementById(
                "modifier-metadata-rescan-btn"
            )
        ) {
            return;
        }


        const headerRight =
            document.querySelector(
                "#modifiers-header-right"
            );


        if (!headerRight) {

            warn(
                "Could not find #modifiers-header-right; " +
                "metadata rescan button was not added."
            );

            return;
        }


        const button =
            document.createElement("button");


        button.id =
            "modifier-metadata-rescan-btn";


        button.className =
            "tertiaryButton smallButton";


        button.type =
            "button";


        button.innerHTML =
            '<i class="fa-solid fa-arrows-rotate"></i> Rescan Metadata';


        button.title =
            "Clear AT's Custom Modifier Prompts cache, reload Easy Diffusion, " +
            "and perform a fresh JPG/PNG metadata scan.";


        button.style.marginRight =
            "12px";


        button.style.whiteSpace =
            "nowrap";


        /*
         * Place it before the existing Settings gear.
         */
        headerRight.prepend(
            button
        );


        button.addEventListener(
            "click",

            async (event) => {

                event.preventDefault();
                event.stopPropagation();


                const confirmed =
                    await showMetadataRescanConfirmation();


                if (!confirmed) {
                    return;
                }


                /*
                 * Clear our metadata cache.
                 */
                if (
                    window.ModifierMetadata &&
                    typeof window.ModifierMetadata.clearCache ===
                        "function"
                ) {

                    window.ModifierMetadata.clearCache();

                } else {

                    warn(
                        "ModifierMetadata.clearCache() was unavailable."
                    );

                    return;
                }


                /*
                 * Reload immediately.
                 *
                 * On reload, the plugin sees an empty cache and performs
                 * a fresh scan of all supported custom modifiers.
                 */
                window.location.reload();
            }
        );


        debug(
            "Added Rescan Metadata button to Image Modifiers panel."
        );
    }



    /*
     * Add the button once the ED interface is available.
     */
    if (
        document.readyState === "loading"
    ) {

        document.addEventListener(
            "DOMContentLoaded",
            addMetadataRescanButton,
            {
                once: true
            }
        );

    } else {

        addMetadataRescanButton();
    }

})();