(() => {
    "use strict";
    function family(name, tags = []) {
        if (tags.includes("anima") || /(?:^|[/_-])anima(?:$|[/_.-])/i.test(name)) return "anima";
        if (tags.some(tag => /^(sd_xl|playground_v2_5)/.test(tag))) return "sdxl";
        if (tags.some(tag => /^sd_v1/.test(tag))) return "sd15";
        if (tags.some(tag => /^(sd_v2|sprite_gpt|flux|sd_v3)/.test(tag))) return "unsupported";
        const lower = String(name || "").toLowerCase();
        if (/sdxl|\/xl|pony|illustrious/.test(lower)) return "sdxl";
        if (/flux|sd3|cascade|sd2/.test(lower)) return "unsupported";
        return "sd15";
    }
    function adapter(name, tags = []) {
        const dimension = tags.find(tag => /^ip_adapter_embedding_\d+$/.test(tag));
        const kind = tags.find(tag => /^ip_adapter_(base|plus|anima)$/.test(tag));
        if (dimension && kind) {
            return {dimension: Number(dimension.split("_").pop()), kind: kind.slice(11), detected: true};
        }
        const lower = String(name || "").toLowerCase();
        if (lower.includes("faceid") || lower.includes("anima")) return null;
        const fallbackKind = /plus|perceiver/.test(lower) ? "plus" : "base";
        if (/vit-h|sd15/.test(lower)) return {dimension: fallbackKind === "plus" ? 1280 : 1024, kind: fallbackKind, detected: false};
        if (/vit-g|sdxl/.test(lower)) return {dimension: fallbackKind === "plus" ? 1664 : 1280, kind: fallbackKind, detected: false};
        return null;
    }
    function compatible(name, tags, compatibility) {
        if (!name || !compatibility) return false;
        if (compatibility.kind === "anima") return tags.includes("siglip2_base_patch16_512");
        if (tags.includes("siglip2_base_patch16_512")) return false;
        const prefix = compatibility.kind === "plus" ? "clip_hidden_" : "clip_projection_";
        const dimensionTag = tags.find(tag => String(tag).startsWith(prefix));
        if (dimensionTag) return Number(dimensionTag.slice(prefix.length)) === compatibility.dimension;
        const lower = String(name).toLowerCase();
        const plus = compatibility.kind === "plus";
        if (/(?:vit[-_]?h|vision[-_]?h|clip[-_]?h)(?:\b|_)/.test(lower)) return (plus ? 1280 : 1024) === compatibility.dimension;
        if (/(?:vit[-_]?g|bigg|big[-_]?g|vision[-_]?g|clip[-_]?g)(?:\b|_)/.test(lower)) return (plus ? 1664 : 1280) === compatibility.dimension;
        if (/(?:vit[-_]?l|vision[-_]?l|clip[-_]?l)(?:\b|_)/.test(lower)) return (plus ? 1024 : 768) === compatibility.dimension;
        return false;
    }
    function adapterForFamily(compatibility, checkpointFamily) {
        if (!compatibility || checkpointFamily === "unsupported") return false;
        return (checkpointFamily === "anima") === (compatibility.kind === "anima");
    }
    window.NativeIPAdapterCompatibility = Object.freeze({family, adapter, compatible, adapterForFamily});
})();
