// Shared pixel selection for the legacy and C++ canvas editors.
(() => {
    "use strict";
    function expand(mask, width, height, radius) {
        radius = Math.max(0, Math.floor(radius));
        if (!radius) return mask;
        const queue = new Int32Array(width * height), distance = new Uint16Array(width * height);
        let read = 0, write = 0;
        for (let i = 0; i < mask.length; i++) {
            if (mask[i]) {distance[i] = 1; queue[write++] = i;}
        }
        while (read < write) {
            const pixel = queue[read++], x = pixel % width, depth = distance[pixel];
            if (depth > radius) continue;
            const neighbors = [pixel - 1, pixel + 1, pixel - width, pixel + width];
            for (let i = 0; i < neighbors.length; i++) {
                const next = neighbors[i];
                if (next < 0 || next >= mask.length || distance[next]) continue;
                if ((i === 0 && x === 0) || (i === 1 && x === width - 1)) continue;
                distance[next] = depth + 1; mask[next] = 1; queue[write++] = next;
            }
        }
        return mask;
    }
    function select(rgba, width, height, x, y, sensitivity = 30, buffer = 0) {
        if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 || rgba.length !== width * height * 4) {
            throw new Error('Magic-wand image dimensions do not match its pixels.');
        }
        if (![x,y,sensitivity,buffer].every(Number.isFinite)) throw new Error('Magic-wand coordinates and options must be finite.');
        x = Math.max(0, Math.min(width - 1, Math.floor(x)));
        y = Math.max(0, Math.min(height - 1, Math.floor(y)));
        const selected = new Uint8Array(width * height), queue = new Int32Array(width * height);
        const start = y * width + x, base = start * 4, thresholdSquared = Math.max(0, sensitivity) ** 2;
        let read = 0, write = 0;
        selected[start] = 1; queue[write++] = start;
        while (read < write) {
            const pixel = queue[read++], px = pixel % width;
            const neighbors = [pixel - 1, pixel + 1, pixel - width, pixel + width];
            for (let i = 0; i < neighbors.length; i++) {
                const next = neighbors[i];
                if (next < 0 || next >= selected.length || selected[next]) continue;
                if ((i === 0 && px === 0) || (i === 1 && px === width - 1)) continue;
                const offset = next * 4;
                const dr = rgba[offset] - rgba[base], dg = rgba[offset + 1] - rgba[base + 1], db = rgba[offset + 2] - rgba[base + 2];
                if (dr * dr + dg * dg + db * db > thresholdSquared) continue;
                selected[next] = 1; queue[write++] = next;
            }
        }
        return expand(selected, width, height, buffer);
    }
    if (typeof module !== 'undefined' && module.exports) module.exports = {select, expand};
    else window.EditorMaskTools = {select, expand};
})();
