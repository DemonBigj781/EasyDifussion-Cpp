(async () => {
    "use strict";
    if (window.CppIPAdapter) return;
    const stateKey = "easy-diffusion-native-ip-adapter-v1";
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(stateKey) || "{}"); } catch (_) {}
    window.CppIPAdapter = {requestOptions() {
        if (saved.enabled) throw new Error("IP-Adapter controls are still loading.");
        return {};
    }};
    let policy;
    try { policy = await window.CppKiosk.ready; }
    catch (_) { return; }
    if (policy.enabled) {
        window.CppIPAdapter.requestOptions = () => ({});
        return;
    }
    const root = document.getElementById("generate-plugin-panels");
    if (!root) return;
    const panel = document.createElement("section");
    panel.id = "cpp-ip-adapter-panel";
    panel.className = "panel-box generate-plugin-panel";
    panel.setAttribute("aria-label", "Image prompt (IP-Adapter)");
    panel.textContent = "Loading image prompt controls…";
    root.append(panel);
    try {
        const [templateResponse, modelResponse] = await Promise.all([
            fetch("/plugins/core/controlnet_plugin/ip-adapter.plugin.html"),
            fetch("/get/models", {cache: "no-store"}),
        ]);
        if (!templateResponse.ok || !modelResponse.ok) throw new Error("Could not load the image prompt model catalog.");
        panel.innerHTML = await templateResponse.text();
        const models = (await modelResponse.json()).models || [];
        const api = window.NativeIPAdapterCompatibility;
        const byId = id => panel.querySelector(`#${id}`);
        const enabled = byId("ip-adapter-enabled");
        const status = byId("ip-adapter-status");
        const adapter = new ModelDropdown(byId("ip-adapter-model"), null, "Select adapter");
        const encoder = new ModelDropdown(byId("ip-adapter-clip"), null, "Select vision encoder");
        const adapterModels = models.filter(item => item.tags?.includes("ip-adapter"));
        const encoderModels = models.filter(item => item.tags?.includes("clip-vision"));
        const named = (items, name) => items.find(item => item.model === name);
        adapter.inputModels = buildTree(adapterModels);
        encoder.inputModels = buildTree(encoderModels);
        adapter.populateModels(); encoder.populateModels();
        const preview = byId("ip-adapter-image-preview");
        const wrapper = byId("ip-adapter-image-wrapper");
        wrapper.hidden = true;
        byId("ip-adapter-use-init").hidden = true;
        const style = document.createElement("style");
        style.textContent = `#cpp-ip-adapter-panel .ip-adapter-grid{display:grid;grid-template-columns:9rem minmax(0,1fr);gap:.65rem;align-items:center}
            #cpp-ip-adapter-panel input[type=number]{width:5rem} #cpp-ip-adapter-panel input[type=file]{max-width:100%}
            #cpp-ip-adapter-panel .ip-adapter-preview{margin:1rem 0;position:relative;min-height:3rem}
            #cpp-ip-adapter-panel img{max-width:100%;max-height:240px;object-fit:contain}
            #cpp-ip-adapter-panel .image_clear_btn{position:absolute;top:.5rem;right:.5rem}
            #cpp-ip-adapter-panel small{display:block;line-height:1.5}
            @media(max-width:600px){#cpp-ip-adapter-panel .ip-adapter-grid{grid-template-columns:1fr}}`;
        panel.append(style);
        byId("ip-adapter-strength").value = saved.strength ?? "1";
        byId("ip-adapter-start").value = saved.start ?? "0";
        byId("ip-adapter-end").value = saved.end ?? "100";
        if (named(adapterModels, saved.model)) adapter.value = saved.model;
        if (named(encoderModels, saved.clip)) encoder.value = saved.clip;
        function save() {
            try { localStorage.setItem(stateKey, JSON.stringify({enabled: enabled.checked, model: adapter.value,
                clip: encoder.value, strength: byId("ip-adapter-strength").value,
                start: byId("ip-adapter-start").value, end: byId("ip-adapter-end").value})); } catch (_) {}
        }
        const compatibility = name => api.adapter(name, named(adapterModels, name)?.tags || []);
        const matchesEncoder = (name, spec) => api.compatible(name, named(encoderModels, name)?.tags || [], spec);
        function family() {
            const field = document.getElementById("stable_diffusion_model");
            const name = field?.dataset.path || field?.value || "";
            return api.family(name, named(models, name)?.tags || []);
        }
        function refresh() {
            const selectedFamily = family();
            enabled.disabled = selectedFamily === "unsupported";
            if (enabled.disabled) enabled.checked = false;
            adapter.setModelPredicate?.(name => api.adapterForFamily(compatibility(name), selectedFamily));
            if (!api.adapterForFamily(compatibility(adapter.value), selectedFamily)) {
                const next = adapterModels.find(item => api.adapterForFamily(compatibility(item.model), selectedFamily))?.model || "";
                if (adapter.value !== next) adapter.value = next;
            }
            const spec = compatibility(adapter.value);
            encoder.setModelPredicate?.(name => matchesEncoder(name, spec));
            if (!matchesEncoder(encoder.value, spec)) {
                const next = encoderModels.find(item => matchesEncoder(item.model, spec))?.model || "";
                if (encoder.value !== next) encoder.value = next;
            }
            byId("ip-adapter-strength").min = selectedFamily === "anima" ? "0" : "-10";
            status.textContent = enabled.disabled ? "IP-Adapter is available for Anima, SD 1.x and SDXL." :
                !spec ? "No compatible adapter is installed for this checkpoint." :
                !encoder.value ? "No compatible vision encoder is installed." :
                spec.kind === "anima" ? "Anima · SigLIP2 base patch16-512 · CPU/CUDA only. Strength 0 disables both image attention and adapter LoRA." :
                `${spec.kind} adapter · vision embedding width ${spec.dimension}.`;
        }
        for (const field of [adapter, encoder]) field.addEventListener("change", () => { refresh(); save(); });
        document.getElementById("stable_diffusion_model")?.addEventListener("change", refresh);
        panel.querySelectorAll("input").forEach(input => input.addEventListener("change", save));
        byId("ip-adapter-image-input").addEventListener("change", event => {
            const file = event.target.files?.[0];
            if (!file || !file.type.startsWith("image/")) return;
            const reader = new FileReader();
            reader.onload = () => {
                preview.src = String(reader.result); wrapper.hidden = false;
                wrapper.classList.remove("displayNone"); enabled.checked = !enabled.disabled; save();
            };
            reader.onerror = () => { status.textContent = "Could not read the reference image."; };
            reader.readAsDataURL(file);
        });
        byId("ip-adapter-image-clear").addEventListener("click", () => {
            preview.removeAttribute("src"); wrapper.hidden = true; enabled.checked = false;
            byId("ip-adapter-image-input").value = ""; save();
        });
        window.CppIPAdapter.requestOptions = () => {
            if (policy.enabled || !enabled.checked) return {};
            const spec = compatibility(adapter.value);
            if (!api.adapterForFamily(spec, family()) || !matchesEncoder(encoder.value, spec)) {
                throw new Error("Select a compatible IP-Adapter and vision encoder.");
            }
            const image = preview.getAttribute("src") || "";
            if (!image.startsWith("data:image/")) throw new Error("Upload an IP-Adapter reference image.");
            const strength = Number(byId("ip-adapter-strength").value);
            const start = Number(byId("ip-adapter-start").value);
            const end = Number(byId("ip-adapter-end").value);
            if (![strength, start, end].every(Number.isFinite) || strength < (spec.kind === "anima" ? 0 : -10) ||
                strength > 10 || start < 0 || end > 100 || start >= end) {
                throw new Error("Use a valid IP-Adapter strength and an increasing step range within 0–100%.");
            }
            return {ip_adapter_model: adapter.value, ip_adapter_clip_vision: encoder.value, ip_adapter_image: image,
                ip_adapter_strength: strength, ip_adapter_start_percent: start, ip_adapter_end_percent: end};
        };
        refresh();
    } catch (error) {
        panel.textContent = error.message;
        window.CppIPAdapter.requestOptions = () => { throw new Error(error.message); };
    }
})();
