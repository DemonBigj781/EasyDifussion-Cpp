(() => {
    "use strict";
    const storageKey = "user_settings_v2";
    const fields = [
        ["prompt", "prompt", "Prompt"], ["negative_prompt", "negative_prompt", "Negative prompt"],
        ["seed", "seed", "Seed"], ["num_outputs_total", "num_images", "Image count"],
        ["width", "width", "Width"], ["height", "height", "Height"],
        ["num_inference_steps", "steps", "Steps"], ["guidance_scale", "guidance_scale", "Guidance scale"],
        ["sampler_name", "sampler", "Sampler"], ["output_format", "output_format", "Output format"],
        ["stable_diffusion_model", "stable_diffusion_model", "Checkpoint"],
        ["vae_model", "vae_model", "VAE"], ["text_encoder_model", "text_encoder_model", "Text encoder"],
        ["lora_model", "lora_model", "LoRA models and weights"],
        ["controlnet_mode", "controlnet_mode", "ControlNet mode"],
        ["controlnet_model", "controlnet_model", "ControlNet model"],
        ["controlnet_alpha", "controlnet_alpha", "ControlNet strength"],
        ["controlnet_union_type", "controlnet_union_type", "ControlNet type"],
        ["lllite_model", "lllite-model", "LLLite model"],
        ["lllite_strength", "lllite-strength", "LLLite strength"],
        ["lllite_start", "lllite-start", "LLLite start"],
        ["lllite_end", "lllite-end", "LLLite end"],
    ];
    const read = () => {
        const value = JSON.parse(localStorage.getItem(storageKey) || "[]");
        if (!Array.isArray(value)) throw new Error("Saved input preferences are not a list");
        return value;
    };
    const allowed = (key, entries) => entries.find(entry => entry.key === "auto_save_settings")?.value !== false
        && entries.find(entry => entry.key === key)?.ignore !== true;
    function update(key, properties) {
        const entries = read();
        let entry = entries.find(item => item.key === key);
        if (!entry) { entry = {key, ignore: false}; entries.push(entry); }
        Object.assign(entry, properties);
        localStorage.setItem(storageKey, JSON.stringify(entries));
    }
    function get(key, oldKey) {
        try {
            const entries = read();
            if (!allowed(key, entries)) return undefined;
            const entry = entries.find(item => item.key === key);
            return entry && "value" in entry ? entry.value : oldKey ? localStorage.getItem(oldKey) ?? undefined : undefined;
        } catch (error) { console.warn("Could not restore input preferences", error); }
    }
    function save(key, value) {
        try { if (allowed(key, read())) update(key, {value}); }
        catch (error) { console.warn("Could not save input preferences", error); }
    }
    function bindBasic() {
        for (const [key, id] of fields) {
            const field = document.getElementById(id);
            if (!field || field.dataset.path !== undefined || !["INPUT", "TEXTAREA", "SELECT"].includes(field.tagName)
                || ["controlnet_mode", "controlnet_model", "controlnet_union_type"].includes(key)) continue;
            const value = get(key);
            if (value !== undefined) {
                if (field.type === "checkbox") field.checked = value === true;
                else field.value = String(value);
                field.dispatchEvent(new Event("input", {bubbles: true}));
                field.dispatchEvent(new Event("change", {bubbles: true}));
            }
            const store = () => save(key, field.type === "checkbox" ? field.checked : field.value);
            field.addEventListener("input", store);
            field.addEventListener("change", store);
        }
    }
    window.CppInputPreferences = {get, save, bindBasic};
    const host = document.getElementById("cpp-input-saving");
    if (!host) return;
    host.innerHTML = '<legend>Input saving</legend><label><input id="auto_save_settings" type="checkbox"> Auto-save and restore inputs</label>'
        + '<p>Uses the same per-input preferences as the legacy UI. Changes are saved automatically in this browser.</p>'
        + '<button type="button" id="cpp-configure-input-saving">Choose inputs to save…</button>'
        + '<dialog id="cpp-save-settings-config"><h3>Save Settings Configuration</h3><div id="cpp-input-saving-list"></div>'
        + '<button type="button" id="cpp-close-input-saving">Close</button></dialog><p id="cpp-input-saving-status" role="status"></p>';
    const status = host.querySelector("#cpp-input-saving-status");
    const master = host.querySelector("#auto_save_settings");
    const dialog = host.querySelector("dialog");
    dialog.style.cssText = "color:var(--ui-text,#ededf0);background:var(--ui-panel,#202329);border:1px solid var(--ui-border,#393d46);border-radius:10px;padding:18px;max-width:calc(100vw - 32px);max-height:85vh;overflow:auto;box-sizing:border-box";
    const list = host.querySelector("#cpp-input-saving-list");
    const report = fn => {
        try { fn(); status.textContent = "Saved."; }
        catch (error) { status.textContent = `Not saved: ${error.message}`; }
    };
    function refresh() {
        const entries = read();
        master.checked = entries.find(entry => entry.key === "auto_save_settings")?.value !== false;
        list.replaceChildren();
        for (const [key, , name] of fields) {
            const label = document.createElement("label"), checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.dataset.inputSetting = key;
            checkbox.checked = entries.find(entry => entry.key === key)?.ignore !== true;
            label.style.cssText = "display:flex;gap:.75rem;align-items:center;margin:.5rem 0";
            checkbox.style.width = "auto";
            label.append(checkbox, document.createTextNode(name));
            list.append(label);
            checkbox.addEventListener("change", () => report(() => update(key, {ignore: !checkbox.checked})));
        }
    }
    master.addEventListener("change", () => report(() => update("auto_save_settings", {value: master.checked})));
    host.querySelector("#cpp-configure-input-saving").addEventListener("click", () => report(() => { refresh(); dialog.showModal(); }));
    host.querySelector("#cpp-close-input-saving").addEventListener("click", () => dialog.close());
    window.addEventListener("storage", event => { if (event.key === storageKey || event.key === null) report(refresh); });
    report(refresh);
})();
