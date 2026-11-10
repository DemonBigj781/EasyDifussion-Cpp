(async () => {
    "use strict";
    if (window.CppSamSegmentation) return;

    const panel = document.getElementById("sam-segmentation-panel");
    if (!panel) return;

    const byId = id => document.getElementById(id);
    const status = byId("sam_status");
    const sourcePreview = byId("sam_source_preview");
    const maskPreview = byId("sam_mask_preview");
    const promptType = byId("sam_prompt_type");
    const pointFields = byId("sam_point_fields");
    const boxFields = byId("sam_box_fields");
    const fileInput = byId("sam_image");
    const modelInput = byId("sam_model");
    const generate = byId("sam_generate_mask");
    if (!status || !sourcePreview || !maskPreview || !promptType || !pointFields || !boxFields
        || !fileInput || !modelInput || !generate || typeof ModelDropdown !== "function") return;

    window.CppSamSegmentation = true;
    const modelPicker = new ModelDropdown(modelInput, "sam", "Select a SAM model");
    const storageKey = "easy-diffusion-cpp-sam-v1";
    let sourceData = "";

    function setStatus(message, error = false) {
        status.textContent = message;
        status.dataset.state = error ? "error" : "";
    }

    function updatePromptFields() {
        const isPoint = promptType.value === "point";
        pointFields.hidden = !isPoint;
        boxFields.hidden = isPoint;
    }

    function readNumber(id, label) {
        const value = Number.parseFloat(byId(id).value);
        if (!Number.isFinite(value)) throw new Error(`${label} must be a number.`);
        return value;
    }

    function readBox() {
        const values = ["sam_box_x1", "sam_box_y1", "sam_box_x2", "sam_box_y2"]
            .map((id, index) => readNumber(id, `Box coordinate ${index + 1}`));
        return {x1: values[0], y1: values[1], x2: values[2], y2: values[3]};
    }

    async function loadModels() {
        try {
            const response = await fetch("/get/other", {cache: "no-store"});
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            const models = (payload.models || []).filter(item => item.tags?.[0] === "sam");
            modelPicker.inputModels = buildTree(models);
            modelPicker.populateModels();
            let saved = "";
            try { saved = localStorage.getItem(storageKey) || ""; } catch (_) {}
            if (models.some(item => item.model === saved)) modelPicker.value = saved;
            setStatus(models.length ? `${models.length} SAM model${models.length === 1 ? "" : "s"} available.` : "No SAM models found.");
        } catch (error) {
            setStatus(`Could not load SAM models (${error.message}).`, true);
        }
    }

    fileInput.addEventListener("change", event => {
        const file = event.target.files?.[0];
        if (!file || !file.type.startsWith("image/")) {
            sourceData = "";
            sourcePreview.hidden = true;
            return;
        }
        const reader = new FileReader();
        reader.addEventListener("load", () => {
            sourceData = String(reader.result || "");
            sourcePreview.src = sourceData;
            sourcePreview.hidden = !sourceData;
            setStatus("Source image loaded. Choose a prompt and generate a mask.");
        });
        reader.addEventListener("error", () => setStatus("Could not read the source image.", true));
        reader.readAsDataURL(file);
    });

    modelInput.addEventListener("change", () => {
        try { localStorage.setItem(storageKey, modelPicker.value || ""); } catch (_) {}
    });
    promptType.addEventListener("change", updatePromptFields);
    generate.addEventListener("click", () => {
        Promise.resolve().then(async () => {
            if (!sourceData) throw new Error("Choose a source image first.");
            if (!modelPicker.value) throw new Error("Choose a SAM model first.");
            const payload = {
                image: sourceData,
                model: modelPicker.value,
                prompt_type: promptType.value,
                multimask: false,
            };
            if (promptType.value === "point") {
                payload.point = {
                    x: readNumber("sam_point_x", "Point X"),
                    y: readNumber("sam_point_y", "Point Y"),
                };
            } else {
                payload.box = readBox();
            }
            generate.disabled = true;
            setStatus("Generating SAM mask…");
            const response = await fetch("/sam/segment", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.detail || `SAM request failed (HTTP ${response.status}).`);
            if (typeof result.mask !== "string" || !result.mask.startsWith("data:image/png;base64,"))
                throw new Error("SAM returned no valid PNG mask.");
            maskPreview.src = result.mask;
            maskPreview.hidden = false;
            setStatus(`SAM mask generated at ${result.width} × ${result.height}.`);
        }).catch(error => setStatus(error.message || "SAM segmentation failed.", true))
            .finally(() => { generate.disabled = false; });
    });

    updatePromptFields();
    loadModels();
})();
