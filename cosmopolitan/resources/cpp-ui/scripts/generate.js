// BEGIN VERBATIM SHARED EDITOR MASK TOOLS (source equality tested)
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
// END VERBATIM SHARED EDITOR MASK TOOLS
// BEGIN PORTABLE ORDINARY MASK PIXELS
(() => {
    "use strict";
    function dimensions(width,height,length,channels=1) {
        if (!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width*height>4*1024*1024||length!==width*height*channels) throw new Error("Invalid or oversized ordinary-mask pixels.");
    }
    function fromRgba(rgba,width,height) {
        dimensions(width,height,rgba.length,4);const mask=new Uint8ClampedArray(width*height);
        for(let i=0;i<mask.length;i++) {const p=i*4;if(rgba[p+3]!==255)throw new Error("Mask editor requires opaque pixels; transparent uploaded masks remain available unchanged for generation.");mask[i]=(77*rgba[p]+150*rgba[p+1]+29*rgba[p+2])>>8;}
        return mask;
    }
    function toRgba(mask,width,height) {
        dimensions(width,height,mask.length);const rgba=new Uint8ClampedArray(mask.length*4);
        for(let i=0;i<mask.length;i++){const p=i*4;rgba[p]=rgba[p+1]=rgba[p+2]=mask[i];rgba[p+3]=255;}return rgba;
    }
    function options(target,strength) { if (![0,255].includes(target)||!Number.isFinite(strength)||strength<=0||strength>1) throw new Error("Mask strength must be greater than 0 through 1."); }
    function select(mask,selected,width,height,target,strength) {
        dimensions(width,height,mask.length);dimensions(width,height,selected.length);options(target,strength);
        for(let i=0;i<mask.length;i++)if(selected[i])mask[i]=Math.round(mask[i]+(target-mask[i])*strength);
    }
    function stroke(mask,width,height,from,to,size,target,strength) {
        dimensions(width,height,mask.length);options(target,strength);
        if(from.length!==2||to.length!==2||![...from,...to,size].every(Number.isFinite)||size<1||size>512)throw new Error("Invalid mask brush geometry.");
        from=from.map(value=>Math.floor(value)+.5);to=to.map(value=>Math.floor(value)+.5);
        const radius=size/2,dx=to[0]-from[0],dy=to[1]-from[1],length=dx*dx+dy*dy;
        const x0=Math.max(0,Math.floor(Math.min(from[0],to[0])-radius)),x1=Math.min(width-1,Math.ceil(Math.max(from[0],to[0])+radius));
        const y0=Math.max(0,Math.floor(Math.min(from[1],to[1])-radius)),y1=Math.min(height-1,Math.ceil(Math.max(from[1],to[1])+radius));
        for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const t=length?Math.max(0,Math.min(1,((x+.5-from[0])*dx+(y+.5-from[1])*dy)/length)):0;
            const sx=x+.5-(from[0]+t*dx),sy=y+.5-(from[1]+t*dy);if(sx*sx+sy*sy<=radius*radius){const i=y*width+x;mask[i]=Math.round(mask[i]+(target-mask[i])*strength);}}
    }
    function invert(mask,width,height) {
        dimensions(width,height,mask.length);
        for(let i=0;i<mask.length;i++)mask[i]=255-mask[i];
        return true;
    }
    function fill(mask,width,height,value) {
        dimensions(width,height,mask.length);
        if(![0,255].includes(value))throw new Error("Whole-mask fill must be black or white.");
        let changed=false;
        for(let i=0;i<mask.length;i++)if(mask[i]!==value){mask[i]=value;changed=true;}
        return changed;
    }
    window.PortableMaskPixels={fromRgba,toRgba,select,stroke,invert,fill};
})();
// END PORTABLE ORDINARY MASK PIXELS
(() => {
    "use strict";
    if (window.CppNativeGeneration) return;
    window.CppNativeGeneration = true;
    const byId = id => document.getElementById(id);
    const controls = document.querySelector(".generation-controls");
    if (!controls) return;
    const start = byId("makeImage"), stop = byId("stopImage");
    const status = byId("generation-queue-status"), progress = byId("generation-progress");
    const preview = byId("preview-content"), modelStatus = byId("generation-model-status");
    const editable = ["prompt", "negative_prompt", "seed", "width", "height", "steps", "guidance_scale", "sampler"];
    let imageInputsSupported = false, maskInputsSupported = false, guideInputsSupported = false;
    let editingMask = false;
    let editingGuide = false, editorLoading = false, editorRevision = 0;
    let ready = false, active = null, loadingModels = false, savingDefaults = false, models = new Set();
    const devices = new Map();

    for (const control of controls.querySelectorAll("input,select,textarea,button")) control.disabled = true;
    byId("show-download-popup").disabled = true;
    byId("show-download-popup").title = "Use the PNG download link beneath each generated image.";
    byId("output_format").value = "png";
    byId("output_format").title = "The native image API returns PNG.";
    byId("num_images").value = "1";
    byId("num_images").title = "This interface runs one image per request.";
    for (const id of ["vae_model", "text_encoder_model"]) {
        byId(id).value = "";
        byId(id).placeholder = "Use a complete checkpoint";
        byId(id).title = "Separate companion models are not supported by this native form.";
    }
    byId("generation-companion-status").textContent = "Use a complete checkpoint with its text encoder and VAE. Separate components, LoRAs, ControlNet, generation plugins are not connected to this form.";
    byId("cpp-modifier-status").textContent = "Image modifier presets are unavailable here. Describe the style in the prompt.";
    byId("cpp-modifier-grid").setAttribute("aria-busy", "false");
    byId("lora_model").textContent = "LoRA selection is unavailable in this native form.";
    for (const id of ["generate-plugin-panels", "cpp-image-modifiers"]) {
        byId(id).setAttribute("aria-disabled", "true");
        byId(id).title = "Not supported by this native generation form.";
    }

    const oldCheckpoint = byId("stable_diffusion_model");
    const checkpoint = document.createElement("select");
    checkpoint.id = oldCheckpoint.id;
    checkpoint.name = oldCheckpoint.name;
    checkpoint.disabled = true;
    checkpoint.add(new Option("Loading checkpoints…", ""));
    oldCheckpoint.replaceWith(checkpoint);
    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.textContent = "Refresh checkpoints";
    refresh.disabled = true;
    modelStatus.after(refresh);
    const saveDefaults = document.createElement("button");
    saveDefaults.id = "cosmo-save-generation-defaults";
    saveDefaults.type = "button";
    saveDefaults.textContent = "Save generation defaults";
    saveDefaults.title = "Save this checkpoint and image recipe for the UI, shell and API. Device startup preferences are saved in Settings.";
    saveDefaults.disabled = true;
    refresh.after(saveDefaults);
    const backendLabel = document.createElement("label");
    backendLabel.textContent = "Compute device ";
    const backend = document.createElement("select");
    backend.id = "cosmo-generation-backend";
    backend.disabled = true;
    backendLabel.append(backend);
    refresh.after(backendLabel);
    const backendHelp = document.createElement("p");
    backendHelp.textContent = "Software adapters run on the CPU. Hardware entries identify an actual integrated or discrete GPU. Unsupported operations may fall back to GGML CPU; device memory is shown only when measured.";
    backendLabel.after(backendHelp);
    const sampler = byId("sampler");
    sampler.replaceChildren(new Option("Euler", "euler"), new Option("Euler a", "euler_a"), new Option("DPM++ 2M", "dpm++2m"));
    for (const id of ["width", "height"]) {
        const update = () => { byId(`${id}-value`).textContent = `${byId(id).value} px`; };
        byId(id).addEventListener("input", update);
        update();
    }

    // Inputs stay browser-local until Generate. Never save image data in defaults.
    const imagePanel = document.createElement("fieldset");
    imagePanel.id = "cosmo-image-inputs";
    imagePanel.hidden = true;
    const imageLegend = document.createElement("legend");
    imageLegend.textContent = "Image-to-image / Inpaint / Advanced Inpaint";
    const imageHelp = document.createElement("p");
    imageHelp.textContent = "Optional PNG or JPEG initial image. Output uses the selected width and height. An optional mask uses white to repaint and black to preserve. Advanced Inpaint uses a separate RGBA PNG color guide: opacity weights starting-image color, not paint over the output. Source and guide dimensions must match. Mask inversion and blur apply to the combined edit mask. Full selection editors remain available in the original UIs.";
    imagePanel.append(imageLegend, imageHelp);
    const uploads = {};
    for (const kind of ["initial", "mask", "guide"]) {
        const label = document.createElement("label");
        label.textContent = kind === "initial" ? "Initial image " : kind === "mask" ? "Inpainting mask " : "Advanced Inpaint color guide ";
        const input = document.createElement("input");
        input.id = `cosmo-${kind}-file`; input.type = "file"; input.accept = kind === "guide" ? "image/png" : "image/png,image/jpeg";
        const thumbnail = document.createElement("img");
        thumbnail.id = `cosmo-${kind}-preview`; thumbnail.alt = label.textContent.trim();
        thumbnail.hidden = true; thumbnail.style.maxWidth = "240px"; thumbnail.style.maxHeight = "180px";
        const clear = document.createElement("button");
        clear.id = `cosmo-${kind}-clear`; clear.type = "button"; clear.textContent = `Remove ${kind} image`;
        label.append(input); imagePanel.append(label, thumbnail, clear);
        const enableLabel = document.createElement("label");
        const enabled = document.createElement("input"); enabled.id = `cosmo-${kind}-enabled`; enabled.type = "checkbox"; enabled.checked = true;
        enableLabel.append(enabled, document.createTextNode(`Use ${kind === "guide" ? "Advanced Inpaint guide" : "ordinary mask"}`));
        enableLabel.hidden = kind === "initial"; imagePanel.append(enableLabel);
        enabled.addEventListener("change", updateControls);
        const entry = uploads[kind] = {input, label, thumbnail, clear, enabled, enableLabel, data: null, width: 0, height: 0, revision: 0, loading: false};
        clear.addEventListener("click", () => { if (!active) clearUpload(kind); });
        input.addEventListener("change", () => loadUpload(kind, input.files?.[0]));
    }
    // Small local brush editor complements upload; original advanced selection
    // and automatic-mask editors are preserved, not replaced by this surface.
    const editGuide = document.createElement("button"); editGuide.id = "cosmo-guide-edit"; editGuide.type = "button"; editGuide.textContent = "Paint color guide";
    imagePanel.append(editGuide);
    const editor = document.createElement("fieldset"); editor.id = "cosmo-guide-editor"; editor.hidden = true;
    const canvas = document.createElement("canvas"); canvas.id = "cosmo-guide-canvas"; canvas.setAttribute("aria-label", "Paint color guidance with pointer; use RGBA upload as an alternative"); canvas.style.maxWidth = "100%"; canvas.style.width = "512px"; canvas.style.height = "auto"; canvas.style.touchAction = "none"; canvas.style.backgroundSize = "100% 100%";
    const editorControls = [];
    function editorInput(id, title, type, value) {
        const label = document.createElement("label"); label.textContent = title + " ";
        const input = document.createElement("input"); input.id = id; input.type = type; input.value = value;
        label.append(input); editor.append(label); editorControls.push(input); return input;
    }
    const brushColor = editorInput("cosmo-guide-color", "Color", "color", "#3366ff");
    const brushOpacity = editorInput("cosmo-guide-opacity", "Opacity", "number", "0.5"); brushOpacity.min = "0.01"; brushOpacity.max = "1"; brushOpacity.step = "0.05";
    const brushSize = editorInput("cosmo-guide-size", "Brush pixels", "number", "24"); brushSize.min = "1"; brushSize.max = "512";
    const toolLabel = document.createElement("label"); toolLabel.textContent = "Guide tool ";
    const guideTool = document.createElement("select"); guideTool.id = "cosmo-guide-tool";
    guideTool.add(new Option("Brush", "brush")); guideTool.add(new Option("Magic wand (source color)", "wand")); toolLabel.append(guideTool); editor.append(toolLabel); editorControls.push(guideTool);
    const sensitivity = editorInput("cosmo-guide-sensitivity", "Wand RGB sensitivity", "number", "30"); sensitivity.min = "0"; sensitivity.max = "255"; sensitivity.step = "1";
    const selectionBuffer = editorInput("cosmo-guide-buffer", "Wand selection buffer (pixels)", "number", "0"); selectionBuffer.min = "0"; selectionBuffer.max = "64"; selectionBuffer.step = "1";
    const wandHelp = document.createElement("p"); wandHelp.textContent = "Magic wand selects connected similar source colors, then expands the region by the buffer. It colors or erases only this guide, not the source or ordinary mask. It is not SAM segmentation."; editor.append(wandHelp);
    const erase = editorInput("cosmo-guide-erase", "Erase", "checkbox", "");
    editor.append(canvas);
    function editorButton(id, title, fn) { const button = document.createElement("button"); button.id = id; button.type = "button"; button.textContent = title; button.addEventListener("click",fn); editor.append(button); editorControls.push(button); return button; }
    let painting = false, lastPoint = null, undo = [], sourcePixels = null;
    const context = () => canvas.getContext("2d", {willReadFrequently:true});
    function closeEditor() { editingGuide = false; editorLoading = false; ++editorRevision; painting = false; undo = []; sourcePixels = null; editor.hidden = true; updateControls(); }
    editorButton("cosmo-guide-undo", "Undo stroke", () => { if (editingGuide && undo.length) context().putImageData(undo.pop(),0,0); });
    editorButton("cosmo-guide-reset", "Clear painted guide", () => { if (editingGuide) { undo = []; context().clearRect(0,0,canvas.width,canvas.height); } });
    editorButton("cosmo-guide-save", "Use painted guide", () => {
        if (!editingGuide || editorLoading) return;
        const data = canvas.toDataURL("image/png"); closeEditor();
        // Use the same decode/alpha/dimension validation as uploaded guides.
        const bytes = Uint8Array.from(atob(data.split(",")[1]), value => value.charCodeAt(0));
        loadUpload("guide", new File([bytes], "painted-guide.png", {type:"image/png"}));
    });
    editorButton("cosmo-guide-cancel", "Cancel guide edits", closeEditor);
    editGuide.addEventListener("click", async () => {
        if (active || editingGuide || editingMask || !guideInputsSupported || !uploads.initial.data) return;
        if (uploads.initial.width * uploads.initial.height > 4 * 1024 * 1024) { message("Brush editor supports up to 4 megapixels; upload a prepared guide for larger images.",true); return; }
        editingGuide = true; editor.hidden = false; canvas.width = uploads.initial.width; canvas.height = uploads.initial.height;
        const sourceData = uploads.initial.data, guideData = uploads.guide.data, revision = ++editorRevision;
        canvas.style.backgroundImage = `url("${sourceData}")`; undo = []; sourcePixels = null; editorLoading = true; updateControls();
        const decode = data => new Promise((resolve,reject) => { const img = new Image(); img.onload = () => resolve(img); img.onerror = () => reject(new Error("Could not load source or guide in editor.")); img.src = data; });
        try {
            const [source, guide] = await Promise.all([decode(sourceData), guideData ? decode(guideData) : Promise.resolve(null)]);
            if (!editingGuide || revision !== editorRevision) return;
            const original = document.createElement("canvas"); original.width = canvas.width; original.height = canvas.height;
            const ctx = original.getContext("2d", {willReadFrequently:true}); ctx.drawImage(source,0,0);
            sourcePixels = ctx.getImageData(0,0,canvas.width,canvas.height).data;
            if (guide) context().drawImage(guide,0,0);
            editorLoading = false; updateControls();
        } catch (error) { if (revision === editorRevision) { closeEditor(); message(error.message,true); } }
    });
    function snapshotGuide() { undo.push(context().getImageData(0,0,canvas.width,canvas.height)); if (undo.length > 5) undo.shift(); }
    function selectGuide(event) {
        const threshold = Number(sensitivity.value), buffer = Number(selectionBuffer.value), opacity = Number(brushOpacity.value);
        if (!Number.isFinite(threshold) || threshold < 0 || threshold > 255 || !Number.isInteger(buffer) || buffer < 0 || buffer > 64 || !(opacity > 0 && opacity <= 1)) {
            message("Wand sensitivity must be 0–255, buffer a whole number 0–64, and opacity greater than 0 through 1.",true); return;
        }
        const bounds = canvas.getBoundingClientRect();
        const x = (event.clientX-bounds.left)*canvas.width/bounds.width, y = (event.clientY-bounds.top)*canvas.height/bounds.height;
        try {
            const selected = window.EditorMaskTools.select(sourcePixels,canvas.width,canvas.height,x,y,threshold,buffer);
            const fill = document.createElement("canvas"); fill.width = canvas.width; fill.height = canvas.height;
            const ctx = fill.getContext("2d"), pixels = ctx.createImageData(canvas.width,canvas.height);
            const rgb = [1,3,5].map(offset => parseInt(brushColor.value.slice(offset,offset+2),16));
            const alpha = Math.round(255*opacity);
            for (let i=0; i<selected.length; i++) if (selected[i]) { const offset = i*4; pixels.data[offset] = rgb[0]; pixels.data[offset+1] = rgb[1]; pixels.data[offset+2] = rgb[2]; pixels.data[offset+3] = alpha; }
            ctx.putImageData(pixels,0,0); snapshotGuide();
            const target = context(); target.globalAlpha = 1; target.globalCompositeOperation = erase.checked ? "destination-out" : "source-over";
            target.drawImage(fill,0,0); target.globalCompositeOperation = "source-over";
        } catch (error) { message(error.message,true); }
    }

    function paint(event) {
        if (!painting || !editingGuide || editorLoading || active) return;
        const rect = canvas.getBoundingClientRect(), point = [(event.clientX-rect.left)*canvas.width/rect.width,(event.clientY-rect.top)*canvas.height/rect.height];
        const opacity = Number(brushOpacity.value), size = Number(brushSize.value);
        if (!(opacity > 0 && opacity <= 1 && size >= 1 && size <= 512)) return;
        const ctx = context(); ctx.globalCompositeOperation = erase.checked ? "destination-out" : "source-over"; ctx.globalAlpha = opacity; ctx.strokeStyle = brushColor.value; ctx.fillStyle = brushColor.value; ctx.lineWidth = size; ctx.lineCap = "round";
        ctx.beginPath(); if (lastPoint) { ctx.moveTo(...lastPoint); ctx.lineTo(...point); ctx.stroke(); } else { ctx.arc(...point,size/2,0,Math.PI*2); ctx.fill(); }
        ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over"; lastPoint = point;
    }
    canvas.addEventListener("pointerdown", event => { if (!editingGuide || editorLoading || active) return; if (event.button !== undefined && event.button !== 0) return; if (guideTool.value === "wand") { selectGuide(event); return; } snapshotGuide(); painting = true; lastPoint = null; canvas.setPointerCapture(event.pointerId); paint(event); });
    canvas.addEventListener("pointermove", paint);
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(name, () => { painting = false; lastPoint = null; });
    imagePanel.append(editor);
    // Ordinary masks use explicit grayscale pixels, never guide/canvas alpha.
    const maskEdit = document.createElement("button"); maskEdit.id="cosmo-mask-edit";maskEdit.type="button";maskEdit.textContent="Edit ordinary mask";imagePanel.append(maskEdit);
    const maskEditor=document.createElement("fieldset");maskEditor.id="cosmo-mask-editor";maskEditor.hidden=true;imagePanel.append(maskEditor);
    const maskHelp=document.createElement("p");maskHelp.textContent="Ordinary mask only: white repaints, black preserves. Pink previews the selected region over the unchanged source. Save exports opaque grayscale. Invert mask pixels changes the saved mask; the separate request Invert mask still applies afterward and is not changed. Blur remains request-only. Intermediate gray is a mask value, not guaranteed proportional denoising; the model may threshold it. Transparent uploads cannot be edited but can still be submitted unchanged.";maskEditor.append(maskHelp);
    const maskEditorControls=[];
    function maskControl(id,title,type,value){const label=document.createElement("label");label.textContent=title+" ";const input=document.createElement("input");input.id=id;input.type=type;input.value=value;label.append(input);maskEditor.append(label);maskEditorControls.push(input);return input;}
    const maskSize=maskControl("cosmo-mask-size","Brush pixels","number","24");maskSize.min="1";maskSize.max="512";
    const maskStrength=maskControl("cosmo-mask-strength","Mask value blend (not denoising strength)","number","1");maskStrength.min="0.01";maskStrength.max="1";maskStrength.step="0.05";
    const maskErase=maskControl("cosmo-mask-erase","Erase to black","checkbox","");
    const maskThreshold=maskControl("cosmo-mask-sensitivity","Wand RGB sensitivity","number","30");maskThreshold.min="0";maskThreshold.max="255";
    const maskBuffer=maskControl("cosmo-mask-buffer","Wand buffer (pixels)","number","0");maskBuffer.min="0";maskBuffer.max="64";
    const maskToolLabel=document.createElement("label");maskToolLabel.textContent="Ordinary mask tool ";const maskTool=document.createElement("select");maskTool.id="cosmo-mask-tool";maskTool.add(new Option("Brush","brush"));maskTool.add(new Option("Magic wand (source color)","wand"));maskToolLabel.append(maskTool);maskEditor.append(maskToolLabel);maskEditorControls.push(maskTool);
    const maskCanvas=document.createElement("canvas");maskCanvas.id="cosmo-mask-canvas";maskCanvas.style.width="512px";maskCanvas.style.maxWidth="100%";maskCanvas.style.height="auto";maskCanvas.style.touchAction="none";maskCanvas.setAttribute("aria-label","Paint ordinary mask with pointer; upload is an alternative");maskEditor.append(maskCanvas);
    let maskLoading=false,maskRevision=0,maskPixels=null,maskSource=null,maskHistory=[],maskPointer=null,maskLast=null,maskStroke=null;
    function maskContext(){return maskCanvas.getContext("2d",{willReadFrequently:true});}
    function closeMaskEditor(){const pointer=maskPointer;editingMask=false;maskLoading=false;++maskRevision;maskPixels=maskSource=null;maskHistory=[];maskPointer=maskLast=null;maskStroke=null;if(pointer!==null&&maskCanvas.hasPointerCapture?.(pointer))maskCanvas.releasePointerCapture(pointer);maskEditor.hidden=true;updateControls();}
    function maskSnapshot(){maskHistory.push(maskPixels.slice());if(maskHistory.length>5)maskHistory.shift();}
    function renderMask(){if(!maskPixels||!maskSource)return;const ctx=maskContext();ctx.putImageData(maskSource,0,0);const overlay=document.createElement("canvas");overlay.width=maskCanvas.width;overlay.height=maskCanvas.height;const oc=overlay.getContext("2d"),pixels=oc.createImageData(overlay.width,overlay.height);for(let i=0;i<maskPixels.length;i++){const p=i*4;pixels.data[p]=255;pixels.data[p+2]=128;pixels.data[p+3]=Math.round(maskPixels[i]*.55);}oc.putImageData(pixels,0,0);ctx.drawImage(overlay,0,0);}
    function maskButton(id,title,fn){const b=document.createElement("button");b.id=id;b.type="button";b.textContent=title;b.addEventListener("click",fn);maskEditor.append(b);maskEditorControls.push(b);return b;}
    maskButton("cosmo-mask-undo","Undo mask operation",()=>{if(editingMask&&!maskLoading&&maskPointer===null&&maskHistory.length){maskPixels=maskHistory.pop();renderMask();}});
    function wholeMaskAction(operation) {
        if(!editingMask||maskLoading||maskPointer!==null||active)return;
        const next=maskPixels.slice();
        if(operation(next,maskCanvas.width,maskCanvas.height)){maskSnapshot();maskPixels=next;renderMask();}
    }
    maskButton("cosmo-mask-invert-pixels","Invert mask pixels (255 − value)",()=>wholeMaskAction(window.PortableMaskPixels.invert));
    maskButton("cosmo-mask-fill","Fill entire mask white",()=>wholeMaskAction((pixels,width,height)=>window.PortableMaskPixels.fill(pixels,width,height,255)));
    maskButton("cosmo-mask-reset","Clear entire mask black",()=>wholeMaskAction((pixels,width,height)=>window.PortableMaskPixels.fill(pixels,width,height,0)));
    maskButton("cosmo-mask-cancel","Cancel mask edits",closeMaskEditor);
    maskButton("cosmo-mask-save","Use ordinary mask",()=>{if(!editingMask||maskLoading||maskPointer!==null)return;const output=document.createElement("canvas");output.width=maskCanvas.width;output.height=maskCanvas.height;const ctx=output.getContext("2d"),pixels=ctx.createImageData(output.width,output.height);pixels.data.set(window.PortableMaskPixels.toRgba(maskPixels,output.width,output.height));ctx.putImageData(pixels,0,0);const data=output.toDataURL("image/png");closeMaskEditor();const bytes=Uint8Array.from(atob(data.split(",")[1]),v=>v.charCodeAt(0));loadUpload("mask",new File([bytes],"ordinary-mask.png",{type:"image/png"}));});
    maskEdit.addEventListener("click",async()=>{
        if(active||editingMask||editingGuide||!maskInputsSupported||!uploads.initial.data)return;
        if(uploads.initial.width*uploads.initial.height>4*1024*1024){message("Mask editor supports up to 4 megapixels; larger masks can be uploaded.",true);return;}
        editingMask=true;maskLoading=true;maskEditor.hidden=false;maskCanvas.width=uploads.initial.width;maskCanvas.height=uploads.initial.height;const revision=++maskRevision,sourceData=uploads.initial.data,existing=uploads.mask.data;updateControls();
        const decode=data=>new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error("Could not load source or ordinary mask in editor."));img.src=data;});
        try{const[source,mask]=await Promise.all([decode(sourceData),existing?decode(existing):Promise.resolve(null)]);if(!editingMask||revision!==maskRevision)return;
            const temp=document.createElement("canvas");temp.width=maskCanvas.width;temp.height=maskCanvas.height;const ctx=temp.getContext("2d",{willReadFrequently:true});ctx.drawImage(source,0,0);maskSource=ctx.getImageData(0,0,temp.width,temp.height);
            maskPixels=new Uint8ClampedArray(temp.width*temp.height);
            if(mask){if(mask.naturalWidth!==temp.width||mask.naturalHeight!==temp.height)throw new Error("To edit an uploaded mask, its dimensions must match the source; the uploaded mask remains unchanged for generation.");ctx.clearRect(0,0,temp.width,temp.height);ctx.drawImage(mask,0,0);maskPixels=window.PortableMaskPixels.fromRgba(ctx.getImageData(0,0,temp.width,temp.height).data,temp.width,temp.height);}
            maskHistory=[];maskLoading=false;renderMask();updateControls();
        }catch(error){if(revision===maskRevision){closeMaskEditor();message(error.message,true);}}
    });
    function maskPosition(event){const rect=maskCanvas.getBoundingClientRect();return[Math.max(0,Math.min(maskCanvas.width-1e-6,(event.clientX-rect.left)*maskCanvas.width/rect.width)),Math.max(0,Math.min(maskCanvas.height-1e-6,(event.clientY-rect.top)*maskCanvas.height/rect.height))];}
    function maskPaint(event){if(!editingMask||maskLoading||active||maskPointer===null||event.pointerId!==maskPointer)return;try{const point=maskPosition(event);window.PortableMaskPixels.stroke(maskStroke.coverage,maskCanvas.width,maskCanvas.height,maskLast||point,point,maskStroke.size,255,1);const next=maskStroke.base.slice();window.PortableMaskPixels.select(next,maskStroke.coverage,maskCanvas.width,maskCanvas.height,maskStroke.target,maskStroke.strength);maskPixels=next;maskStroke.dirty=true;maskLast=point;renderMask();}catch(error){message(error.message,true);}}
    maskCanvas.addEventListener("pointerdown",event=>{if(!editingMask||maskLoading||active||maskPointer!==null||(event.button!==undefined&&event.button!==0))return;
        if(maskTool.value==="wand"){try{const threshold=Number(maskThreshold.value),buffer=Number(maskBuffer.value);if(!Number.isFinite(threshold)||threshold<0||threshold>255||!Number.isInteger(buffer)||buffer<0||buffer>64)throw new Error("Wand sensitivity must be 0–255 and buffer a whole number 0–64.");const point=maskPosition(event),selected=window.EditorMaskTools.select(maskSource.data,maskCanvas.width,maskCanvas.height,...point,threshold,buffer);const next=maskPixels.slice();window.PortableMaskPixels.select(next,selected,maskCanvas.width,maskCanvas.height,maskErase.checked?0:255,Number(maskStrength.value));maskSnapshot();maskPixels=next;renderMask();}catch(error){message(error.message,true);}return;}
        maskStroke={base:maskPixels.slice(),coverage:new Uint8ClampedArray(maskPixels.length),target:maskErase.checked?0:255,strength:Number(maskStrength.value),size:Number(maskSize.value),dirty:false};maskPointer=event.pointerId;maskLast=null;maskCanvas.setPointerCapture(maskPointer);maskPaint(event);updateControls();
    });
    maskCanvas.addEventListener("pointermove",maskPaint);
    maskCanvas.addEventListener("pointerup",event=>{if(event.pointerId===maskPointer){if(maskStroke?.dirty){maskHistory.push(maskStroke.base);if(maskHistory.length>5)maskHistory.shift();}maskStroke=null;maskPointer=maskLast=null;updateControls();}});
    for(const name of ["pointercancel","lostpointercapture"])maskCanvas.addEventListener(name,event=>{if(event.pointerId===maskPointer){if(maskStroke)maskPixels=maskStroke.base;maskStroke=null;maskPointer=maskLast=null;renderMask();updateControls();}});
    function numericControl(id, title, value, min, max, step) {
        const label = document.createElement("label"); label.textContent = title + " "; label.id = `${id}-label`;
        const input = document.createElement("input"); input.id = id; input.type = "number";
        input.value = value; input.min = min; input.max = max; input.step = step;
        label.append(input); imagePanel.append(label); return input;
    }
    const strength = numericControl("cosmo-image-strength", "Denoising strength", "0.75", "0", "1", "0.05");
    const maskBlur = numericControl("cosmo-mask-blur", "Mask blur (output pixels)", "4", "0", "64", "1");
    const invertLabel = document.createElement("label"); invertLabel.textContent = "Invert mask ";
    const invertMask = document.createElement("input"); invertMask.id = "cosmo-mask-invert"; invertMask.type = "checkbox";
    invertLabel.append(invertMask); imagePanel.append(invertLabel);
    controls.append(imagePanel);
    function clearUpload(kind) {
        const entry = uploads[kind]; ++entry.revision; entry.loading = false; entry.data = null;
        entry.width = entry.height = 0; entry.input.value = ""; entry.thumbnail.removeAttribute("src"); entry.thumbnail.hidden = true;
        if (kind === "initial") { if (editingMask) closeMaskEditor(); if (editingGuide) closeEditor(); clearUpload("mask"); clearUpload("guide"); }
        updateControls();
    }
    async function loadUpload(kind, file) {
        if (active || !imageInputsSupported || (kind === "mask" && !maskInputsSupported) || (kind === "guide" && !guideInputsSupported)) return;
        clearUpload(kind);
        if (!file) return;
        const entry = uploads[kind], revision = entry.revision;
        entry.loading = true; updateControls();
        try {
            if (!["image/png", "image/jpeg"].includes(file.type) || file.size > 16 * 1024 * 1024)
                throw new Error("Choose a PNG or JPEG no larger than 16 MiB.");
            if (kind === "guide" && file.type !== "image/png") throw new Error("Advanced Inpaint requires an RGBA PNG color guide.");
            const data = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = () => reject(new Error("Could not read the selected image."));
                reader.readAsDataURL(file);
            });
            const decoded = new Image();
            await new Promise((resolve, reject) => {
                decoded.onload = resolve; decoded.onerror = () => reject(new Error("The selected file is not a decodable image."));
                decoded.src = data;
            });
            if (!decoded.naturalWidth || !decoded.naturalHeight || decoded.naturalWidth * decoded.naturalHeight > 32 * 1024 * 1024)
                throw new Error("Image dimensions exceed the 32 megapixel input limit.");
            if (entry.revision !== revision) return;
            if (kind === "guide") {
                if (!uploads.initial.data || decoded.naturalWidth !== uploads.initial.width || decoded.naturalHeight !== uploads.initial.height)
                    throw new Error("Color guide and initial image must have matching dimensions.");
                // PNG color types 4 and 6 contain an explicit alpha channel. Do
                // not infer alpha from canvas, which also invents it for RGB.
                const bytes = atob(data.split(",")[1]);
                if (bytes.slice(1,4) !== "PNG" || ![4,6].includes(bytes.charCodeAt(25)))
                    throw new Error("Color guide PNG must contain an explicit alpha channel (RGBA or gray-alpha).");
                const canvas = document.createElement("canvas"); canvas.width = decoded.naturalWidth; canvas.height = decoded.naturalHeight;
                const context = canvas.getContext("2d", {willReadFrequently:true}); context.drawImage(decoded, 0, 0);
                const pixels = context.getImageData(0,0,canvas.width,canvas.height).data;
                let painted = false;
                for (let i=3; i<pixels.length; i+=4) if (pixels[i]) { painted = true; break; }
                if (!painted) throw new Error("Color guide contains no painted pixels.");
            }
            entry.width = decoded.naturalWidth; entry.height = decoded.naturalHeight;
            entry.data = data; entry.thumbnail.src = data; entry.thumbnail.hidden = false;
            message(`${kind === "initial" ? "Initial image" : kind === "mask" ? "Mask" : "Color guide"} ready (${decoded.naturalWidth} × ${decoded.naturalHeight}). Upload occurs only when you choose Generate.`);
        } catch (error) {
            if (entry.revision === revision) message(error.message, true);
        } finally {
            if (entry.revision === revision) { entry.loading = false; updateControls(); }
        }
    }

    function message(text, failed = false) {
        status.textContent = text;
        status.dataset.state = failed ? "error" : "";
    }
    function updateControls() {
        const readingImage = Object.values(uploads).some(entry => entry.loading);
        const available = !editingMask && !editingGuide && ready && active === null && !loadingModels && !savingDefaults && !readingImage;
        for (const [kind, entry] of Object.entries(uploads)) {
            entry.input.disabled = editingMask || editingGuide || !imageInputsSupported || !ready || active !== null || savingDefaults || (kind === "mask" && (!maskInputsSupported || !uploads.initial.data)) || (kind === "guide" && (!guideInputsSupported || !uploads.initial.data));
            entry.enabled.disabled = entry.input.disabled || !entry.data;
            entry.clear.disabled = editingMask || editingGuide || !imageInputsSupported || !ready || active !== null || (!entry.data && !entry.loading);
        }
        maskEdit.disabled = !maskInputsSupported || !available || !uploads.initial.data;
        for(const control of maskEditorControls)control.disabled=active!==null||!editingMask||((maskLoading||maskPointer!==null)&&control.id!=="cosmo-mask-cancel");
        editGuide.disabled = !guideInputsSupported || !available || !uploads.initial.data;
        for (const control of editorControls) control.disabled = active !== null || !editingGuide || (editorLoading && control.id !== "cosmo-guide-cancel");
        strength.disabled = !imageInputsSupported || !available || !uploads.initial.data;
        maskBlur.disabled = !(maskInputsSupported || guideInputsSupported) || !available || !(uploads.mask.data && uploads.mask.enabled.checked) && !(uploads.guide.data && uploads.guide.enabled.checked);
        invertMask.disabled = !maskInputsSupported || !available || !(uploads.mask.data && uploads.mask.enabled.checked);
        for (const id of editable) byId(id).disabled = !available;
        checkpoint.disabled = !available || models.size === 0;
        backend.disabled = !available;
        refresh.disabled = !available;
        saveDefaults.disabled = !available || !models.has(checkpoint.value);
        start.disabled = !available || !models.has(checkpoint.value) || !devices.has(backend.value);
        stop.disabled = !active || !active.sampling || active.stopping;
    }
    async function request(path, body) {
        let response, text;
        try {
            response = await fetch(path, body === undefined ? {cache: "no-store"} : {
                method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
            });
            text = await response.text();
        } catch (error) {
            error.transportFailed = true;
            throw error;
        }
        let value;
        try { value = text ? JSON.parse(text) : {}; } catch (_) { value = {message: text}; }
        if (!response.ok) {
            const error = new Error(value.message || value.detail || `HTTP ${response.status}`);
            error.status = response.status;
            throw error;
        }
        return value;
    }
    function integer(id, min, max) {
        const value = Number(byId(id).value);
        if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`${id}: enter an integer between ${min} and ${max}.`);
        return value;
    }
    async function loadModels(rescan = false) {
        loadingModels = true;
        updateControls();
        try {
            if (rescan) await request("/v1/sdapi/v1/refresh-checkpoints", {});
            const [result, options] = await Promise.all([
                request("/v1/sdapi/v1/checkpoints"), request("/v1/sdapi/v1/options"),
            ]);
            if (!Array.isArray(result.models) || result.models.some(model => typeof model.name !== "string")) throw new Error("Invalid checkpoint list from the native server");
            if (typeof options.sd_model_checkpoint !== "string") throw new Error("Invalid saved checkpoint from the native server");
            const previous = checkpoint.value;
            models = new Set(result.models.map(model => model.name));
            checkpoint.replaceChildren(new Option("Select a complete checkpoint", ""));
            for (const name of models) checkpoint.add(new Option(name, name));
            const saved = options.sd_model_checkpoint;
            if (models.has(saved)) checkpoint.value = saved;
            else if (!saved && models.has(previous)) checkpoint.value = previous;
            else if (!saved && models.size === 1) checkpoint.value = [...models][0];
            modelStatus.textContent = models.size ? `${models.size} checkpoint(s) indexed. The selected file must contain the components required by its architecture.` : "No checkpoints found. Start sdkit with --ckpt-dir pointing to your model directory, then refresh.";
            if (saved && !models.has(saved)) modelStatus.textContent = `Saved checkpoint ${saved} is unavailable in the model index. Select an available checkpoint or update its directory in Settings.`;
        } finally { loadingModels = false; updateControls(); }
    }
    function endJob(job) {
        if (active !== job) return;
        clearTimeout(job.timer);
        active = null;
        updateControls();
    }
    async function poll(job) {
        if (active !== job) return;
        try {
            const state = await request("/v1/internal/progress", {id_task: job.id, live_preview: false});
            if (active !== job) return;
            if (state.completed && job.transportFailed) {
                endJob(job);
                message("The generation connection was lost. The server has finished, but its image response was not received. You may submit a new request.", true);
                return;
            }
            if (!state.completed && Number(state.current_step) > 0) {
                job.sampling = true;
                progress.max = Number(state.total_steps) || job.steps;
                progress.value = Number(state.current_step);
                if (!job.stopping && !job.transportFailed) message(`Sampling ${state.current_step} / ${state.total_steps}. The image still needs VAE decoding afterward.`);
            }
            updateControls();
        } catch (error) {
            if (active !== job) return;
            if (error.status !== 404 && !job.transportFailed) message(`Generation is still pending; progress is unavailable (${error.message}).`);
        }
        if (active === job) job.timer = setTimeout(() => poll(job), 750);
    }
    function showImage(data, job) {
        if (typeof data !== "string" || !data.startsWith("iVBORw0KGgo")) throw new Error("The native server did not return a PNG image.");
        const card = document.createElement("figure");
        card.className = "cpp-generated-image";
        const image = document.createElement("img");
        image.src = `data:image/png;base64,${data}`;
        image.alt = `Generated image: ${job.prompt}`;
        const caption = document.createElement("figcaption");
        caption.textContent = `${job.model} · seed ${job.seed} · ${job.backend}`;
        const download = document.createElement("a");
        download.href = image.src;
        download.download = `image-${job.seed}.png`;
        download.textContent = "Download PNG";
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "Remove";
        remove.addEventListener("click", () => card.remove());
        card.append(image, caption, download, remove);
        preview.prepend(card);
        byId("initial-text")?.remove();
    }
    function imageRecipe() {
        const width = integer("width", 64, 2048), height = integer("height", 64, 2048);
        if (width % 64 || height % 64) throw new Error("Width and height must be multiples of 64.");
        const cfg = Number(byId("guidance_scale").value);
        if (!Number.isFinite(cfg) || cfg < 0 || cfg > 30) throw new Error("Guidance must be between 0 and 30.");
        return {
            prompt: byId("prompt").value.trim(), negative_prompt: byId("negative_prompt").value,
            width, height, steps: integer("steps", 1, 200), cfg_scale: cfg,
            seed: integer("seed", -1, 2147483647), sampler_name: sampler.value, scheduler: "discrete",
        };
    }
    async function persistDefaults() {
        if (!ready || active || savingDefaults || loadingModels) return;
        try {
            if (!models.has(checkpoint.value)) throw new Error("Select an indexed checkpoint before saving defaults.");
            const recipe = imageRecipe();
            savingDefaults = true;
            updateControls();
            const result = await request("/v1/sdapi/v1/settings", {
                options: {sd_model_checkpoint: checkpoint.value}, inference: {image: recipe},
            });
            if (result.saved?.options?.sd_model_checkpoint !== checkpoint.value ||
                !result.saved?.inference?.image) throw new Error("The native server did not confirm saved generation defaults.");
            message("Checkpoint and generation defaults saved for this application. The UI, shell and API use the same configuration file.");
        } catch (error) { message(`Could not save generation defaults: ${error.message}`, true); }
        finally { savingDefaults = false; updateControls(); }
    }
    async function generate() {
        if (!ready || active || editingGuide || editingMask || Object.values(uploads).some(entry => entry.loading)) return;
        let job;
        try {
            const prompt = byId("prompt").value.trim();
            if (!prompt) throw new Error("Enter a prompt first.");
            if (!models.has(checkpoint.value)) throw new Error("Select an indexed complete checkpoint.");
            if (!devices.has(backend.value)) throw new Error("Select an available compute device.");
            const recipe = imageRecipe();
            const imageInputs = {};
            if (uploads.initial.data) {
                const denoise = Number(strength.value);
                if (!Number.isFinite(denoise) || denoise < 0 || denoise > 1) throw new Error("Denoising strength must be between 0 and 1.");
                imageInputs.init_images = [uploads.initial.data.split(",")[1]];
                imageInputs.denoising_strength = denoise;
                if (uploads.mask.data && uploads.mask.enabled.checked) {
                    const blur = Number(maskBlur.value);
                    if (!Number.isFinite(blur) || blur < 0 || blur > 64) throw new Error("Mask blur must be between 0 and 64.");
                    imageInputs.mask = uploads.mask.data.split(",")[1];
                    imageInputs.mask_blur = blur;
                    imageInputs.inpainting_mask_invert = invertMask.checked ? 1 : 0;
                }
                if (uploads.guide.data && uploads.guide.enabled.checked) {
                    if (!(denoise > 0 && denoise < 1)) throw new Error("Advanced Inpaint requires denoising strength strictly between 0 and 1.");
                    imageInputs.inpaint_color_guide = uploads.guide.data.split(",")[1];
                    const blur = Number(maskBlur.value);
                    if (!Number.isFinite(blur) || blur < 0 || blur > 64) throw new Error("Mask blur must be between 0 and 64.");
                    imageInputs.mask_blur = blur;
                }
            }
            let seed = recipe.seed;
            if (seed === -1) seed = crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
            const id = `cosmo-${Date.now()}-${crypto.getRandomValues(new Uint32Array(1))[0]}`;
            job = {id, prompt, model: checkpoint.value, seed, steps: recipe.steps, backend: backend.value, sampling: false, stopping: false, transportFailed: false, timer: null};
            active = job;
            updateControls();
            progress.removeAttribute("value");
            message("Loading the checkpoint and preparing generation. Stop becomes available after the first sampling step completes.");
            const body = {
                ...recipe, ...imageInputs, force_task_id: id, seed, batch_size: 1, backend: job.backend,
                override_settings: {sd_model_checkpoint: job.model, forge_additional_modules: [], live_previews_enable: false},
            };
            job.timer = setTimeout(() => poll(job), 500);
            const result = await request(imageInputs.init_images ? "/v1/sdapi/v1/img2img" : "/v1/sdapi/v1/txt2img", body);
            if (!Array.isArray(result.images)) throw new Error("Native generation returned no image list.");
            if (!result.images.length) {
                if (!job.stopping) throw new Error("Generation completed without an image.");
                message("Generation stopped.");
            } else {
                for (const data of result.images) showImage(data, job);
                progress.max = 1;
                progress.value = 1;
                message(job.stopping ? "The image completed before the stop request took effect." : "Generation complete. Download the PNG beneath the image.");
            }
            endJob(job);
        } catch (error) {
            if (job && error.transportFailed) {
                job.transportFailed = true;
                message("Connection to the generation request was lost. Waiting for the server to finish before allowing another request; reload only after checking the server.", true);
            } else {
                if (job) endJob(job);
                message(error.message || "Generation failed.", true);
            }
        }
    }
    async function cancel() {
        const job = active;
        if (!job || !job.sampling || job.stopping) return;
        job.stopping = true;
        updateControls();
        message("Stop requested. Waiting for the native generation call to return…");
        try { await request("/v1/sdapi/v1/interrupt", {id_task: job.id}); }
        catch (error) {
            if (active !== job) return;
            job.stopping = false;
            message(`Stop request failed: ${error.message}. The generation request remains active.`, true);
            updateControls();
        }
    }
    start.addEventListener("click", generate);
    stop.addEventListener("click", cancel);
    checkpoint.addEventListener("change", updateControls);
    saveDefaults.addEventListener("click", persistDefaults);
    backend.addEventListener("change", () => {
        updateControls();
        const selected = devices.get(backend.value);
        if (selected) message(`Selected ${selected.label}. This request uses ${selected.selector}.`);
    });
    refresh.addEventListener("click", () => loadModels(true).catch(error => message(error.message, true)));
    byId("clear-all-previews").addEventListener("click", () => preview.replaceChildren());
    window.addEventListener("beforeunload", event => { if (active) { event.preventDefault(); event.returnValue = ""; } });
    updateControls();
    (async () => {
        try {
            const state = await window.CppKiosk.ready;
            if (state.capabilities.txt2img !== true || state.capabilities.max_concurrent_generations !== 1) throw new Error("Unsupported native generation protocol");
            imageInputsSupported = state.capabilities.img2img_ui === true;
            maskInputsSupported = imageInputsSupported && state.capabilities.uploaded_mask_inpainting_ui === true;
            guideInputsSupported = imageInputsSupported && state.capabilities.advanced_inpaint_ui === true;
            uploads.mask.label.hidden = uploads.mask.clear.hidden = uploads.mask.enableLabel.hidden = maskEdit.hidden = !maskInputsSupported;
            uploads.guide.label.hidden = uploads.guide.clear.hidden = uploads.guide.enableLabel.hidden = editGuide.hidden = !guideInputsSupported;
            byId("cosmo-mask-blur-label").hidden = !(maskInputsSupported || guideInputsSupported);
            invertLabel.hidden = !maskInputsSupported;
            if (!maskInputsSupported) imageHelp.textContent = "Optional PNG or JPEG initial image. Output uses the selected width and height. Uploaded-mask inpainting is unavailable on this server.";
            imagePanel.hidden = !imageInputsSupported;
            const [result, config] = await Promise.all([
                request("/v1/sdapi/v1/backend-devices"), request("/v1/sdapi/v1/config"),
            ]);
            if (!Array.isArray(result.devices)) throw new Error("Invalid device list from the native server");
            const compute = config.effective?.compute;
            if (!compute || !["cpu", "webgpu"].includes(compute.backend)) throw new Error("Invalid effective compute configuration");
            const recipe = config.effective?.inference?.image;
            if (!recipe || recipe.scheduler !== "discrete") throw new Error("Invalid image defaults from the native configuration");
            const fields = {prompt: "prompt", negative_prompt: "negative_prompt", width: "width", height: "height",
                steps: "steps", cfg_scale: "guidance_scale", seed: "seed", sampler_name: "sampler"};
            for (const [key, id] of Object.entries(fields)) {
                if (recipe[key] === undefined) throw new Error(`Missing image default: ${key}`);
                byId(id).value = recipe[key];
            }
            imageRecipe();
            for (const id of ["width", "height"]) byId(`${id}-value`).textContent = `${byId(id).value} px`;
            for (const device of result.devices) {
                const kind = String(device.backend).toLowerCase();
                if (!["cpu", "webgpu"].includes(kind)) continue;
                if (typeof device.selector !== "string" || !device.selector || devices.has(device.selector))
                    throw new Error("Missing or duplicate native device selector");
                let label;
                if (kind === "cpu") label = "GGML CPU";
                else {
                    if (typeof device.software !== "boolean" || typeof device.provider !== "string")
                        throw new Error("WebGPU device classification is unavailable");
                    const type = device.software ? "Software CPU" : device.type === "integrated-gpu" ? "Integrated GPU" : device.type === "gpu" ? "Discrete GPU" : "Unknown adapter type";
                    label = `${type} · ${device.description || device.selector} · ${device.provider}`;
                    if (device.memory_known === true && Number(device.memory_total) > 0)
                        label += ` · ${(Number(device.memory_total) / 1073741824).toFixed(1)} GiB`;
                }
                label += ` [${device.selector}]`;
                devices.set(device.selector, {...device, kind, label});
                backend.add(new Option(label, device.selector));
            }
            if (!backend.options.length) throw new Error("No supported native image device is available");
            const requested = compute.device || "auto";
            const choices = [...devices.values()].filter(device => device.kind === compute.backend);
            let selected;
            if (requested === "auto") {
                selected = compute.backend === "cpu" ? choices[0] : choices.find(device => device.selector === result.default_webgpu_selector);
            } else {
                const matches = choices.filter(device => device.selector === requested ||
                    (device.stable_id_available === true && device.stable_id === requested));
                if (matches.length === 1) selected = matches[0];
            }
            if (selected) backend.value = selected.selector;
            else {
                const unavailable = new Option(`Configured device unavailable: ${requested}`, "");
                unavailable.disabled = true;
                backend.add(unavailable);
                backend.value = "";
            }
            await loadModels();
            ready = true;
            updateControls();
            message(selected ? "Ready to generate using a complete checkpoint. Optional image inputs appear when supported by this server. Models load when you choose Generate." :
                "The configured device is unavailable. Choose an available compute device or update Settings; no replacement was selected.", !selected);
        } catch (error) {
            ready = false;
            updateControls();
            message(`${error.message}. Generation is disabled.`, true);
        }
    })();
})();
