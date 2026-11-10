(() => {
    "use strict";
    const MODES = ["draw", "inpaint", "advanced"];
    const MAX_PIXELS = 16 * 1024 * 1024;
    const main = document.querySelector('.generation-controls');
    const surface = document.getElementById('image-editor-canvas');
    if (!main && !surface) return;

    function canvas(width, height) {
        const result = document.createElement('canvas');
        result.width = width; result.height = height;
        return result;
    }
    function blob(image) {
        return new Promise((resolve, reject) => image.toBlob(value => value ? resolve(value) : reject(Error('Could not encode the canvas.')), 'image/png'));
    }
    async function image(value) {
        const url = URL.createObjectURL(value), result = new Image();
        try { result.src = url; await result.decode(); return result; }
        finally { URL.revokeObjectURL(url); }
    }
    async function storage(operation, value) {
        const db = await new Promise((resolve, reject) => {
            const open = indexedDB.open('EasyDiffusionCanvas', 1);
            open.onupgradeneeded = () => open.result.createObjectStore('workspace');
            open.onsuccess = () => resolve(open.result);
            open.onerror = () => reject(open.error);
        });
        try {
            return await new Promise((resolve, reject) => {
                const tx = db.transaction('workspace', operation === 'read' ? 'readonly' : 'readwrite');
                const store = tx.objectStore('workspace');
                const request = operation === 'read' ? store.get('input') : value ? store.put(value, 'input') : store.delete('input');
                tx.oncomplete = () => resolve(request.result);
                tx.onerror = tx.onabort = () => reject(tx.error || Error('Canvas storage failed.'));
            });
        } finally { db.close(); }
    }
    async function sourceRecord(file) {
        if (file.size > 32 * 1024 * 1024) throw Error('Source images must be under 32 MB.');
        const source = await image(file), {naturalWidth:width, naturalHeight:height} = source;
        if (!width || !height || width * height > MAX_PIXELS) throw Error('Source images must contain between 1 and 16 million pixels.');
        const base = canvas(width, height), ctx = base.getContext('2d');
        ctx.drawImage(source, 0, 0);
        return {width, height, base:await blob(base), layers:{}, mode:'advanced', enabled:true};
    }
    async function compose(record) {
        const base = canvas(record.width, record.height), layers = {};
        base.getContext('2d').drawImage(await image(record.base), 0, 0);
        for (const mode of MODES) {
            layers[mode] = canvas(record.width, record.height);
            if (record.layers[mode]) layers[mode].getContext('2d').drawImage(await image(record.layers[mode]), 0, 0);
        }
        // Draw changes the base; ordinary inpaint and color guidance remain
        // independent inputs, regardless of which editor was visited last.
        base.getContext('2d').drawImage(layers.draw, 0, 0);
        let mask = null;
        {
            const pixels = layers.inpaint.getContext('2d').getImageData(0, 0, record.width, record.height);
            let painted = false;
            for (let offset = 0; offset < pixels.data.length; offset += 4) {
                const strength = pixels.data[offset + 3];
                painted ||= strength > 0;
                pixels.data[offset] = pixels.data[offset + 1] = pixels.data[offset + 2] = strength;
                pixels.data[offset + 3] = 255;
            }
            if (painted) {
                const output=canvas(record.width,record.height);
                output.getContext('2d').putImageData(pixels, 0, 0);
                mask = output.toDataURL('image/png');
            }
        }
        const guidePixels=layers.advanced.getContext('2d').getImageData(0,0,record.width,record.height).data;
        let hasGuide=false;
        for(let i=3;i<guidePixels.length;i+=4)if(guidePixels[i]){hasGuide=true;break;}
        const original=base.toDataURL('image/png');
        return {init_image:original, init_image_mask:mask,
            inpaint_color_guide:hasGuide?layers.advanced.toDataURL('image/png'):null,
            previews:{draw:original,inpaint:mask?ImagePreviewTabs.render(base,layers.inpaint,'inpaint'):original,
                advanced:hasGuide?ImagePreviewTabs.render(base,layers.advanced,'advanced'):original}};
    }
    function element(tag, properties, parent) {
        const result = Object.assign(document.createElement(tag), properties);
        if (parent) parent.append(result);
        return result;
    }
    function button(parent, id, text, action) {
        const result = element('button', {id, type:'button', textContent:text, className:'tertiaryButton'}, parent);
        result.addEventListener('click', action);
        return result;
    }
    function field(parent, id, label, type, value, attributes = {}) {
        const holder = element('label', {htmlFor:id, textContent:label}, parent);
        const input = element('input', {id, type, ...attributes, value}, holder);
        if (type==='checkbox') {holder.className='cpp-canvas-checkbox';holder.prepend(input);}
        return input;
    }
    element('style', {textContent:`
        .cpp-canvas-controls,.cpp-canvas-actions{display:flex;flex-wrap:wrap;gap:12px;align-items:center;min-width:0}
        .cpp-canvas-actions a{display:inline-flex;align-items:center;padding:7px 10px;border:1px solid var(--background-color4,#7776);border-radius:6px;background:var(--background-color3,#7772);color:inherit;text-decoration:none}
        #cpp-canvas-panel .cpp-canvas-checkbox{display:flex;flex-direction:row;align-items:center;gap:8px}
        #cpp-canvas-panel .cpp-canvas-checkbox input{width:auto;flex:0 0 auto;margin:0}
        .cpp-canvas-controls label{display:flex;flex-direction:column;gap:4px;min-width:0}
        .cpp-canvas-controls input[type=file]{max-width:100%}
        .cpp-canvas-palette{display:flex;flex-wrap:wrap;gap:5px}
        .cpp-canvas-palette button{width:30px;height:30px;border:2px solid #888;border-radius:5px;padding:0}
        .cpp-canvas-palette button[aria-pressed=true]{outline:2px solid currentColor;outline-offset:2px}
        .canvas-toolbar{display:flex;flex-wrap:wrap;gap:8px}
        .canvas-toolbar [aria-pressed=true]{outline:2px solid currentColor}
        #image-editor-canvas{display:block;width:100%;max-width:768px;height:auto;touch-action:none;cursor:crosshair;background:#fff;border:1px solid #777;margin:12px auto}
        #cpp-canvas-panel{min-width:0}.cpp-canvas-controls>*{max-width:100%}
    `}, document.head);

    let loaded = false, current = null, prepared = null, useInput, useInpaint, useGuide, mainStatus;
    window.CppCanvas = {
        applyToRequest(request) {
            if (!loaded) throw Error('Wait for the canvas input to finish loading.');
            if (current?.enabled && useInput?.checked && prepared) {
                request.init_image = prepared.init_image;
                if (useInpaint.checked && prepared.init_image_mask) request.init_image_mask = prepared.init_image_mask;
                else delete request.init_image_mask;
                if (useGuide.checked && prepared.inpaint_color_guide) request.inpaint_color_guide=prepared.inpaint_color_guide;
                else delete request.inpaint_color_guide;
                request.prompt_strength = Number(document.getElementById('cpp-canvas-strength').value);
                if(request.inpaint_color_guide && !(request.prompt_strength>0&&request.prompt_strength<1))
                    throw Error('Color-weighted inpaint needs prompt strength between 0 and 1, not including the endpoints.');
            }
        },
        ready:null,
    };

    async function initializeMain() {
        const panel = element('section', {id:'cpp-canvas-panel',className:'panel-box'});
        document.getElementById('width').closest('section').after(panel);
        element('h3', {textContent:'Image Loader'}, panel);
        const file = field(panel,'cpp-canvas-input','Initial image','file','',{accept:'image/*'});
        file.disabled = true;
        const previews = new ImagePreviewTabs(panel,'cpp-input-preview');
        const actions = element('div', {className:'cpp-canvas-actions'}, panel);
        for (const [mode,label] of [['draw','Draw'],['inpaint','Inpaint'],['advanced','Advanced Inpaint']]) {
            const link = element('a', {id:`cpp-canvas-open-${mode}`,href:`/cpp-ui/canvas?mode=${mode}`,textContent:`Edit ${label}`,className:'tertiaryButton'}, actions);
            link.addEventListener('click',event=>{if(!loaded){event.preventDefault();mainStatus.textContent='Wait for the canvas input to finish loading.';}});
        }
        useInput = field(panel,'cpp-canvas-use','Use canvas image and mask for Generate','checkbox','');
        useInput.disabled = true;
        useInpaint = field(panel,'cpp-canvas-inpaint-use','Use ordinary inpaint mask (first influence)','checkbox','');
        useGuide = field(panel,'cpp-canvas-guide-use','Use color-weighted inpaint guidance (second influence)','checkbox','');
        useInpaint.disabled=useGuide.disabled=true;
        const strength=field(panel,'cpp-canvas-strength','Prompt strength','range','.65',{min:'0',max:'1',step:'.01'});
        const strengthValue=element('output',{htmlFor:'cpp-canvas-strength',textContent:Number(strength.value).toFixed(2)},strength.parentElement);
        strength.addEventListener('input',()=>{strengthValue.textContent=Number(strength.value).toFixed(2);});
        mainStatus = element('p', {id:'cpp-canvas-status',textContent:'Loading canvas input…'}, panel);
        mainStatus.setAttribute('role','status');
        function setBusy(value) {
            loaded = !value;
            file.disabled = clear.disabled = value;
            useInput.disabled = value || !current;
            useInpaint.disabled = value || !prepared?.init_image_mask;
            useGuide.disabled = value || !prepared?.inpaint_color_guide;
        }
        function display() {
            useInput.disabled = !current;
            useInput.checked = Boolean(current?.enabled);
            useInpaint.checked=Boolean(prepared?.init_image_mask)&&current.inpaintEnabled!==false;
            useGuide.checked=Boolean(prepared?.inpaint_color_guide)&&current.guideEnabled!==false;
            previews.setImages(prepared?.previews);
            mainStatus.textContent = current ? `${current.width} × ${current.height} · Draw base, ordinary mask, and color guidance remain separate. Inpaint is processed first, then color guidance in one generation. Output size uses Image Size settings.` : 'No canvas image. Load a source or open an editor to start on a blank canvas.';
        }
        async function update(record) {
            setBusy(true);
            try {
                const result = record ? await compose(record) : null;
                await storage('write', record);
                current = record; prepared = result; display();
            } catch (error) {mainStatus.textContent = error.message;}
            finally {setBusy(false);}
        }
        file.addEventListener('change', async () => {
            if (!loaded || !file.files[0]) return;
            setBusy(true);mainStatus.textContent='Loading source image…';
            try {await update(await sourceRecord(file.files[0]));}
            catch (error) {mainStatus.textContent = error.message;}
            finally {setBusy(false);}
        });
        useInput.addEventListener('change', () => {if (current) update({...current, enabled:useInput.checked});});
        useInpaint.addEventListener('change',()=>{if(current)update({...current,inpaintEnabled:useInpaint.checked});});
        useGuide.addEventListener('change',()=>{if(current)update({...current,guideEnabled:useGuide.checked});});
        const clear = button(actions,'cpp-canvas-clear','Clear canvas input',()=>{file.value='';update(null);});
        clear.disabled = true;
        current = await storage('read');
        if (current) prepared = await compose(current);
        display();if(current?.mode)previews.select(current.mode);setBusy(false);
    }

    async function initializeCanvas() {
        const controls = document.getElementById('image-editor-page-content');
        surface.before(controls);
        controls.className = 'cpp-canvas-controls';
        const source = field(controls,'canvas-source','Source image','file','',{accept:'image/*'});
        const color = field(controls,'canvas-color','Guidance color','color','#ff0000');
        const size = field(controls,'canvas-brush-size','Brush size (px)','range','1',{min:'1',max:'128',step:'1'});
        const sensitivity = field(controls,'canvas-wand-sensitivity','Magic wand sensitivity','range','30',{min:'0',max:'255',step:'1'});
        const sensitivityValue = element('output',{htmlFor:'canvas-wand-sensitivity',textContent:'30'},sensitivity.parentElement);
        const buffer = field(controls,'canvas-wand-buffer','Selection buffer (px)','range','0',{min:'0',max:'64',step:'1'});
        const bufferValue = element('output',{htmlFor:'canvas-wand-buffer',textContent:'0 px'},buffer.parentElement);
        const opacity = field(controls,'canvas-opacity','Color / mask strength','range','50',{min:'1',max:'100',step:'1'});
        const opacityValue = element('output', {htmlFor:'canvas-opacity',textContent:'50%'}, opacity.parentElement);
        const toolHolder = element('label',{htmlFor:'canvas-tool',textContent:'Tool'},controls);
        const tool = element('select',{id:'canvas-tool'},toolHolder);
        for (const [value,text] of [['brush','Brush'],['erase','Erase'],['wand','Magic Wand']]) element('option',{value,textContent:text},tool);
        const palette = element('div',{className:'cpp-canvas-palette'},controls);
        for (const value of ['#ff0000','#ff9900','#ffff00','#00ff00','#00ffff','#0000ff','#ff00ff','#000000','#ffffff']) {
            const swatch = button(palette,'','',()=>{color.value=value;refresh();});
            swatch.style.background = value; swatch.dataset.color = value;
            swatch.setAttribute('aria-label',`Use ${value}`);
        }
        const status = element('p',{id:'canvas-status'},controls);status.style.flexBasis='100%';status.setAttribute('role','status');
        let mode = new URLSearchParams(location.search).get('mode') || 'advanced';
        if (!MODES.includes(mode)) mode='advanced';
        let base, layers, stroke, working, pointer=null, last=null, busy=true;
        let preferences={inpaintEnabled:true,guideEnabled:true};
        let undo={}, redo={};
        const undoButton=button(controls,'canvas-undo','Undo',()=>rewind(undo,redo));
        const redoButton=button(controls,'canvas-redo','Redo',()=>rewind(redo,undo));
        button(controls,'canvas-clear','Clear this mode',()=>{
            if (busy || pointer!==null) return;
            snapshot();layers[mode].getContext('2d').clearRect(0,0,surface.width,surface.height);refresh();
        });
        function snapshot() {
            const limit=Math.max(1,Math.min(20,Math.floor(64*1024*1024/(surface.width*surface.height*4))));
            undo[mode].push(layers[mode].getContext('2d').getImageData(0,0,surface.width,surface.height));
            if (undo[mode].length>limit) undo[mode].shift();
            redo[mode]=[];
        }
        function rewind(from,to) {
            if (busy || pointer!==null || !from[mode].length) return;
            const ctx=layers[mode].getContext('2d');
            to[mode].push(ctx.getImageData(0,0,surface.width,surface.height));ctx.putImageData(from[mode].pop(),0,0);refresh();
        }
        function refresh() {
            if (!base) return;
            opacityValue.textContent=`${opacity.value}%`;
            opacity.parentElement.firstChild.nodeValue=mode==='advanced'?'Color hint weight':mode==='inpaint'?'Mask strength':'Paint opacity';
            color.parentElement.firstChild.nodeValue=mode==='advanced'?'Inpaint color hint':'Paint color';
            color.parentElement.style.display=mode==='inpaint'?'none':'';
            palette.style.display=mode==='inpaint'?'none':'flex';
            sensitivityValue.textContent=sensitivity.value;
            bufferValue.textContent=`${buffer.value} px`;
            palette.querySelectorAll('button').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.color===color.value)));
            for (const name of MODES) document.getElementById(name==='advanced'?'canvas-advanced-inpaint':`canvas-${name}`).setAttribute('aria-pressed',String(name===mode));
            undoButton.disabled=!undo[mode].length;redoButton.disabled=!redo[mode].length;
            const ctx=surface.getContext('2d'), work=working.getContext('2d');
            work.clearRect(0,0,surface.width,surface.height);work.drawImage(layers[mode],0,0);
            if (pointer!==null) {
                work.globalAlpha=Number(opacity.value)/100;
                work.globalCompositeOperation=tool.value==='erase'?'destination-out':'source-over';
                work.drawImage(stroke,0,0);work.globalAlpha=1;work.globalCompositeOperation='source-over';
            }
            ctx.clearRect(0,0,surface.width,surface.height);ctx.drawImage(base,0,0);
            if (mode==='inpaint') ctx.globalAlpha=.55;
            ctx.drawImage(working,0,0);ctx.globalAlpha=1;
            status.textContent=`${surface.width} × ${surface.height} · ${mode==='advanced'?'Color-weighted inpaint: this is a guidance overlay, not a drawing. Opacity weights the color hint; ordinary inpaint stays separate and is processed first.':mode==='inpaint'?'Inpaint: paint the region to change; source colors stay unchanged.':'Draw: color strokes change the base image.'}`;
        }
        async function load(record) {
            busy=true;
            try {
                const original=await image(record.base), next={};
                for (const name of MODES) {
                    next[name]=canvas(record.width,record.height);
                    if (record.layers[name]) next[name].getContext('2d').drawImage(await image(record.layers[name]),0,0);
                }
                surface.width=record.width;surface.height=record.height;
                base=canvas(record.width,record.height);base.getContext('2d').drawImage(original,0,0);
                layers=next;stroke=canvas(record.width,record.height);working=canvas(record.width,record.height);
                preferences={inpaintEnabled:record.inpaintEnabled!==false,guideEnabled:record.guideEnabled!==false};
                undo={draw:[],inpaint:[],advanced:[]};redo={draw:[],inpaint:[],advanced:[]};refresh();
            } finally {busy=false;}
        }
        function position(event) {
            const bounds=surface.getBoundingClientRect();
            return {x:(event.clientX-bounds.left)*surface.width/bounds.width,y:(event.clientY-bounds.top)*surface.height/bounds.height};
        }
        function paint(point) {
            const ctx=stroke.getContext('2d');ctx.fillStyle=ctx.strokeStyle=mode==='inpaint'?'#ff55aa':color.value;
            ctx.lineWidth=Number(size.value);ctx.lineCap=ctx.lineJoin='round';ctx.beginPath();
            if (last) {ctx.moveTo(last.x,last.y);ctx.lineTo(point.x,point.y);ctx.stroke();}
            else {ctx.arc(point.x,point.y,Number(size.value)/2,0,Math.PI*2);ctx.fill();}
            last=point;refresh();
        }
        surface.addEventListener('pointerdown',event=>{
            if (busy || pointer!==null || event.button!==0) return;
            if (tool.value==='wand') {
                event.preventDefault();snapshot();
                const point=position(event),width=surface.width,height=surface.height;
                const selected=EditorMaskTools.select(base.getContext('2d').getImageData(0,0,width,height).data,
                    width,height,point.x,point.y,Number(sensitivity.value),Number(buffer.value));
                const fill=canvas(width,height),ctx=fill.getContext('2d'),pixels=ctx.createImageData(width,height);
                const hex=mode==='inpaint'?'#ff55aa':color.value;
                const rgb=[1,3,5].map(offset=>parseInt(hex.slice(offset,offset+2),16));
                for(let i=0;i<selected.length;i++) if(selected[i]) {
                    pixels.data[i*4]=rgb[0];pixels.data[i*4+1]=rgb[1];pixels.data[i*4+2]=rgb[2];
                    pixels.data[i*4+3]=Math.round(255*Number(opacity.value)/100);
                }
                ctx.putImageData(pixels,0,0);layers[mode].getContext('2d').drawImage(fill,0,0);refresh();return;
            }
            event.preventDefault();pointer=event.pointerId;last=null;
            stroke.getContext('2d').clearRect(0,0,surface.width,surface.height);
            surface.setPointerCapture(pointer);paint(position(event));
        });
        surface.addEventListener('pointermove',event=>{if (event.pointerId===pointer) paint(position(event));});
        function finish(event,commit) {
            if (event.pointerId!==pointer) return;
            if (commit) {
                snapshot();const ctx=layers[mode].getContext('2d');
                ctx.globalAlpha=Number(opacity.value)/100;ctx.globalCompositeOperation=tool.value==='erase'?'destination-out':'source-over';
                ctx.drawImage(stroke,0,0);ctx.globalAlpha=1;ctx.globalCompositeOperation='source-over';
            }
            const old=pointer;pointer=null;last=null;
            if (surface.hasPointerCapture(old)) surface.releasePointerCapture(old);
            refresh();
        }
        surface.addEventListener('pointerup',event=>finish(event,true));
        surface.addEventListener('pointercancel',event=>finish(event,false));
        surface.addEventListener('lostpointercapture',event=>finish(event,false));
        for (const name of MODES) document.getElementById(name==='advanced'?'canvas-advanced-inpaint':`canvas-${name}`).addEventListener('click',()=>{if(!busy&&pointer===null){mode=name;refresh();}});
        for (const input of [color,size,opacity,tool,sensitivity,buffer]) input.addEventListener('input',refresh);
        source.addEventListener('change',async()=>{try{if(source.files[0])await load(await sourceRecord(source.files[0]));}catch(error){status.textContent=error.message;}});
        document.getElementById('canvas-cancel').addEventListener('click',()=>location.assign('/cpp-ui'));
        document.getElementById('canvas-generate').addEventListener('click',async()=>{
            if (busy || pointer!==null) return;
            busy=true;
            try {
                const serialized={};for(const name of MODES)serialized[name]=await blob(layers[name]);
                await storage('write',{width:surface.width,height:surface.height,base:await blob(base),layers:serialized,mode,enabled:true,
                    inpaintEnabled:mode==='inpaint'?true:preferences.inpaintEnabled,
                    guideEnabled:mode==='advanced'?true:preferences.guideEnabled});
                location.assign('/cpp-ui');
            } catch(error){status.textContent=error.message;busy=false;}
        });
        const saved=await storage('read');
        if(saved) await load(saved);
        else {
            const blank=canvas(512,512),ctx=blank.getContext('2d');ctx.fillStyle='#ffffff';ctx.fillRect(0,0,512,512);
            await load({width:512,height:512,base:await blob(blank),layers:{}});
        }
    }
    window.CppCanvas.ready=(surface?initializeCanvas():initializeMain()).catch(error=>{
        const target=document.getElementById('canvas-status') || mainStatus;
        if(target)target.textContent=`Canvas unavailable: ${error.message}`;
        console.error('Canvas workspace:',error);
    }).finally(()=>{loaded=true;});
})();
