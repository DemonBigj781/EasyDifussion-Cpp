(() => {
    "use strict";
    const randomSeed = () => Math.floor(Math.random() * 2147483647);
    function frameTimes(start, end, fps) {
        if (!Number.isFinite(fps) || fps < .1 || fps > 30) throw Error("FPS must be between 0.1 and 30.");
        if (![start,end].every(Number.isFinite) || start < 0 || end <= start) throw Error("Invalid animation time range.");
        const count = Math.ceil((end-start)*fps);
        if (count > 120) throw Error("Animation limit is 120 frames; shorten the range or reduce FPS.");
        return Array.from({length:count}, (_,i) => start + i/fps);
    }
    function rabbitRequests(base, settings) {
        const max = Number(settings.max ?? 16);
        if (!Number.isInteger(max) || max < 1 || max > 512) throw Error("Rabbit Hole maximum must be 1–512 images.");
        function range(count, step, mid, minimum, maximum, whole = false) {
            count = Number(count ?? 1); step = Number(step ?? 0); mid = Number(mid);
            if (!Number.isInteger(count) || count < 1 || count > 32 || !Number.isFinite(step) || !Number.isFinite(mid))
                throw Error("Invalid Rabbit Hole range (counts must be 1–32).");
            return [...new Set(Array.from({length:count}, (_,i) => {
                const value = Math.min(maximum, Math.max(minimum, mid+(i-(count-1)/2)*step));
                return whole ? Math.round(value) : Number(value.toFixed(4));
            }))];
        }
        const seedCount = Number(settings.seeds ?? 1);
        if (!Number.isInteger(seedCount) || seedCount < 1 || seedCount > 32) throw Error("Seeds must be 1–32.");
        const axes = [
            ["seed",Array.from({length:seedCount},(_,i)=>i ? randomSeed() : base.seed)],
            ["num_inference_steps",range(settings.stepsCount,settings.stepsStep,settings.stepsMid ?? base.num_inference_steps,1,200,true)],
            ["guidance_scale",range(settings.cfgCount,settings.cfgStep,settings.cfgMid ?? base.guidance_scale,0,30)],
        ];
        if (base.init_image) axes.push(["prompt_strength",range(settings.strengthCount,settings.strengthStep,settings.strengthMid ?? base.prompt_strength ?? .6,0,1)]);
        for (const [name,key] of [["models","use_stable_diffusion_model"],["vaes","use_vae_model"],["samplers","sampler_name"],["loras","use_lora_model"],["faces","use_face_correction"]]) {
            if (settings[name]?.length) axes.push([key,settings[name]]);
        }
        if (settings.loras?.length || base.use_lora_model) {
            const weights = Array.isArray(base.lora_alpha) ? base.lora_alpha : [base.lora_alpha ?? .5];
            if (settings.loras?.length) axes.push(["lora_alpha",range(settings.loraCount,settings.loraStep,weights[0],-2,2)]);
            else weights.forEach((weight,index) => axes.push([`lora:${index}`,range(settings.loraCount,settings.loraStep,weight,-2,2)]));
        }
        if (settings.modifiers?.length) axes.push(["prompt",settings.modifiers.map(value=>`${base.prompt}, ${value}`)]);
        const jobs = [];
        function visit(index, request) {
            if (jobs.length >= max) return;
            if (index === axes.length) { jobs.push(structuredClone({...request,num_outputs:1})); return; }
            const [key,values] = axes[index];
            for (const value of values) {
                const next = {...request};
                if (key.startsWith("lora:")) {
                    const weights = Array.isArray(request.lora_alpha) ? [...request.lora_alpha] : [request.lora_alpha ?? .5];
                    weights[Number(key.split(":")[1])] = value;
                    next.lora_alpha = Array.isArray(base.lora_alpha) ? weights : weights[0];
                } else next[key] = value;
                visit(index+1,next);
                if (jobs.length >= max) break;
            }
        }
        visit(0,base); return jobs;
    }
    if (typeof module !== "undefined" && module.exports) { module.exports = {frameTimes,rabbitRequests}; return; }

    (async () => {
        const policy = await window.CppKiosk.ready;
        if (!window.CppGeneration) await new Promise(resolve => window.addEventListener("cpp-generation-ready",resolve,{once:true}));
        const generation = window.CppGeneration, prefs = window.LocalPluginPreferences;
        const byId = id => document.getElementById(id), enabled = id => prefs.isEnabled(id);
        const root = document.querySelector(".generation-controls");
        const style = document.createElement("style"); style.textContent = `
            .cpp-creative-panel {margin-top:1rem;min-width:0}
            .cpp-creative-panel summary {font-weight:600;cursor:pointer;padding:.5rem 0}
            .cpp-creative-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:.65rem}
            .cpp-creative-panel label {display:flex;flex-direction:column;gap:.25rem;min-width:0}
            .cpp-creative-panel input,.cpp-creative-panel select,.cpp-creative-panel textarea {width:100%;box-sizing:border-box;min-width:0}
            .cpp-creative-panel textarea {min-height:4em}
            .cpp-creative-panel p {overflow-wrap:anywhere}
            #preview-content.cpp-rabbit-gallery {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:.8rem}
            .cpp-generated-image {min-width:0;overflow-wrap:anywhere}
            .cpp-generated-image[data-rabbit-selected] {outline:3px solid var(--ui-accent,#7ac)}
            .cpp-rabbit-actions[hidden],.cpp-creative-panel[hidden] {display:none!important}
            .cpp-creative-output img,.cpp-creative-output video {max-width:100%}
        `; document.head.append(style);
        function panel(id,title) {
            const element = document.createElement("details"); element.id=id; element.className="panel-box cpp-creative-panel";
            const summary=document.createElement("summary"); summary.textContent=title; element.append(summary); root.append(element); return element;
        }
        function text(parent,content) {const p=document.createElement("p");p.textContent=content;parent.append(p);return p;}
        function input(parent,id,label,value,type="number",attrs={}) {
            const row=document.createElement("label");row.htmlFor=id;row.textContent=label;
            const field=document.createElement(type === "textarea" ? "textarea" : "input");field.id=id;
            if(type!=="textarea")field.type=type;
            Object.assign(field,attrs);field.value=String(value);row.append(field);parent.append(row);return field;
        }
        function button(parent,id,label,action) {
            const element=document.createElement("button");element.id=id;element.type="button";element.textContent=label;
            element.addEventListener("click",()=>Promise.resolve().then(action).catch(error=>{message.textContent=error.message;}));
            parent.append(element);return element;
        }
        const rabbit=panel("cpp-rabbit-hole","Rabbit Hole — explore variations");
        text(rabbit,"Explore combinations of seeds, steps, guidance, models and modifiers. Start from the current prompt or branch from an image. Counts include the midpoint; the maximum caps the Cartesian product.");
        const grid=document.createElement("div");grid.className="cpp-creative-grid";rabbit.append(grid);
        let saved={};try{saved=JSON.parse(localStorage.getItem("cpp-rabbit-settings-v1")||"{}");}catch(_){}
        const fields={};
        for(const [key,label,value,min,max,step] of [
            ["max","Maximum images",16,1,512,1],["seeds","Seeds",2,1,32,1],
            ["stepsCount","Step count",1,1,32,1],["stepsStep","Step increment",5,0,200,1],["stepsMid","Step midpoint",25,1,200,1],
            ["cfgCount","Guidance count",1,1,32,1],["cfgStep","Guidance increment",1,0,30,.1],["cfgMid","Guidance midpoint",7.5,0,30,.1],
            ["strengthCount","Img2img strength count",1,1,32,1],["strengthStep","Strength increment",.1,0,1,.01],["strengthMid","Strength midpoint",.6,0,1,.01],
            ["loraCount","LoRA weight count",1,1,32,1],["loraStep","LoRA weight increment",.1,0,4,.01],
        ]) fields[key]=input(grid,`cpp-rabbit-${key}`,label,saved[key]??value,"number",{min,max,step});
        const lists={};
        for(const [key,label] of [["models","Checkpoints"],["vaes","VAEs"],["loras","LoRAs"],["samplers","Samplers"],["faces","Face correction models"]]) {
            const row=document.createElement("label");row.textContent=`${label} (none selected = current)`;
            const select=document.createElement("select");select.id=`cpp-rabbit-${key}`;select.multiple=true;select.size=3;row.htmlFor=select.id;
            row.append(select);rabbit.append(row);lists[key]=select;
            select.addEventListener("change",save);
        }
        fields.modifiers=input(rabbit,"cpp-rabbit-modifiers","Custom modifier alternatives (one per line)",saved.modifiers?.join("\n")||"","textarea");
        fields.stepButtons=input(rabbit,"cpp-rabbit-step-buttons","Image step buttons (comma-separated increments)",saved.stepButtons||"-20,-10,10,20,40","text");
        fields.cfgButtons=input(rabbit,"cpp-rabbit-cfg-buttons","Image guidance buttons (comma-separated increments)",saved.cfgButtons||"-2,-1,1,2,3","text");
        fields.chain=input(rabbit,"cpp-rabbit-chain","Img2img chain length",saved.chain||5,"number",{min:1,max:32});
        for(const field of Object.values(fields))field.addEventListener("change",save);
        function settings() {
            const values=Object.fromEntries(Object.entries(fields).map(([key,field])=>[key,field.type==="number"?Number(field.value):field.value]));
            values.modifiers=fields.modifiers.value.split("\n").map(value=>value.trim()).filter(Boolean);
            for(const [key,list] of Object.entries(lists))values[key]=Array.from(list.selectedOptions,option=>option.value);
            return values;
        }
        function save(){localStorage.setItem("cpp-rabbit-settings-v1",JSON.stringify(settings()));}
        const message=text(rabbit,"");message.id="cpp-creative-status";message.setAttribute("role","status");
        async function explore(base) {
            if(!enabled("rabbit-hole"))throw Error("Rabbit Hole is disabled.");
            const jobs=rabbitRequests(base,settings());
            message.textContent=`Queued ${jobs.length} Rabbit Hole images.`;
            const results=await Promise.allSettled(jobs.map(job=>generation.enqueue(job)));
            const failures=results.filter(result=>result.status==="rejected");
            message.textContent=failures.length ? `${failures.length} tasks failed: ${failures[0].reason.message}` : `Completed ${jobs.length} Rabbit Hole images.`;
        }
        button(rabbit,"cpp-rabbit-start","Start Rabbit Hole",()=>explore(generation.buildRequest()));
        button(rabbit,"cpp-rabbit-calculate","Calculate capped image count",()=>{message.textContent=`${rabbitRequests(generation.buildRequest(),settings()).length} images will be queued.`;});
        button(rabbit,"cpp-rabbit-layout","Toggle gallery / classic layout",()=>byId("preview-content").classList.toggle("cpp-rabbit-gallery"));
        button(rabbit,"cpp-rabbit-select-all","Select all images",()=>document.querySelectorAll(".cpp-generated-image").forEach(card=>card.setAttribute("data-rabbit-selected","")));
        button(rabbit,"cpp-rabbit-remove-selected","Remove selected previews",()=>document.querySelectorAll("[data-rabbit-selected]").forEach(card=>card.remove()));
        button(rabbit,"cpp-rabbit-branch-selected","Branch selected images",async()=>{
            const cards=Array.from(document.querySelectorAll("[data-rabbit-selected]"));
            if(!cards.length)throw Error("Select an image first.");
            if(cards.length*Number(fields.max.value)>512)throw Error("Selected branches exceed the 512-task limit.");
            await Promise.all(cards.map(card=>explore(card.cppRequest)));
        });
        async function loadChoices() {
            const endpoints={models:"/get/model",vaes:"/get/vae",loras:"/get/lora",faces:"/get/models"};
            for(const [key,url] of Object.entries(endpoints)) {
                if(policy.enabled && ["loras","faces"].includes(key)){lists[key].closest("label").hidden=true;continue;}
                const response=await fetch(url,{cache:"no-store"});if(!response.ok)throw Error(`Cannot load ${key}: HTTP ${response.status}`);
                const data=await response.json();
                for(const model of data.models||[]) {
                    if(key==="faces" && !model.tags?.some(tag=>["gfpgan","codeformer"].includes(tag)))continue;
                    const value=String(model.model||model.name||"");if(!value)continue;
                    const option=new Option(value,value);option.selected=saved[key]?.includes(value)||false;lists[key].add(option);
                }
            }
            for(const source of byId("sampler")?.options||[]) {
                const option=new Option(source.textContent,source.value);option.selected=saved.samplers?.includes(source.value)||false;lists.samplers.add(option);
            }
        }
        loadChoices().catch(error=>{message.textContent=error.message;});
        let chainEpoch=0;
        window.addEventListener("cpp-generation-cancel",()=>{chainEpoch++;});
        function numericButtons(value){return value.split(",").map(Number).filter(value=>Number.isFinite(value)&&value!==0).slice(0,10);}
        async function chain(request,image) {
            const epoch=chainEpoch, count=Number(fields.chain.value);
            if(!Number.isInteger(count)||count<1||count>32)throw Error("Chain length must be 1–32.");
            let source=image.src;
            for(let i=0;i<count;i++) {
                if(epoch!==chainEpoch || !enabled("rabbit-hole"))break;
                const result=await generation.enqueue({...request,init_image:source,prompt_strength:Number(fields.strengthMid.value),num_outputs:1,seed:randomSeed()});
                const item=result?.output?.[0];if(!item?.data)throw Error("Chain stopped: render returned no image.");
                source=item.data.startsWith("data:")?item.data:`data:image/${request.output_format||"jpeg"};base64,${item.data}`;
            }
        }
        window.addEventListener("cpp-generation-result",({detail:{card,request,image}})=>{
            const actions=document.createElement("div");actions.className="cpp-image-actions cpp-rabbit-actions";card.append(actions);actions.hidden=!enabled("rabbit-hole");
            button(actions,"","Select",()=>card.toggleAttribute("data-rabbit-selected"));
            button(actions,"","Rabbit Hole",()=>explore(request));
            button(actions,"","Img2img chain",()=>chain(request,image));
            button(actions,"","Focus image",()=>{
                const dialog=document.createElement("dialog"),copy=image.cloneNode();copy.style.cssText="max-width:85vw;max-height:80vh;object-fit:contain";
                const close=document.createElement("button");close.textContent="Close";close.onclick=()=>dialog.close();dialog.append(close,copy);document.body.append(dialog);
                dialog.addEventListener("close",()=>dialog.remove(),{once:true});dialog.showModal();
            });
            for(const [name,key,minimum,maximum,values] of [
                ["Steps","num_inference_steps",1,200,numericButtons(fields.stepButtons.value)],
                ["Guidance","guidance_scale",0,30,numericButtons(fields.cfgButtons.value)],
            ]) for(const delta of values) button(actions,"",`${name} ${delta>0?"+":""}${delta}`,()=>generation.enqueue({...request,num_outputs:1,[key]:Math.min(maximum,Math.max(minimum,Number(request[key])+delta))}));
        });

        const animate=panel("cpp-animate","Animate — video, GIF or image sequence");
        text(animate,"Choose a video, animated GIF, or ordered images. Each frame is rendered with the current prompt and model. Frames run sequentially to keep GPU load bounded. Maximum 120 frames / 64 million output pixels per animation.");
        const files=input(animate,"cpp-animate-files","Source video / GIF / images","","file",{accept:"video/*,image/gif,image/png,image/jpeg,image/webp",multiple:true});
        const animationGrid=document.createElement("div");animationGrid.className="cpp-creative-grid";animate.append(animationGrid);
        const fps=input(animationGrid,"cpp-animate-fps","Frames per second",5,"number",{min:.1,max:30,step:.1});
        const from=input(animationGrid,"cpp-animate-from","Start time (video seconds)",0,"number",{min:0,step:.1});
        const to=input(animationGrid,"cpp-animate-to","End time (0 = video end)",0,"number",{min:0,step:.1});
        const strength=input(animationGrid,"cpp-animate-strength","Prompt strength",.6,"number",{min:0,max:1,step:.01});
        const format=document.createElement("select");format.id="cpp-animate-format";format.setAttribute("aria-label","Animation output format");
        for(const [value,label] of [["gif","Animated GIF"],["video","Video (WebM)"],["render","Render frames only"]])format.add(new Option(label,value));animate.append(format);
        for(const control of [fps,from,to,strength,format]) {
            try{const value=localStorage.getItem(control.id);if(value!==null)control.value=value;}catch(_){}
            control.addEventListener("change",()=>localStorage.setItem(control.id,control.value));
        }
        const animationStatus=text(animate,"");animationStatus.id="cpp-animate-status";animationStatus.setAttribute("role","status");
        const output=document.createElement("div");output.className="cpp-creative-output";animate.append(output);
        let animationEpoch=0,animating=false,downloadUrl=null,libraryPromise=null;
        window.addEventListener("cpp-generation-cancel",()=>{animationEpoch++;});
        function libraries() {
            if(!libraryPromise)libraryPromise=new Promise((resolve,reject)=>{
                const script=document.createElement("script");script.src="/cpp-ui/scripts/animation_libraries.js";
                script.onload=resolve;script.onerror=()=>{libraryPromise=null;reject(Error("Could not load local animation libraries."));};document.head.append(script);
            });return libraryPromise;
        }
        function eventOnce(element,name,action) {
            return new Promise((resolve,reject)=>{
                const timer=setTimeout(()=>done(Error(`Timed out waiting for ${name}.`)),15000);
                const ok=()=>done(),bad=()=>done(Error("Cannot decode this media file."));
                function done(error){clearTimeout(timer);element.removeEventListener(name,ok);element.removeEventListener("error",bad);error?reject(error):resolve();}
                element.addEventListener(name,ok,{once:true});element.addEventListener("error",bad,{once:true});action();
            });
        }
        async function decodeImage(url){const image=new Image();await eventOnce(image,"load",()=>{image.src=url;});return image;}
        function checkBudget(count,base){if(count>120||count*base.width*base.height>64000000)throw Error("Animation frame/pixel limit exceeded; reduce frame count or image size.");}
        async function exportFrames(frames,rate,kind,base,check) {
            const canvas=document.createElement("canvas");canvas.width=base.width;canvas.height=base.height;const ctx=canvas.getContext("2d");
            if(kind==="gif") {
                await libraries();check();
                const gif=new GIFa({workers:1,quality:10,width:canvas.width,height:canvas.height,workerScript:"/cpp-ui/scripts/animation_gif_worker.js"});
                for(const source of frames){check();ctx.drawImage(await decodeImage(source),0,0,canvas.width,canvas.height);gif.addFrame(ctx,{copy:true,delay:1000/rate});}
                return await new Promise((resolve,reject)=>{
                    const cancel=()=>gif.abort();
                    const cleanup=()=>{
                        clearTimeout(timer);window.removeEventListener("cpp-generation-cancel",cancel);
                        for(const worker of [...gif.activeWorkers,...gif.freeWorkers])worker.terminate();
                    };
                    const timer=setTimeout(()=>{cleanup();reject(Error("GIF encoding timed out."));},120000);
                    window.addEventListener("cpp-generation-cancel",cancel,{once:true});
                    gif.on("finished",blob=>{cleanup();resolve(blob);});
                    gif.on("abort",()=>{cleanup();reject(Error("GIF encoding cancelled."));});
                    try{gif.render();}catch(error){cleanup();reject(error);}
                });
            }
            if(!window.MediaRecorder || !canvas.captureStream)throw Error("This browser cannot export WebM; choose GIF.");
            const mime=["video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm"].find(type=>MediaRecorder.isTypeSupported(type));
            if(!mime)throw Error("WebM encoding is not supported; choose GIF.");
            const stream=canvas.captureStream(rate),recorder=new MediaRecorder(stream,{mimeType:mime}),chunks=[];
            const completed=new Promise((resolve,reject)=>{recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data);};recorder.onstop=()=>resolve(new Blob(chunks,{type:mime}));recorder.onerror=()=>reject(Error("Video encoding failed."));});
            completed.catch(()=>{});
            try{
                ctx.drawImage(await decodeImage(frames[0]),0,0,canvas.width,canvas.height);recorder.start();
                for(const source of frames){check();ctx.drawImage(await decodeImage(source),0,0,canvas.width,canvas.height);await new Promise(resolve=>setTimeout(resolve,1000/rate));}
                recorder.stop();return await completed;
            }finally{if(recorder.state!=="inactive")recorder.stop();stream.getTracks().forEach(track=>track.stop());}
        }
        async function runAnimation() {
            if(animating)return;
            if(!enabled("animate"))throw Error("Animate is disabled.");
            const selected=Array.from(files.files);if(!selected.length)throw Error("Choose a source video, GIF or images first.");
            const base=generation.buildRequest(),rate=Number(fps.value),amount=Number(strength.value),kind=format.value;
            if(!Number.isFinite(amount)||amount<0||amount>1)throw Error("Prompt strength must be 0–1.");
            frameTimes(0,1,rate);
            const epoch=++animationEpoch,frames=[],urls=[];let video=null;
            const check=()=>{if(epoch!==animationEpoch || !enabled("animate"))throw Error("Animation cancelled.");};
            const canvas=document.createElement("canvas");canvas.width=base.width;canvas.height=base.height;const ctx=canvas.getContext("2d");
            async function renderFrame(source,index,total) {
                check();ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(source,0,0,canvas.width,canvas.height);
                animationStatus.textContent=`Rendering frame ${index+1} / ${total}…`;
                const result=await generation.enqueue({...base,init_image:canvas.toDataURL("image/png"),prompt_strength:amount,num_outputs:1});
                check();const item=result?.output?.[0];if(!item?.data)throw Error("Render returned no animation frame.");
                if(kind!=="render")frames.push(item.data.startsWith("data:")?item.data:`data:image/${base.output_format};base64,${item.data}`);
            }
            animating=true;animateStart.disabled=true;animateStop.disabled=false;
            animationStatus.textContent="Preparing animation…";
            try {
                if(selected.length===1 && selected[0].type.startsWith("video/")) {
                    video=document.createElement("video");video.muted=true;video.preload="auto";const url=URL.createObjectURL(selected[0]);urls.push(url);
                    await eventOnce(video,"loadeddata",()=>{video.src=url;video.load();});check();
                    const end=Number(to.value)||video.duration;
                    if(end>video.duration+.01)throw Error("End time exceeds the video duration.");
                    const times=frameTimes(Number(from.value),end,rate);checkBudget(times.length,base);
                    for(const [index,time] of times.entries()) {check();if(Math.abs(video.currentTime-time)>.001)await eventOnce(video,"seeked",()=>{video.currentTime=time;});await renderFrame(video,index,times.length);}
                } else if(selected.length===1 && (selected[0].type==="image/gif" || selected[0].name.toLowerCase().endsWith(".gif"))) {
                    await libraries();check();const host=document.createElement("div"),image=new Image();host.append(image);
                    const gif=new SuperGif({gif:image,auto_play:false,show_progress_bar:false});
                    const bytes=new Uint8Array(await selected[0].arrayBuffer());
                    await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error("Cannot decode this GIF.")),15000);gif.load_raw(bytes,()=>{clearTimeout(timer);resolve();});});
                    const count=gif.get_length();if(!count)throw Error("GIF contains no frames.");checkBudget(count,base);
                    for(let index=0;index<count;index++){check();gif.move_to(index);await renderFrame(gif.get_canvas(),index,count);}gif.pause();
                } else {
                    if(selected.some(file=>!file.type.startsWith("image/")))throw Error("Choose one video or an image-only sequence.");
                    checkBudget(selected.length,base);
                    for(const [index,file] of selected.entries()){check();const url=URL.createObjectURL(file);urls.push(url);await renderFrame(await decodeImage(url),index,selected.length);}
                }
                check();
                if(kind!=="render") {
                    animationStatus.textContent="Encoding animation…";
                    const blob=await exportFrames(frames,rate,kind,base,check);check();
                    if(downloadUrl)URL.revokeObjectURL(downloadUrl);downloadUrl=URL.createObjectURL(blob);output.replaceChildren();
                    const media=document.createElement(kind==="gif"?"img":"video");media.src=downloadUrl;media.alt="Rendered animation";if(kind!=="gif")media.controls=true;
                    const link=document.createElement("a");link.href=downloadUrl;link.download=`animation.${kind==="gif"?"gif":"webm"}`;link.textContent="Download animation";output.append(media,link);
                }
                animationStatus.textContent="Animation complete.";
            } catch(error){animationStatus.textContent=error.message;}
            finally{animating=false;animateStart.disabled=false;animateStop.disabled=true;urls.forEach(url=>URL.revokeObjectURL(url));if(video){video.removeAttribute("src");video.load();}}
        }
        const animateStart=button(animate,"cpp-animate-start","Animate",()=>runAnimation().catch(error=>{animationStatus.textContent=error.message;}));
        const animateStop=button(animate,"cpp-animate-stop","Stop animation and queue",()=>generation.cancel());animateStop.disabled=true;
        function refresh() {
            rabbit.hidden=!enabled("rabbit-hole");animate.hidden=!enabled("animate");
            document.querySelectorAll(".cpp-rabbit-actions").forEach(element=>{element.hidden=rabbit.hidden;});
            if(rabbit.hidden){chainEpoch++;byId("preview-content").classList.remove("cpp-rabbit-gallery");}
            if(animate.hidden)animationEpoch++;
        }
        window.addEventListener("local-plugin-preferences-changed",refresh);refresh();
        window.CppCreativePlugins={rabbitRequests,frameTimes};
    })().catch(error=>console.error("Modern creative plugins:",error));
})();
