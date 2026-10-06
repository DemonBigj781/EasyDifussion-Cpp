(() => {
    "use strict";
    function shuttle(request,mode,value){
        if(!request.use_lora_model)throw Error("This image has no LoRA.");
        const weights=Array.isArray(request.lora_alpha)?request.lora_alpha:[request.lora_alpha];
        if(!weights.length||weights.some(weight=>!Number.isFinite(Number(weight))))throw Error("This image has no valid LoRA weights.");
        const next=weights.map(weight=>Number(Math.max(0,Math.min(1,mode==="set"?value:Number(weight)+value)).toFixed(2)));
        return {...request,num_outputs:1,lora_alpha:Array.isArray(request.lora_alpha)?next:next[0]};
    }
    function schedule(base,index,every,delta,reset,min,max){
        if(![base,index,every,delta,reset].every(Number.isFinite)||!Number.isInteger(every)||every<0||!Number.isInteger(reset)||reset<0)
            throw Error("Adjustment intervals must be non-negative integers and values must be finite.");
        const step=reset?index%reset:index;
        return Number(Math.max(min,Math.min(max,base+(every?Math.floor(step/every)*delta:0))).toFixed(6));
    }
    if(typeof module!=="undefined"&&module.exports){module.exports={shuttle,schedule};return;}
    (async()=>{
        const ui=await window.CppPluginUI.ready,gen=window.CppGeneration;
        const panel=ui.panel("stig-image-to-img2img","Stig image-to-img2img");
        ui.note(panel,"Start with an uploaded image, a generated image, or a text-to-image first frame. Each completed output can feed the next request. Guidance, strength, LoRA and seed adjustments use independent intervals and resets.");
        const file=ui.field(panel,"cpp-img2img-file","Source image","","file",{accept:"image/*"});
        const preview=document.createElement("img");preview.id="cpp-img2img-source-preview";preview.alt="Img2img source";preview.hidden=true;panel.append(preview);
        const useSource=ui.field(panel,"cpp-img2img-use-source","Use source for Generate and feed results into the sequence",true,"checkbox");
        const amount=ui.field(panel,"cpp-img2img-count","Number of generations",4,"number",{min:1,max:512});
        const strength=ui.field(panel,"cpp-img2img-strength","Initial prompt strength",.6,"number",{min:0,max:.99,step:.01});
        const groups={};
        for(const [key,title,every,delta] of [["guidance","Guidance",2,.2],["strength","Prompt strength",1,.06],["lora","LoRA weight",1,.05],["seed","Seed",2,1]]){
            const group=document.createElement("fieldset");const legend=document.createElement("legend");legend.textContent=title;group.append(legend);group.className="cpp-workbench-grid";panel.append(group);
            groups[key]={every:ui.field(group,`cpp-img2img-${key}-every`,"Adjust every N frames (0 = off)",every,"number",{min:0,max:512}),
                delta:ui.field(group,`cpp-img2img-${key}-delta`,"Adjustment",delta,"number",{step:"any"}),
                reset:ui.field(group,`cpp-img2img-${key}-reset`,"Reset every N frames (0 = off)",0,"number",{min:0,max:512})};
        }
        const geometry=document.createElement("details");geometry.className="cpp-workbench";const heading=document.createElement("summary");heading.textContent="Zoom, rotate and pan";geometry.append(heading);panel.append(geometry);
        const transform={};
        for(const [key,label,value,min,max,step] of [
            ["zoomEvery","Zoom every N frames (0 = off)",1,0,512,1],["zoomX","Zoom width factor",1.05,.001,10,.001],["zoomY","Zoom height factor",1.05,.001,10,.001],
            ["zoomXEvery","Adjust zoom width factor every N frames",0,0,512,1],["zoomXDelta","Zoom width factor increment",0,-10,10,.01],
            ["zoomYEvery","Adjust zoom height factor every N frames",0,0,512,1],["zoomYDelta","Zoom height factor increment",0,-10,10,.01],
            ["zoomReset","Reset zoom factors every N frames",0,0,512,1],
            ["rotateEvery","Rotate every N frames (0 = off)",0,0,512,1],["angle","Rotation angle (degrees)",0,-360,360,1],
            ["panEvery","Pan every N frames (0 = off)",0,0,512,1],["panX","Pan X (pixels)",0,-2048,2048,1],["panY","Pan Y (pixels)",0,-2048,2048,1],
        ])transform[key]=ui.field(geometry,`cpp-img2img-${key}`,label,value,"number",{min,max,step});
        const promptMode=ui.select(panel,"cpp-img2img-prompt-mode","Prompt source",[["current","Current prompt"],["sequential","Stig wildcard — sequential"],["random","Stig wildcard — random"]],"current");
        const promptEvery=ui.field(panel,"cpp-img2img-prompt-every","Change wildcard prompt every N frames",2,"number",{min:1,max:512});
        const delay=ui.field(panel,"cpp-img2img-delay","Delay between tasks (milliseconds)",100,"number",{min:0,max:60000});
        const progress=ui.note(panel,"");progress.id="cpp-img2img-status";progress.setAttribute("role","status");
        let source="",running=false,epoch=0;
        async function setSource(value){await ui.image(value);source=value;preview.src=value;preview.hidden=false;useSource.checked=true;}
        file.addEventListener("change",async()=>{
            try{const input=file.files?.[0];if(!input)return;if(input.size>32000000)throw Error("Source images must be under 32 MB.");
                const reader=new FileReader();reader.onload=()=>setSource(String(reader.result)).catch(error=>{progress.textContent=error.message;});reader.onerror=()=>{progress.textContent="Could not read the image.";};reader.readAsDataURL(input);
            }catch(error){progress.textContent=error.message;}
        });
        ui.button(panel,"cpp-img2img-clear","Clear source",()=>{source="";preview.removeAttribute("src");preview.hidden=true;file.value="";});
        ui.button(panel,"cpp-img2img-zero","Zero all adjustments",()=>{
            for(const group of Object.values(groups)){group.every.value="0";group.delta.value="0";group.reset.value="0";}
            for(const key of ["zoomEvery","zoomXEvery","zoomYEvery","rotateEvery","panEvery"])transform[key].value="0";
        });
        function readNumber(field){const value=Number(field.value);if(!Number.isFinite(value)||!field.checkValidity())throw Error(`Invalid value: ${field.labels?.[0]?.textContent||field.id}`);return value;}
        async function transformed(url,index,options,width,height){
            const zoom=index>0&&options.zoomEvery>0&&index%options.zoomEvery===0;
            const rotate=index>0&&options.rotateEvery>0&&index%options.rotateEvery===0;
            const pan=index>0&&options.panEvery>0&&index%options.panEvery===0;
            if(!zoom&&!rotate&&!pan)return url;
            const img=await ui.image(url),canvas=document.createElement("canvas");canvas.width=width;canvas.height=height;
            const ctx=canvas.getContext("2d");ctx.drawImage(img,0,0,width,height);
            ctx.translate(width/2+(pan?options.panX:0),height/2+(pan?options.panY:0));
            if(rotate)ctx.rotate(options.angle*Math.PI/180);
            if(zoom)ctx.scale(schedule(options.zoomX,index,options.zoomXEvery,options.zoomXDelta,options.zoomReset,.001,10),schedule(options.zoomY,index,options.zoomYEvery,options.zoomYDelta,options.zoomReset,.001,10));
            ctx.drawImage(img,-width/2,-height/2,width,height);return canvas.toDataURL("image/png");
        }
        async function run(){
            if(running)return;
            const total=readNumber(amount);if(!Number.isInteger(total))throw Error("Generation count must be an integer.");
            const base=gen.buildRequest(),initialStrength=readNumber(strength),pause=readNumber(delay);
            const settings=Object.fromEntries(Object.entries(groups).map(([key,fields])=>[key,Object.fromEntries(Object.entries(fields).map(([name,field])=>[name,readNumber(field)]))]));
            const options=Object.fromEntries(Object.entries(transform).map(([key,field])=>[key,readNumber(field)]));
            for(const [key,value] of Object.entries(options))if((key.endsWith("Every")||key.endsWith("Reset"))&&!Number.isInteger(value))throw Error("Transform intervals must be integers.");
            const mode=promptMode.value,changeEvery=readNumber(promptEvery),feedback=useSource.checked;
            if(!Number.isInteger(changeEvery))throw Error("Prompt interval must be an integer.");
            if(mode!=="current")window.CppWildcard.getPrompt(0,mode==="random");
            for(const setting of Object.values(settings))schedule(0,0,setting.every,setting.delta,setting.reset,0,1);
            let image=source,prompt=base.prompt;const runEpoch=++epoch;running=true;start.disabled=true;
            try{
                for(let index=0;index<total;index++){
                    if(runEpoch!==epoch||!ui.enabled("stig-image-to-img2img"))throw Error("Img2img sequence cancelled.");
                    const request={...base,num_outputs:1};
                    for(const [name,key,value,min,max] of [["guidance","guidance_scale",base.guidance_scale,1.1,50],["strength","prompt_strength",initialStrength,0,.99],["seed","seed",base.seed,0,4294967295]]){
                        const setting=settings[name];request[key]=schedule(Number(value),index,setting.every,setting.delta,setting.reset,min,max);
                    }
                    if(base.use_lora_model){const setting=settings.lora;const weights=Array.isArray(base.lora_alpha)?base.lora_alpha:[base.lora_alpha??.5];
                        const next=weights.map(weight=>schedule(Number(weight),index,setting.every,setting.delta,setting.reset,-50,50));request.lora_alpha=Array.isArray(base.lora_alpha)?next:next[0];}
                    if(mode!=="current"&&index%changeEvery===0)prompt=window.CppWildcard.getPrompt(Math.floor(index/changeEvery),mode==="random");
                    request.prompt=prompt;request.original_prompt=prompt;
                    if(feedback&&image)request.init_image=await transformed(image,index,options,base.width,base.height);else delete request.init_image;
                    if(runEpoch!==epoch||!ui.enabled("stig-image-to-img2img"))throw Error("Img2img sequence cancelled.");
                    progress.textContent=`Rendering ${index+1} / ${total}…`;
                    const result=await gen.enqueue(request),item=result?.output?.[0];
                    if(!item?.data)throw Error("Img2img sequence stopped: no output image.");
                    if(feedback)image=item.data.startsWith("data:")?item.data:`data:image/${request.output_format||"jpeg"};base64,${item.data}`;
                    if(index+1<total&&pause)await new Promise(resolve=>setTimeout(resolve,pause));
                }
                progress.textContent=`Completed ${total} generations.`;
            }catch(error){progress.textContent=error.message;}finally{running=false;start.disabled=false;}
        }
        const start=ui.button(panel,"cpp-img2img-start","Start generating images",run);
        ui.button(panel,"cpp-img2img-stop","Stop sequence and queue",()=>gen.cancel());
        window.addEventListener("cpp-generation-cancel",()=>{epoch++;});
        window.addEventListener("local-plugin-preferences-changed",()=>{if(!ui.enabled("stig-image-to-img2img"))epoch++;});
        window.CppImageTools={setSource,applyToRequest(request){
            if(ui.enabled("stig-image-to-img2img")&&useSource.checked&&source){request.init_image=source;request.prompt_strength=readNumber(strength);}
        }};
        const utilityPanel=ui.panel("stig-image-utilities","Stig image utilities");
        ui.note(utilityPanel,"Tools appear on generated images: rotation, horizontal/vertical tiling, information, and an image download with a JSON metadata sidecar. Transformations edit previews, not server-saved originals.");
        const loraPanel=ui.panel("stig-lora-shuttle","Stig LoRA shuttle controls");
        ui.note(loraPanel,"On images generated with LoRAs: set all weights to 0, 0.5 or 1; shift by ±0.1; or queue the complete 0–1 grid. Each action retains the source request and seed.");
        const story=ui.panel("storyteller","Storyteller");
        ui.note(story,"Add generated images using their Add to Storyteller action. The board stays in this browser tab until it is cleared or the page is reloaded.");
        const view=document.createElement("section");view.id="cpp-story-view";view.className="panel-box cpp-workbench";view.hidden=true;
        const title=document.createElement("h2");title.textContent="Storyteller";view.append(title);document.getElementById("page-content").append(view);
        const items=document.createElement("div");items.id="cpp-story-items";view.append(items);
        const nav=document.createElement("div");nav.setAttribute("role","tablist");nav.setAttribute("aria-label","Generation workspace");
        document.querySelector(".generate-columns").before(ui.gate(nav,"storyteller"));
        function showStory(show){view.hidden=!show;document.querySelector(".generate-columns").hidden=show;generationTab.setAttribute("aria-selected",String(!show));storyTab.setAttribute("aria-selected",String(show));}
        const generationTab=ui.button(nav,"cpp-generation-tab","Generate",()=>showStory(false));generationTab.setAttribute("role","tab");
        const storyTab=ui.button(nav,"cpp-story-tab","Storyteller",()=>showStory(true));storyTab.setAttribute("role","tab");showStory(false);
        function saveItem(item){ui.download(item.querySelector("canvas").toDataURL("image/png"),`storyteller-${Array.from(items.children).indexOf(item)+1}.png`);}
        ui.button(view,"cpp-story-save-all","Save all",()=>items.querySelectorAll("figure").forEach(saveItem));
        ui.button(view,"cpp-story-remove-all","Remove all",()=>items.replaceChildren());
        window.addEventListener("local-plugin-preferences-changed",()=>{if(!ui.enabled("storyteller"))showStory(false);});
        async function addStory(image){await image.decode();const figure=document.createElement("figure"),canvas=document.createElement("canvas");canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;canvas.getContext("2d").drawImage(image,0,0);figure.append(canvas);items.append(figure);
            ui.button(figure,"","Save",()=>saveItem(figure));ui.button(figure,"","Remove",()=>figure.remove());ui.status.textContent=`Added image to Storyteller (${items.children.length}).`;}
        async function transformPreview(image,kind){
            await image.decode();const w=image.naturalWidth,h=image.naturalHeight,canvas=document.createElement("canvas");
            const rotation=kind==="left"||kind==="right";
            canvas.width=rotation?h:w*(kind==="x"||kind==="xy"?2:1);canvas.height=rotation?w:h*(kind==="y"||kind==="xy"?2:1);
            if(canvas.width*canvas.height>32000000)throw Error("Transformed preview exceeds 32 million pixels.");
            const ctx=canvas.getContext("2d");
            if(rotation){ctx.translate(canvas.width/2,canvas.height/2);ctx.rotate((kind==="left"?-1:1)*Math.PI/2);ctx.drawImage(image,-w/2,-h/2);}
            else for(let y=0;y<canvas.height;y+=h)for(let x=0;x<canvas.width;x+=w)ctx.drawImage(image,x,y);
            image.src=canvas.toDataURL("image/png");await image.decode();
        }
        window.addEventListener("cpp-generation-result",({detail:{card,image,request}})=>{
            const sourceActions=ui.actions(card,"stig-image-to-img2img");ui.button(sourceActions,"","Use as img2img source",()=>setSource(image.src));
            const utilities=ui.actions(card,"stig-image-utilities");
            for(const [kind,label] of [["left","Rotate left"],["right","Rotate right"],["x","Tile horizontally"],["y","Tile vertically"],["xy","Tile 2 × 2"]])ui.button(utilities,"",label,()=>transformPreview(image,kind));
            ui.button(utilities,"","Image information",()=>ui.showInfo("Image information",JSON.stringify(request,null,2)));
            ui.button(utilities,"","Download image + metadata",async()=>{await image.decode();const canvas=document.createElement("canvas");canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;canvas.getContext("2d").drawImage(image,0,0);
                const name=`image-${request.seed}`;ui.download(canvas.toDataURL("image/png"),`${name}.png`);ui.download(new Blob([JSON.stringify(request,null,2)],{type:"application/json"}),`${name}.json`);});
            const storyActions=ui.actions(card,"storyteller");ui.button(storyActions,"","Add to Storyteller",()=>addStory(image));
            if(request.use_lora_model){
                const controls=ui.actions(card,"stig-lora-shuttle");
                for(const [label,mode,value] of [["LoRA 0","set",0],["LoRA −0.1","shift",-.1],["LoRA 0.5","set",.5],["LoRA +0.1","shift",.1],["LoRA 1","set",1]])ui.button(controls,"",label,()=>gen.enqueue(shuttle(request,mode,value)));
                ui.button(controls,"","LoRA grid 0–1",()=>ui.batch(Array.from({length:11},(_,index)=>shuttle(request,"set",index/10))));
            }
        });
    })().catch(error=>console.error("Image workbench:",error));
})();
