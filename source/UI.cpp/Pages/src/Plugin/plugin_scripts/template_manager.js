(() => {
    "use strict";
    function validateTemplates(value){
        if(!Array.isArray(value))throw Error("Template backup must be an array.");
        if(value.length>10000)throw Error("A backup may contain at most 10,000 templates.");
        for(const template of value){
            if(!template||typeof template.name!=="string"||!template.name.trim()||!template.task?.reqBody||typeof template.task.reqBody!=="object"||Array.isArray(template.task.reqBody))throw Error("Invalid template record.");
            if(typeof template.task.reqBody.prompt!=="string")throw Error("Template prompt must be text.");
            if(template.task.reqBody.negative_prompt!==undefined&&typeof template.task.reqBody.negative_prompt!=="string")throw Error("Template negative prompt must be text.");
        }
        return value;
    }
    if(typeof module!=="undefined"&&module.exports){module.exports={validateTemplates};return;}
    (async()=>{
        const ui=await window.CppPluginUI.ready,gen=window.CppGeneration;
        const panel=ui.panel("template-manager","Template manager");
        ui.note(panel,"Uses the same template database as the legacy UI. Save current settings or load/edit a saved request below. Run template preserves all stored request options, including options not exposed by the modern form.");
        const name=ui.field(panel,"cpp-template-name","Template name","");
        const editor=ui.field(panel,"cpp-template-request","Template request (JSON)","","textarea");
        const seed=ui.field(panel,"cpp-template-restore-seeds","Restore template seeds",localStorage.getItem("restore_seeds")==="true","checkbox");
        const copies=ui.field(panel,"cpp-template-copies","Images per template",localStorage.getItem("slideshow_image_count")||1,"number",{min:1,max:512});
        const filter=ui.field(panel,"cpp-template-filter","Search templates","","search");
        const status=ui.note(panel,"");status.id="cpp-template-status";status.setAttribute("role","status");
        const list=document.createElement("div");list.id="cpp-template-list";panel.append(list);
        let templates=[],selectedName=null;
        function openDB(){return new Promise((resolve,reject)=>{
            const request=indexedDB.open("EasyDiffusionSettingsDatabase",1);
            request.onupgradeneeded=()=>{if(!request.result.objectStoreNames.contains("EasyDiffusionSettings"))request.result.createObjectStore("EasyDiffusionSettings",{keyPath:"id"});};
            request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);request.onblocked=()=>reject(Error("Close other template database dialogs and retry."));
        });}
        async function store(update){
            const db=await openDB();return new Promise((resolve,reject)=>{
                const tx=db.transaction("EasyDiffusionSettings",update?"readwrite":"readonly"),records=tx.objectStore("EasyDiffusionSettings");
                let result,failure;const request=records.get("task templates");
                request.onsuccess=()=>{
                    try{
                        const current=validateTemplates(request.result?.value===undefined?[]:JSON.parse(request.result.value));
                        result=update?validateTemplates(update(current)):current;
                        if(update)records.put({id:"task templates",value:JSON.stringify(result)});
                    }catch(error){failure=error;tx.abort();}
                };
                tx.oncomplete=()=>{db.close();resolve(result);};
                tx.onabort=()=>{db.close();reject(failure||tx.error||Error("Template transaction aborted."));};
                tx.onerror=()=>{failure=tx.error;};
            });
        }
        function visible(){const terms=filter.value.toLowerCase().split(/\s+/).filter(Boolean);return templates.filter(template=>terms.every(term=>template.name.toLowerCase().includes(term)));}
        function load(template){selectedName=template.name;name.value=template.name;editor.value=JSON.stringify(template.task.reqBody,null,2);status.textContent=`Loaded ${template.name}.`;}
        function readEditor(){const record={name:name.value.trim(),task:{reqBody:JSON.parse(editor.value)}};validateTemplates([record]);return record;}
        function render(){
            list.replaceChildren();
            for(const template of visible()){
                const row=document.createElement("div");row.className="cpp-template-entry";const title=document.createElement("strong");title.textContent=template.name;row.append(title);list.append(row);
                ui.button(row,"","Load / edit",()=>load(template));
                ui.button(row,"","Run template",()=>run([template]));
                ui.button(row,"","Delete",async()=>{if(!confirm(`Delete template “${template.name}”?`))return;await mutate(current=>current.filter(item=>item.name!==template.name));});
            }
        }
        async function reload(){templates=await store();render();}
        async function mutate(update){templates=await store(update);render();localStorage.setItem("cpp-template-manager-change",`${Date.now()}-${Math.random()}`);}
        async function run(chosen){
            const amount=Number(copies.value);if(!Number.isInteger(amount)||amount<1||amount>512||chosen.length*amount>512)throw Error("Template runs must total 1–512 images.");
            if(!chosen.length)throw Error("No matching templates.");
            const requests=chosen.flatMap(template=>Array.from({length:amount},(_,index)=>{
                const request=structuredClone(template.task.reqBody);request.num_outputs=1;request.stream_image_progress=true;
                request.seed=seed.checked&&Number.isFinite(Number(request.seed))&&Number(request.seed)>=0?(Number(request.seed)+index)%4294967296:Math.floor(Math.random()*2147483647);
                delete request.taskSeed;return request;
            }));
            status.textContent=`Queued ${requests.length} template images.`;await ui.batch(requests);status.textContent=`Completed ${requests.length} template images.`;
        }
        ui.button(panel,"cpp-template-capture","Capture current settings",()=>{editor.value=JSON.stringify(gen.buildRequest(),null,2);selectedName=null;if(!name.value.trim())name.value=document.getElementById("prompt").value.trim().slice(0,60)||"New template";});
        ui.button(panel,"cpp-template-save","Save template",async()=>{
            const record=readEditor(),old=selectedName;
            if(templates.some(template=>template.name===record.name)&&!confirm(`Replace template “${record.name}”?`))return;
            await mutate(current=>{
                const existing=current.find(template=>template.name===(old||record.name));
                const replacement={...existing,...record,task:{...existing?.task,...record.task}};
                return [...current.filter(template=>template.name!==record.name&&(!old||template.name!==old)),replacement];
            });selectedName=record.name;status.textContent=`Saved ${record.name}.`;
        });
        ui.button(panel,"cpp-template-run-editor","Run edited template",()=>run([readEditor()]));
        ui.button(panel,"cpp-template-run-all","Run all matching templates",()=>run(visible()));
        ui.button(panel,"cpp-template-use-prompt","Use template prompts in generator",()=>{
            const request=readEditor().task.reqBody;ui.change(document.getElementById("prompt"),request.original_prompt??request.prompt);ui.change(document.getElementById("negative_prompt"),request.negative_prompt||"");
            status.textContent="Copied prompts. Use Run template to reproduce its complete stored settings.";
        });
        ui.button(panel,"cpp-template-export","Export templates",async()=>ui.download(new Blob([JSON.stringify(await store(),null,2)],{type:"application/json"}),"Templates backup.json"));
        const file=ui.field(panel,"cpp-template-import","Import a legacy or modern template backup","","file",{accept:".json,application/json"});
        ui.note(panel,"Import adds new names and preserves existing same-name templates. To replace one, load it in the editor and use Save template.");
        file.addEventListener("change",async()=>{
            try{
                const input=file.files?.[0];if(!input)return;if(input.size>16000000)throw Error("Template backups must be under 16 MB.");
                const imported=validateTemplates(JSON.parse(await input.text()));let added=0,skipped=0;
                await mutate(current=>{const next=[...current],names=new Set(current.map(item=>item.name));for(const entry of imported){if(names.has(entry.name)){skipped++;continue;}names.add(entry.name);next.push(entry);added++;}return next;});
                status.textContent=`Imported ${added} templates; preserved ${skipped} same-name templates.`;
            }catch(error){status.textContent=`Import failed: ${error.message}`;}finally{file.value="";}
        });
        filter.addEventListener("input",render);
        seed.addEventListener("change",()=>localStorage.setItem("restore_seeds",String(seed.checked)));
        copies.addEventListener("change",()=>localStorage.setItem("slideshow_image_count",copies.value));
        window.addEventListener("storage",event=>{if(event.key==="cpp-template-manager-change")reload().catch(error=>{status.textContent=error.message;});});
        panel.addEventListener("toggle",()=>{if(panel.open)reload().catch(error=>{status.textContent=error.message;});});
        window.addEventListener("cpp-generation-result",({detail:{card,request}})=>{
            const row=ui.actions(card,"template-manager");ui.button(row,"","Save as template",()=>{
                selectedName=null;name.value=request.prompt.trim().slice(0,60)||`Seed ${request.seed}`;editor.value=JSON.stringify(request,null,2);panel.open=true;panel.scrollIntoView({block:"start"});
            });
        });
        await reload();status.textContent=`${templates.length} saved templates.`;
        window.CppTemplateManager={reload};
    })().catch(error=>{
        const status=document.getElementById("cpp-template-status");if(status)status.textContent=`Template manager unavailable: ${error.message}`;
        else console.error("Template manager:",error);
    });
})();
