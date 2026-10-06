(() => {
    "use strict";
    const api = window.CppPluginUI = {};
    api.ready = (async () => {
        const policy = await window.CppKiosk.ready;
        if (!window.CppGeneration) await new Promise(resolve => window.addEventListener("cpp-generation-ready", resolve, {once:true}));
        const root = document.querySelector(".generation-controls");
        const prefs = window.LocalPluginPreferences;
        const status = document.createElement("p");status.id="cpp-workbench-status";status.setAttribute("role","status");root.append(status);
        const style=document.createElement("style");style.textContent=`
            [data-workbench-plugin][hidden],#cpp-story-view[hidden] {display:none!important}
            .cpp-workbench {margin-top:1rem;min-width:0}
            .cpp-workbench summary {cursor:pointer;font-weight:600;padding:.5rem 0}
            .cpp-workbench label {display:flex;flex-direction:column;gap:.25rem;margin:.5rem 0;min-width:0}
            .cpp-workbench input,.cpp-workbench textarea,.cpp-workbench select {min-width:0;max-width:100%;box-sizing:border-box}
            .cpp-workbench textarea {width:100%;min-height:5rem}
            .cpp-workbench input[type=checkbox] {width:auto;align-self:flex-start}
            .cpp-workbench-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:.6rem}
            .cpp-workbench button,.cpp-workbench a {overflow-wrap:anywhere}
            .cpp-workbench-list {display:flex;flex-wrap:wrap;gap:.35rem;max-height:22rem;overflow:auto}
            .cpp-workbench-list button {margin:0}
            .cpp-workbench-note {font-size:.9em;opacity:.8}
            .cpp-token-duplicate {outline:2px solid #e8bc67}
            .cpp-template-entry {border-bottom:1px solid var(--ui-border);padding:.6rem 0;overflow-wrap:anywhere}
            .cpp-template-entry strong {display:block}
            #cpp-story-items {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:1rem}
            #cpp-story-items figure {margin:0;min-width:0} #cpp-story-items canvas {width:100%;height:auto}
            #cpp-img2img-source-preview {max-width:100%;max-height:180px;object-fit:contain}
            .cpp-workbench-dialog {color:var(--ui-text);background:var(--ui-panel);max-width:90vw;max-height:85vh;overflow:auto}
            .cpp-workbench-dialog pre {white-space:pre-wrap;overflow-wrap:anywhere}
        `;document.head.append(style);
        const enabled = id => prefs.isEnabled(id) && !(policy.enabled && id==="stig-lora-shuttle");
        function gate(element,id) {element.dataset.workbenchPlugin=id;element.hidden=!enabled(id);return element;}
        function refresh(){document.querySelectorAll("[data-workbench-plugin]").forEach(element=>{element.hidden=!enabled(element.dataset.workbenchPlugin);});}
        function panel(id,title){const section=document.createElement("details");section.id=`cpp-${id}`;section.className="panel-box cpp-workbench";const summary=document.createElement("summary");summary.textContent=title;section.append(summary);root.append(gate(section,id));return section;}
        function note(parent,label){const p=document.createElement("p");p.textContent=label;p.className="cpp-workbench-note";parent.append(p);return p;}
        function field(parent,id,label,value,type="text",attrs={}){
            const row=document.createElement("label");row.htmlFor=id;row.textContent=label;
            const input=document.createElement(type==="textarea"?"textarea":"input");input.id=id;if(type!=="textarea")input.type=type;
            Object.assign(input,attrs);if(type==="checkbox")input.checked=!!value;else input.value=value;
            row.append(input);parent.append(row);return input;
        }
        function select(parent,id,label,options,value){
            const row=document.createElement("label");row.htmlFor=id;row.textContent=label;
            const input=document.createElement("select");input.id=id;for(const [value,text] of options)input.add(new Option(text,value));
            input.value=value;row.append(input);parent.append(row);return input;
        }
        function button(parent,id,label,action){
            const button=document.createElement("button");button.id=id;button.type="button";button.textContent=label;
            button.addEventListener("click",async()=>{
                const plugin=button.closest("[data-workbench-plugin]")?.dataset.workbenchPlugin;
                if(plugin&&!enabled(plugin))return;
                try{await action();}catch(error){status.textContent=error.message;}
            });parent.append(button);return button;
        }
        function actions(card,id){const row=document.createElement("div");row.className="cpp-image-actions";card.append(gate(row,id));return row;}
        function change(field,value){field.value=String(value);field.dispatchEvent(new Event("input",{bubbles:true}));field.dispatchEvent(new Event("change",{bubbles:true}));}
        function download(data,name){const link=document.createElement("a"),blob=data instanceof Blob;link.href=blob?URL.createObjectURL(data):data;link.download=name;link.click();if(blob)setTimeout(()=>URL.revokeObjectURL(link.href),1000);}
        function showInfo(title,content){const dialog=document.createElement("dialog");dialog.className="cpp-workbench-dialog";const heading=document.createElement("h3");heading.textContent=title;const pre=document.createElement("pre");pre.textContent=content;dialog.append(heading,pre);button(dialog,"","Close",()=>dialog.close());dialog.addEventListener("close",()=>dialog.remove(),{once:true});document.body.append(dialog);dialog.showModal();}
        async function image(source){const img=new Image();img.src=source;await img.decode();return img;}
        async function batch(requests){
            if(!requests.length||requests.length>512)throw Error("Choose between 1 and 512 tasks.");
            if(requests.length+window.CppGeneration.state.pending>512)throw Error("This batch would exceed the queue limit.");
            const results=await Promise.allSettled(requests.map(request=>window.CppGeneration.enqueue(request)));
            const failed=results.filter(result=>result.status==="rejected");
            if(failed.length)throw Error(`${failed.length} tasks failed: ${failed[0].reason.message}`);
            status.textContent=`Completed ${requests.length} tasks.`;return results;
        }
        Object.assign(api,{policy,enabled,gate,refresh,panel,note,field,select,button,actions,change,download,showInfo,image,batch,status});
        window.addEventListener("local-plugin-preferences-changed",refresh);
        return api;
    })();
    api.ready.catch(error=>console.error("Plugin workbench:",error));
})();
