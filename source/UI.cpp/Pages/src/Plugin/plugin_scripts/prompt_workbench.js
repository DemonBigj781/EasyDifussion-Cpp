(() => {
    "use strict";
    const parseLines=text=>text.split(/\r?\n/).map(line=>line.trim().replace(/\s+/g," ")).filter(Boolean);
    const composePrompt=(before,prompt,after)=>[before,prompt,after].map(value=>value.trim()).filter(Boolean).join(", ");
    if(typeof module!=="undefined"&&module.exports){module.exports={parseLines,composePrompt};return;}
    (async()=>{
        const ui=await window.CppPluginUI.ready,gen=window.CppGeneration;
        const panel=ui.panel("stig-text-to-prompt","Stig text-to-prompt — wildcard files");
        const file=ui.field(panel,"cpp-wildcard-file","Import a UTF-8 text file (one prompt per line)","","file",{accept:".txt,text/plain"});
        const source=ui.field(panel,"cpp-wildcard-lines","Or paste prompt lines","","textarea");
        const before=ui.field(panel,"cpp-wildcard-before","Pre-prompt","","textarea");
        const after=ui.field(panel,"cpp-wildcard-after","Post-prompt","","textarea");
        const search=ui.field(panel,"cpp-wildcard-search","Search prompts","","search");
        const list=ui.select(panel,"cpp-wildcard-select","Selected prompt",[],"");list.size=5;
        const count=ui.note(panel,"No prompts loaded.");count.id="cpp-wildcard-count";
        const composed=ui.field(panel,"cpp-wildcard-current","Combined prompt","","textarea",{readOnly:true});
        const amount=ui.field(panel,"cpp-wildcard-batch-count","Random batch size",4,"number",{min:1,max:512});
        let lines=[],position=0;
        function selected(){if(!lines.length)throw Error("Import or paste wildcard prompts first.");return composePrompt(before.value,lines[position],after.value);}
        function choose(index,apply=true){
            if(!lines.length)return;position=Math.min(lines.length-1,Math.max(0,index));list.value=String(position);composed.value=selected();
            if(apply)ui.change(document.getElementById("prompt"),composed.value);
        }
        function filter(){
            list.replaceChildren();const query=search.value.toLowerCase();
            lines.forEach((line,index)=>{if(line.toLowerCase().includes(query))list.add(new Option(`${index+1}. ${line}`,String(index)));});
            list.value=String(position);
        }
        function load(text){if(text.length>2000000)throw Error("Wildcard files are limited to 2 MB.");lines=parseLines(text);position=0;source.value=text;filter();count.textContent=`${lines.length} prompts loaded.`;choose(0);}
        file.addEventListener("change",async()=>{try{const input=file.files[0];if(input.size>2000000)throw Error("Wildcard files are limited to 2 MB.");load(await input.text());}catch(error){ui.status.textContent=error.message;}});
        source.addEventListener("change",()=>{try{load(source.value);}catch(error){ui.status.textContent=error.message;}});
        search.addEventListener("input",filter);list.addEventListener("change",()=>choose(Number(list.value)));
        for(const field of [before,after])field.addEventListener("input",()=>choose(position));
        for(const [id,label,get] of [["first","First",()=>0],["previous","Previous",()=>position-1],["next","Next",()=>position+1],["last","Last",()=>lines.length-1],["random","Random",()=>Math.floor(Math.random()*lines.length)]])
            ui.button(panel,`cpp-wildcard-${id}`,label,()=>choose(get()));
        ui.button(panel,"cpp-wildcard-use","Use selected prompt",()=>ui.change(document.getElementById("prompt"),selected()));
        function request(prompt){const request=gen.buildRequest();request.prompt=prompt;request.original_prompt=prompt;return request;}
        ui.button(panel,"cpp-wildcard-single","Generate selected",()=>ui.batch([request(selected())]));
        ui.button(panel,"cpp-wildcard-random-batch","Generate random batch",()=>{
            selected();const total=Number(amount.value);if(!Number.isInteger(total)||total<1||total>512)throw Error("Batch size must be 1–512.");
            return ui.batch(Array.from({length:total},()=>request(composePrompt(before.value,lines[Math.floor(Math.random()*lines.length)],after.value))));
        });
        ui.button(panel,"cpp-wildcard-sequential","Generate sequential batch",()=>{
            selected();if(lines.length>512)throw Error("Sequential batches are limited to 512 prompts.");
            return ui.batch(lines.map(line=>request(composePrompt(before.value,line,after.value))));
        });
        window.CppWildcard={getPrompt(index,random=false){
            if(!ui.enabled("stig-text-to-prompt"))throw Error("Enable Stig text-to-prompt first.");selected();
            const at=random?Math.floor(Math.random()*lines.length):index%lines.length;
            return composePrompt(before.value,lines[at],after.value);
        }};
        const spell=ui.panel("toggle-spellcheck","Browser spellcheck");
        const setting=ui.field(spell,"cpp-enable-spellcheck","Enable browser spellcheck",localStorage.getItem("enable_spellcheck")==="true","checkbox");
        const originals=new Map();
        function applySpellcheck(){
            for(const field of document.querySelectorAll("#prompt,#negative_prompt,#cpp-rabbit-modifiers,#cpp-wildcard-lines,#cpp-wildcard-before,#cpp-wildcard-after")){
                if(!originals.has(field))originals.set(field,field.getAttribute("spellcheck"));
                if(ui.enabled("toggle-spellcheck"))field.spellcheck=setting.checked;
                else if(originals.get(field)===null)field.removeAttribute("spellcheck");else field.setAttribute("spellcheck",originals.get(field));
            }
        }
        window.CppBrowserSpellcheck={apply:applySpellcheck};
        setting.addEventListener("change",()=>{localStorage.setItem("enable_spellcheck",String(setting.checked));applySpellcheck();});
        window.addEventListener("local-plugin-preferences-changed",applySpellcheck);
        window.addEventListener("storage",event=>{if(event.key==="enable_spellcheck"){setting.checked=event.newValue==="true";applySpellcheck();}});
        applySpellcheck();
    })().catch(error=>console.error("Prompt workbench:",error));
})();
