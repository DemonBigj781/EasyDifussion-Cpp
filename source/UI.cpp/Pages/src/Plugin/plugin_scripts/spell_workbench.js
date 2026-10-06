(async()=>{
    "use strict";
    const ui=await window.CppPluginUI.ready;
    const panel=ui.panel("spell-tokenizer","Spell tokenizer and merged tag search");
    ui.note(panel,"Reorder comma-separated prompt tokens by dragging or with the arrow buttons. Alt + wheel changes emphasis. Counts are estimates, not an exact model tokenizer. Tag search reuses the installed merged-tag index.");
    const prompt=document.getElementById("prompt");
    const key="spell-tokenizer-plugin";
    const highlight=ui.field(panel,"cpp-token-highlight","Highlight duplicate tokens",localStorage.getItem(`${key}_duplicate_token_highlight`)==="true","checkbox");
    const safe=ui.field(panel,"cpp-token-safe","Use the bundled SFW tag list",ui.policy.enabled||localStorage.getItem(`${key}_taglist_sfw_setting`)==="true","checkbox");
    safe.disabled=ui.policy.enabled;
    const count=ui.note(panel,"");count.id="cpp-token-count";
    const tokens=document.createElement("div");tokens.id="cpp-token-list";tokens.className="cpp-workbench-list";panel.append(tokens);
    const query=ui.field(panel,"cpp-tag-search","Search merged tags and aliases","","search");query.autocomplete="off";
    const status=ui.note(panel,"");status.id="cpp-tag-status";status.setAttribute("role","status");
    const matches=document.createElement("div");matches.id="cpp-tag-results";matches.className="cpp-workbench-list";panel.append(matches);
    let worker=null,ready=false,requestId=0,dragged=null,safeTags=null;
    const normalize=text=>text.trim().replaceAll("_"," ").toLowerCase();
    function parts(){return prompt.value.split(",").map(value=>value.trim()).filter(Boolean);}
    function save(parts){ui.change(prompt,parts.join(", "));}
    function move(from,to){const values=parts();if(to<0||to>=values.length)return;const [value]=values.splice(from,1);values.splice(to,0,value);save(values);}
    function render(){
        if(!ui.enabled("spell-tokenizer"))return;
        const values=parts(),frequency=new Map();for(const value of values)frequency.set(normalize(value),(frequency.get(normalize(value))||0)+1);
        tokens.replaceChildren();
        values.forEach((value,index)=>{
            const token=document.createElement("span");token.className="cpp-token";token.draggable=true;token.dataset.tokenIndex=index;
            if(highlight.checked&&frequency.get(normalize(value))>1)token.classList.add("cpp-token-duplicate");
            const text=document.createElement("span");text.textContent=value;token.append(text);tokens.append(token);
            ui.button(token,"","←",()=>move(index,index-1)).setAttribute("aria-label",`Move ${value} earlier`);
            ui.button(token,"","→",()=>move(index,index+1)).setAttribute("aria-label",`Move ${value} later`);
            ui.button(token,"","×",()=>{const current=parts();current.splice(index,1);save(current);}).setAttribute("aria-label",`Remove token ${value}`);
            token.addEventListener("dragstart",event=>{dragged=index;event.dataTransfer.setData("text/plain",String(index));});
            token.addEventListener("dragover",event=>event.preventDefault());
            token.addEventListener("drop",event=>{event.preventDefault();if(dragged!==null)move(dragged,index);dragged=null;});
            token.addEventListener("dragend",()=>{dragged=null;});
            token.addEventListener("wheel",event=>{
                if(!event.altKey)return;event.preventDefault();const current=parts(),item=current[index];
                current[index]=event.deltaY>0?(item.startsWith("(")&&item.endsWith(")")?item.slice(1,-1):`[${item}]`):(item.startsWith("[")&&item.endsWith("]")?item.slice(1,-1):`(${item})`);
                save(current);
            },{passive:false});
        });
        const longClip=/long[\s_\/-]*clip/i.test(document.getElementById("text_encoder_model")?.dataset.path||"");
        const estimated=Math.floor(prompt.value.length/2.9),limit=longClip?248:75;
        count.textContent=`Estimated ${estimated} tokens / ${limit} ${longClip?"LongCLIP":"CLIP"} reference limit${estimated>limit?" — over reference limit":""}`;
    }
    function show(results){
        matches.replaceChildren();for(const result of results){
            ui.button(matches,"",`${result.tag} (${result.count})`,()=>{
                const tag=result.tag.replaceAll("_"," ");ui.change(prompt,[prompt.value.trim().replace(/,\s*$/,""),tag].filter(Boolean).join(", "));prompt.focus();
            });
        }
        status.textContent=`${results.length} matching tags.`;
    }
    function stopWorker(){worker?.terminate();worker=null;ready=false;requestId++;}
    async function search(){
        const text=query.value.trim().toLowerCase();requestId++;matches.replaceChildren();
        if(!ui.enabled("spell-tokenizer")||text.length<2){status.textContent="Type at least two characters to search.";return;}
        if(safe.checked||ui.policy.enabled){
            stopWorker();const current=requestId;
            try{
                if(!safeTags){const response=await fetch("/cpp-ui/scripts/spell_safe_tags.json");if(!response.ok)throw Error(`HTTP ${response.status}`);safeTags=await response.json();}
                if(current!==requestId||!ui.enabled("spell-tokenizer"))return;
                show(safeTags.filter(item=>normalize(item.tag).includes(normalize(text))).slice(0,20));
            }catch(error){status.textContent=`SFW tag search unavailable: ${error.message}`;}
            return;
        }
        if(!worker){
            status.textContent="Loading the local merged tag index…";
            worker=new Worker("/plugins/core/prompt_plugin/spell-tokenizer.worker.js");
            worker.addEventListener("message",event=>{
                if(event.data?.type==="ready"){ready=true;search();}
                else if(event.data?.type==="result"&&event.data.requestId===requestId&&ui.enabled("spell-tokenizer"))show(event.data.results||[]);
                else if(event.data?.type==="error"){status.textContent=`Merged tag search unavailable: ${event.data.message}`;stopWorker();}
            });
            worker.addEventListener("error",()=>{status.textContent="Merged tag worker failed. Try again or use the SFW list.";stopWorker();});
        }
        if(ready)worker.postMessage({type:"query",query:text,requestId,limit:20});
    }
    query.addEventListener("input",()=>search().catch(error=>{status.textContent=error.message;}));
    query.addEventListener("keydown",event=>{
        if(event.key==="ArrowDown"){matches.querySelector("button")?.focus();event.preventDefault();}
        if(event.key==="Escape")matches.replaceChildren();
    });
    matches.addEventListener("keydown",event=>{
        const options=Array.from(matches.querySelectorAll("button")),index=options.indexOf(document.activeElement);
        if(["ArrowDown","ArrowUp"].includes(event.key)&&options.length){event.preventDefault();options[(index+(event.key==="ArrowDown"?1:-1)+options.length)%options.length].focus();}
        if(event.key==="Escape")query.focus();
    });
    highlight.addEventListener("change",()=>{localStorage.setItem(`${key}_duplicate_token_highlight`,String(highlight.checked));render();});
    safe.addEventListener("change",()=>{localStorage.setItem(`${key}_taglist_sfw_setting`,String(safe.checked));search();});
    prompt.addEventListener("input",render);prompt.addEventListener("change",render);
    document.getElementById("text_encoder_model")?.addEventListener("change",render);
    window.addEventListener("local-plugin-preferences-changed",()=>{if(!ui.enabled("spell-tokenizer")){stopWorker();matches.replaceChildren();}else render();});
    window.addEventListener("pagehide",stopWorker);render();
})().catch(error=>console.error("Spell tokenizer:",error));
