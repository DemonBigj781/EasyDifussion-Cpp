const assert = require("node:assert/strict")
const fs = require("node:fs")
const http = require("node:http")
const path = require("node:path")
const root = path.resolve(__dirname, "..")
const plugin = fs.readFileSync(path.join(root, "ui/plugins/ui/files_plugin/online-model-browser.plugin.js"), "utf8")
const panel = fs.readFileSync(path.join(root, "ui/plugins/ui/files_plugin/online-model-browser.plugin.html"), "utf8")
const model = {id:123, name:"Example image source", type:"Checkpoint", creator:{username:"Test creator"},
    modelVersion:{id:456,baseModel:"SDXL 1.0",files:[{id:789,name:"example.safetensors",sizeKB:2048}]}}
const image = {id:101,url:"https://image.civitai.com/test/source.png",width:512,height:768,username:"Artist",
    tags:[{id:1,name:"Landscape"},{id:2,name:"Architecture"},{id:3,name:"Night"}],
    meta:{prompt:"Website prompt",negativePrompt:"Website negative",seed:123,steps:20,cfgScale:7}}
const fileReport = {sizeBytes:1234,sha256:"f".repeat(64),format:"PNG",width:256,height:384,
    embedded:{info:{parameters:"File prompt\nNegative prompt: File negative\nSteps: 8, Sampler: Euler, CFG scale: 1, Seed: 42, Size: 256x384",
        workflow:'{"nodes":[{"id":1,"type":"example"}]}',"XML:com.adobe.xmp":"<xmp>file only</xmp>"},exif:{Software:"Image generator"}}}
const requests = []
let slowImages = false
let failSecondGallery = false
let activeImages = 0, maximumImages = 0
let failEnums = false
const fixture = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Civitai image browser regression</title>
<style>:root{--background-color2:#252b36;--background-color3:#3e4758;--text-color2:#b8c4d6}body{background:#151b26;color:#e7edf8;font:15px system-ui;max-width:1000px;margin:24px auto;padding:12px}input,select,button,textarea{font:inherit;color:inherit;background:#252b36;border:1px solid #607088;border-radius:5px;padding:7px;box-sizing:border-box;min-width:0;max-width:100%}button{cursor:pointer}button:disabled{opacity:.4}textarea{width:100%}a{color:#abcfff}#system-settings-table{display:none}.panel-box{min-width:0}dialog{background:#1c2431;color:#e7edf8;border:1px solid #607088;border-radius:12px;padding:20px}dialog::backdrop{background:#0009}</style></head><body><div id="system-settings-table"></div><select id="sampler_name" hidden><option value="euler">Euler</option></select><script>
window.applied = {}; window.TASK_MAPPING = Object.fromEntries(["prompt","original_prompt","negative_prompt","num_inference_steps","seed","guidance_scale","sampler_name","width","height"].map(key=>[key,{setUI:value=>{applied[key]=value}}]));
window.promptField = document.createElement("input"); window.selectTab = ()=>{};
window.createTab = ({content})=>document.body.appendChild(content);
window.loadRequiredPluginHTML = ()=>${JSON.stringify(panel)};
</script><script src="/plugin.js"></script></body></html>`
const server = http.createServer((req,res) => {
    const url = new URL(req.url,"http://localhost")
    url.requestMethod=req.method
    requests.push(url)
    if(url.pathname==="/"){res.setHeader("Content-Type","text/html");return res.end(fixture)}
    if(url.pathname==="/plugin.js"){res.setHeader("Content-Type","text/javascript");return res.end(plugin)}
    res.setHeader("Content-Type","application/json")
    const send = value=>res.end(JSON.stringify(value))
    if(url.pathname.endsWith("/enums"))return setTimeout(()=>{
        if(failEnums){res.statusCode=503;return send({ok:false})}
        send({ok:true,baseModels:["SDXL 1.0","Fresh API Base"],modelTypes:["Checkpoint","Fresh API Type"],modelFileTypes:["Model","VAE"]})
    },250)
    if(url.pathname.endsWith("/queue"))return send({ok:true,jobs:[]})
    if(url.pathname.endsWith("/search")) {
        if(url.pathname.startsWith("/huggingface-api"))return send({ok:true,items:[{...model,id:"example/model",provider:"huggingface"}],metadata:{hasNext:false}})
        if(url.searchParams.get("query")==="empty")return send({ok:true,items:[],metadata:{}})
        if(url.searchParams.get("cursor")==="model-next")return send({ok:true,items:[{...model,id:125,name:"Next model page"}],metadata:{}})
        return send({ok:true,items:[model,{...model,id:124,name:"Another model"}],metadata:{nextCursor:"model-next"}})
    }
    if(url.pathname.endsWith("/download"))return send(req.method==='POST'?{ok:true,downloadId:"fixture",filename:"example.safetensors",status:"queued"}:{ok:true,jobs:[]})
    if(url.pathname.endsWith("/images")){
        activeImages++;maximumImages=Math.max(maximumImages,activeImages)
        const second=url.searchParams.get("cursor")==="image-next"
        const other=url.searchParams.get("modelId")==="124"
        return setTimeout(()=>{
            activeImages--
            if(other&&failSecondGallery){res.statusCode=503;return send({ok:false,detail:"Fixture image service unavailable"})}
            send({ok:true,items:[{...image,id:other?201:second?102:101}],metadata:{nextCursor:second?null:"image-next"}})
        },slowImages?250:25)
    }
    if(url.pathname.endsWith("/image-metadata"))return send({ok:true,file:fileReport})
    res.statusCode=404;send({ok:false,error:"Unknown fixture endpoint"})
})

async function run(){
    await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve))
    const {chromium}=require(process.env.PLAYWRIGHT_MODULE||"playwright")
    const browser=await chromium.launch({headless:true})
    try{
        const context=await browser.newContext({viewport:{width:1100,height:900},permissions:["clipboard-read","clipboard-write"]})
        const page=await context.newPage(),errors=[]
        await page.addInitScript(()=>{
            if(!localStorage.getItem('civitaiEnumCache'))localStorage.setItem('civitaiEnumCache',JSON.stringify({fetchedAt:1,enums:{ok:true,baseModels:['Saved Offline Base'],modelTypes:['Saved Offline Type']}}))
        })
        page.on("pageerror",e=>errors.push(e.message))
        await page.route("https://image.civitai.com/**",route=>route.fulfill({contentType:"image/svg+xml",body:'<svg xmlns="http://www.w3.org/2000/svg" width="512" height="768"><rect width="512" height="768" fill="#355679"/></svg>'}))
        await page.goto(`http://127.0.0.1:${server.address().port}`)
        const search=page.locator("#civitai-downloader-search")
        await search.waitFor()
        assert.equal(await page.locator('#civitai-downloader-base-model option[value="Saved Offline Base"]').count(),1,'cached enums must be available before the network refresh')
        await page.waitForFunction(()=>document.querySelector('#civitai-downloader-base-model option[value="Fresh API Base"]'))
        assert.ok(await page.evaluate(()=>JSON.parse(localStorage.getItem('civitaiEnumCache')).enums.modelFileTypes.includes('VAE')))
        assert.equal(await page.locator('input[name="civitai-downloader-mode"]').count(),2,"Models and Images remain distinct selections")
        assert.equal(await page.locator('#civitai-downloader-tag-ids').count(),0,'manual tag-ID entry is replaced by autocomplete')
        await search.click()
        const first=page.locator('[data-model-id="123"]'),other=page.locator('[data-model-id="124"]')
        await first.locator("[data-action=image-details]").waitFor()
        await other.locator("[data-action=image-details]").waitFor()
        assert.equal(maximumImages,1,"automatic image requests must be sequential")
        assert.equal(await first.locator("[data-action=dl]").isVisible(),false,"model files are separate and collapsed")
        assert.equal(await first.locator('[data-image-gallery] [data-action=dl]').count(),0)
        await first.locator('[data-model-files] summary').click()
        await first.locator('[data-action=dl]').click()
        await page.waitForFunction(()=>document.querySelector('#civitai-downloader-status').textContent.includes('example.safetensors'))
        assert.equal(requests.filter(url=>url.pathname.endsWith('/download')&&url.requestMethod==='POST').length,1)
        await first.locator('[data-model-files] summary').click()
        const imageRequest=requests.find(url=>url.pathname.endsWith("/images")&&url.searchParams.get("modelId")==="123")
        assert.equal(imageRequest.searchParams.get("modelId"),"123")
        assert.equal(imageRequest.searchParams.get("modelVersionId"),null,"do not restrict a model folder to its newest version")
        assert.equal(imageRequest.searchParams.get("nsfw"),"false")
        await first.locator('[data-action=image-next]').click()
        await page.waitForFunction(()=>document.querySelector('[data-model-id="123"] [data-image-gallery]').textContent.includes("102"))
        assert.equal(requests.filter(url=>url.pathname.endsWith("/images")).at(-1).searchParams.get("cursor"),"image-next")
        assert.match(await other.locator('[data-image-gallery]').innerText(),/201/)
        assert.equal(await page.locator('#civitai-downloader-page').innerText(),'Page 1')
        await first.locator('[data-action=image-prev]').click()
        await page.waitForFunction(()=>document.querySelector('[data-model-id="123"] [data-image-gallery]').textContent.includes("101"))
        await page.screenshot({path:'/tmp/civitai-unified-desktop.png'})
        await first.locator("[data-action=image-details]").click()
        const site=page.locator('[data-report-source="civitai"]'),file=page.locator('[data-report-source="file"]')
        assert.match(await site.locator("textarea").inputValue(),/Website prompt/)
        await page.locator("[data-action=inspect-file]").click()
        await page.waitForFunction(()=>document.querySelector('[data-report-source="file"] textarea').value.includes('File prompt'))
        const report=await file.locator("textarea").inputValue()
        assert.ok(report.includes('workflow')&&report.includes('sha256')&&report.includes('XML:com.adobe.xmp'))
        assert.ok(!report.includes('Website prompt'))
        await file.locator("[data-action=apply-report]").click()
        assert.deepEqual(await page.evaluate(()=>({prompt:applied.prompt,negative:applied.negative_prompt,seed:applied.seed,steps:applied.num_inference_steps})),
            {prompt:"File prompt",negative:"File negative",seed:42,steps:8})
        assert.equal(await page.evaluate(()=>applied.sampler_name),"euler")
        await site.locator("[data-action=apply-report]").click()
        assert.equal(await page.evaluate(()=>applied.prompt),"Website prompt")
        await file.locator("[data-action=copy-report]").click()
        assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),report)
        await page.screenshot({path:"/tmp/civitai-metadata-desktop.png"})
        await page.setViewportSize({width:390,height:844})
        assert.ok(await page.evaluate(()=>document.querySelector("dialog").getBoundingClientRect().right<=innerWidth))
        assert.ok(await page.evaluate(()=>{const dialog=document.querySelector("dialog");return dialog.scrollWidth<=dialog.clientWidth+1}),"mobile dialog must not clip horizontally")
        await page.screenshot({path:"/tmp/civitai-metadata-mobile.png"})
        await page.locator("[data-action=close-details]").click()
        await page.locator('#civitai-downloader-next').click()
        await page.locator('[data-model-id="125"] [data-action=image-details]').waitFor()
        assert.equal(await page.locator('#civitai-downloader-page').innerText(),'Page 2')
        await page.locator('#civitai-downloader-prev').click()
        await first.locator('[data-action=image-details]').waitFor()
        await other.locator('[data-action=image-details]').waitFor()
        await page.locator('#civitai-downloader-image-sort').selectOption('Most Comments')
        for (const [name,id] of [['Landscape',1],['Architecture',2],['Night',3]]) {
            await page.locator('#civitai-downloader-tag-search').fill(name)
            if (id===3) await page.locator('#civitai-downloader-tag-search').press('Enter')
            else await page.locator(`[data-tag-id="${id}"]`).click()
        }
        assert.equal(await page.locator('[data-selected-tag]').count(),3)
        await first.locator('[data-action=image-next]').click()
        await page.waitForFunction(()=>document.querySelector('[data-model-id="123"] [data-image-gallery]').textContent.includes("102"))
        assert.equal(requests.filter(url=>url.pathname.endsWith('/images')).at(-1).searchParams.get('sort'),'Newest','draft filters must not affect current galleries')
        await page.locator('#civitai-downloader-query').fill('urn:air:sdxl:checkpoint:civitai:123@456')
        await search.click()
        await first.locator('[data-action=image-details]').waitFor()
        const scoped=requests.filter(url=>url.pathname.endsWith('/images')&&url.searchParams.get('modelId')==='123').at(-1)
        assert.equal(scoped.searchParams.get('modelVersionId'),'456')
        assert.equal(scoped.searchParams.get('sort'),'Most Comments')
        assert.equal(scoped.searchParams.get('tags'),'1,2,3')
        assert.equal(requests.filter(url=>url.pathname.endsWith('/civitai/search')).at(-1).searchParams.get('tags'),null,'image tag IDs must not become model-search filters')
        await other.locator('[data-action=image-details]').waitFor()
        failSecondGallery=true
        await search.click()
        await first.locator('[data-action=image-details]').waitFor()
        await page.waitForFunction(()=>document.querySelector('[data-model-id="124"]').textContent.includes('Fixture image service unavailable'))
        assert.equal(await first.locator('[data-action=image-details]').count(),1,'one failed gallery must not hide other results')
        failSecondGallery=false
        await other.locator('[data-action=image-retry]').click()
        await other.locator('[data-action=image-details]').waitFor()
        await page.screenshot({path:'/tmp/civitai-unified-mobile.png'})
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'unified results must fit mobile')
        slowImages=true
        const imageCount=requests.filter(url=>url.pathname.endsWith('/images')).length
        const slowSearch=page.waitForResponse(response=>response.url().includes('/civitai/search'))
        await search.click()
        await slowSearch
        await page.locator('#civitai-downloader-provider').selectOption('huggingface')
        await search.click()
        await page.locator('[data-model-id="example/model"]').waitFor()
        await page.waitForTimeout(400)
        assert.equal(await page.locator("[data-action=image-details]").count(),0,"stale image response must not overwrite a different provider")
        assert.ok(requests.filter(url=>url.pathname.endsWith('/images')).length<=imageCount+1,'obsolete queued galleries must be canceled')
        slowImages=false
        await page.locator('#civitai-downloader-provider').selectOption('civitai')
        await page.locator('#civitai-downloader-query').fill('empty')
        await search.click()
        await page.waitForFunction(()=>document.querySelector('#civitai-downloader-status').textContent.startsWith('0 models'))
        assert.equal(await page.locator('#civitai-downloader-next').isEnabled(),false)
        assert.equal(await page.locator('[data-image-gallery]').count(),0)
        await page.locator('#civitai-downloader-tag-search').fill('custom words not selected')
        const searchesBeforeInvalid=requests.filter(url=>url.pathname.endsWith('/search')).length
        await search.click()
        await page.waitForFunction(()=>document.querySelector('#civitai-downloader-status').textContent.includes('autocomplete'))
        assert.equal(requests.filter(url=>url.pathname.endsWith('/search')).length,searchesBeforeInvalid,'tag names must not be sent as IDs')
        await page.locator('#civitai-downloader-tag-search').fill('')
        await page.locator('input[name="civitai-downloader-mode"][value="images"]').check()
        assert.equal(await page.locator('#civitai-downloader-query').isVisible(),false,'image searches use selected tags, not a custom text query')
        assert.equal(await page.locator('#civitai-downloader-base-model').isVisible(),false)
        assert.equal(await page.locator('#civitai-downloader-sort').isVisible(),false)
        await search.click()
        await page.locator('[data-action=image-details]').waitFor()
        assert.equal(await page.locator('[data-model-id]').count(),0)
        assert.equal(await page.locator('[data-action=dl]').count(),0)
        assert.equal(await page.locator('#civitai-downloader-queue-panel').isVisible(),false)
        let feed=requests.filter(url=>url.pathname.endsWith('/images')).at(-1)
        assert.equal(feed.searchParams.get('modelId'),null)
        assert.equal(feed.searchParams.get('query'),null)
        assert.equal(feed.searchParams.get('tags'),'1,2,3')
        await page.locator('#civitai-downloader-next').click()
        await page.waitForFunction(()=>document.querySelector('#civitai-downloader-results').textContent.includes('102'))
        await page.locator('#civitai-downloader-prev').click()
        await page.waitForFunction(()=>document.querySelector('#civitai-downloader-results').textContent.includes('101'))
        await page.locator('#civitai-downloader-image-sort').selectOption('Random')
        await search.click()
        await page.locator('[data-action=image-details]').waitFor()
        feed=requests.filter(url=>url.pathname.endsWith('/images')).at(-1)
        assert.equal(feed.searchParams.get('sort'),'Random')
        await page.screenshot({path:'/tmp/civitai-images-mobile.png'})
        await page.locator('input[name="civitai-downloader-mode"][value="models"]').check()
        assert.equal(await page.locator('#civitai-downloader-image-sort').inputValue(),'Newest')
        assert.equal(await page.locator('#civitai-downloader-query').isVisible(),true)
        await page.locator('#civitai-downloader-query').fill('model again')
        await search.click()
        await first.locator('[data-action=image-details]').waitFor()
        await other.locator('[data-action=image-details]').waitFor()
        failEnums=true
        await page.reload()
        await search.waitFor()
        await page.waitForTimeout(350)
        assert.equal(await page.locator('#civitai-downloader-base-model option[value="Fresh API Base"]').count(),1,'offline loads retain the saved API enums')
        assert.equal(requests.filter(url=>url.pathname.endsWith('/enums')).length,2,'refresh enums on every load')
        assert.deepEqual(errors,[])
        assert.equal(await page.locator('[data-selected-tag]').count(),3,'selected autocomplete tags survive reload')
        console.log("PASS: Models with bound galleries, Images direct API feed, autocomplete-only tags, enum cache/refresh/offline loading, independent paging, cancellation, downloads, two-source metadata, desktop/mobile")
    } finally {await browser.close();server.close()}
}
run().catch(error=>{console.error(error);server.close();process.exitCode=1})
