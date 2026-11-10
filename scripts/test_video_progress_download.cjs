const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")
const {test} = require("node:test")
const root = path.resolve(__dirname,"..")
const main = fs.readFileSync(path.join(root,"ui/media/js/main.js"),"utf8")
const tasks = fs.readFileSync(path.join(root,"ui/media/js/task-manager.js"),"utf8")

test("downloads use GIF bytes' type rather than the selected PNG output setting", () => {
    const saves = []
    const context = vm.createContext({
        getDownloadFilename: (_image,format) => `fixture.${format}`,
        dataURItoBlob: source => ({type:source.split(";")[0].slice(5)}),
        saveAs: (blob,name) => saves.push({type:blob.type,name}),
    })
    vm.runInContext(main.slice(main.indexOf("function imageOutputFormat("),main.indexOf("function modifyCurrentRequest(")),context)
    for (const format of ["png","jpeg","webp"]) context.onDownloadImageClick({output_format:format},{src:"data:image/gif;base64,fixture"})
    assert.deepEqual(saves,[1,2,3].map(() => ({type:"image/gif",name:"fixture.gif"})))
    assert.equal(context.imageOutputFormat("data:image/png;base64,test","gif"),"png")
    assert.equal(context.imageOutputFormat("/image/preview","webp"),"webp")
})

test("GIF input handling must not replace an explicit native video task", () => {
    const source = fs.readFileSync(path.join(root,"ui/plugins/ui/video_plugin/mads-gifs-cpp.plugin.js"),"utf8")
    const start = source.indexOf("PLUGINS['TASK_CREATE'].push(function(event)")
    const end = source.indexOf("// Register selftests",start)
    const context = vm.createContext({PLUGINS:{TASK_CREATE:[]},GIF_HEADER:"data:image/gif;base64,",Buffer,
        Stream:function(){throw Error("Legacy GIF decoding started")}})
    vm.runInContext(source.slice(start,end),context)
    const native = {kind:"native-video"}
    const event = {reqBody:{video_mode:true,init_image:"data:image/gif;base64,R0lG"},instance:native}
    assert.doesNotThrow(()=>context.PLUGINS.TASK_CREATE[0](event))
    assert.equal(event.instance,native)
    assert.throws(()=>context.PLUGINS.TASK_CREATE[0]({reqBody:{video_mode:false,init_image:"data:image/gif;base64,R0lG"}}),/Legacy GIF decoding started/)
})

test("video progress uses effective sampling total and distinguishes decoding", async () => {
    const bar = {style:{}}, message = {style:{}}
    const instance = {isPending:true,step:20,total_steps:31,status:"processing"}
    const task = {instances:[instance],batchCount:1,batchesDone:1,outputMsg:message,
        progressBar:{querySelector:()=>bar},taskStatusLabel:{classList:{add(){},remove(){}}}}
    const context = vm.createContext({SD:{TaskStatus:{processing:"processing",completed:"completed"}},
        millisecondsToStr:value=>String(value),document:{dispatchEvent(){}},console})
    vm.runInContext(tasks.slice(tasks.indexOf("function getTaskUpdater("),tasks.indexOf("function onRenderTaskCompleted(")),context)
    const update = context.getTaskUpdater(task,{video_mode:true},{})
    await update.call(instance,{update:{step:20,total_steps:31,step_time:1,phase:"sampling"}})
    assert.equal(bar.style.width,"65%")
    assert.match(message.innerHTML,/step 20 of 31/)
    instance.step = 31
    await update.call(instance,{update:{step:31,total_steps:31,step_time:1,phase:"decoding",phase_current:7,phase_total:32}})
    assert.match(message.textContent,/Sampling finished.*Decoding video frame 7 of 32/)
})
