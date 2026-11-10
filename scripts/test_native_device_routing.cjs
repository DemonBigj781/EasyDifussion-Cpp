const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")
const {test} = require("node:test")
const root = path.resolve(__dirname,"..")

for (const file of ["source/UI.cpp/Pages/src/Plugin/plugin_scripts/native-device-routing.js",
                    "ui/plugins/ui/image_plugin/native-device-routing.plugin.js"]) {
    test(`${file}: an explicit Intel selection must not be rewritten as NVIDIA`, async () => {
        const initial = JSON.stringify({image:{diffusion:"CUDA0"},video:{diffusion:"Vulkan1"}})
        let stored = initial
        const status = {dataset:{}}
        const select = {dataset:{nativeScope:"video",nativeModule:"diffusion"},options:[],
            replaceChildren(...items){this.options=items}, add(item){this.options.push(item)},addEventListener(){}}
        const context = vm.createContext({console,
            document:{getElementById:id=>id==="native-device-routing-status"?status:null,
                querySelectorAll:()=>[select],createElement:()=>({}),head:{appendChild(){}},addEventListener(){},dispatchEvent(){}},
            CustomEvent:class {constructor(type){this.type=type}},
            localStorage:{getItem:()=>stored,setItem:(_key,value)=>{stored=value}},
            Option:class {constructor(text,value){this.text=text;this.value=value}},
            PLUGINS:{TASK_BUILD:[]},
            fetch:async()=>({ok:true,json:async()=>({devices:[
                {selector:"Vulkan0",backend:"Vulkan",description:"NVIDIA GeForce RTX 3060"},
                {selector:"Vulkan1",backend:"Vulkan",description:"Intel Graphics SG1"},
            ]})}),
        })
        context.window=context
        vm.runInContext(fs.readFileSync(path.join(root,file),"utf8"),context)
        await new Promise(setImmediate)
        assert.equal(context.NativeDeviceRouting.assignmentFor("video"),"diffusion=Vulkan1")
        assert.equal(context.NativeDeviceRouting.assignmentFor("image"),"diffusion=CUDA0")
        assert.equal(stored,initial)
    })
}
