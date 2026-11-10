const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")
const {test} = require("node:test")
const root = path.resolve(__dirname, "..")
const source = name => fs.readFileSync(path.join(root, name), "utf8")
const plugin = source("ui/plugins/ui/video_plugin/native-video.plugin.js")
const html = source("ui/plugins/ui/video_plugin/native-video.plugin.html")
const checkpoint = (model, family, type = "stable-diffusion") => ({model, tags: [type, family]})
const models = [
    checkpoint("image", "sdxl"), checkpoint("renamed-video", "mochi_v1_preview"),
    checkpoint("ltx-distilled", "ltx_video_v0_9_7"), checkpoint("wan", "wan_t2v_14b"),
    checkpoint("svd", "svd"), checkpoint("external/mochi", "mochi_v1_preview", "video"),
    ...["image-vae", "mochi/mochi_vae", "mochi/mochi_vae_fp8_e4m3fn", "wan/wan_2.1_vae"].map(model => ({model, tags: ["vae"]})),
    ...["image-encoder", "t5xxl_fp8_e4m3fn", "umt5_xxl_fp8_e4m3fn_scaled"].map(model => ({model, tags: ["text-encoder"]})),
    {model: "motion-module", tags: ["motion-module", "animatediff"]},
    {model: "t5xxl_fp16", tags: ["text-encoder"], installed: false},
]

function harness({selected = "image", state = {}, catalog = models} = {}) {
    const elements = new Map()
    class Element extends EventTarget {
        constructor(id = "") {
            super(); this.id = id; this.dataset = {}; this.value = ""; this.checked = false
            this.parentNode = {after() {}}; this.style = {}; this.attrs = {}; this.hidden = false
            const classes = new Set()
            this.classList = {add: c => classes.add(c), remove: c => classes.delete(c),
                contains: c => classes.has(c), toggle(c, on) {on ? classes.add(c) : classes.delete(c)}}
        }
        set innerHTML(value) {
            for (const match of value.matchAll(/id="([^"]+)"/g)) elements.set(match[1], new Element(match[1]))
        }
        getAttribute(name) {return this.attrs[name] || null}
        removeAttribute(name) {delete this.attrs[name]}
        appendChild() {}
        after() {}
        querySelectorAll() {return [...elements.values()].filter(e => e.id.startsWith("native-video-"))}
    }
    for (const id of ["editor-settings", "stable_diffusion_model", "vae_model", "text_encoder_model", "native-video-motion-module"]) elements.set(id, new Element(id))
    const document = new EventTarget()
    Object.assign(document, {getElementById: id => elements.get(id), querySelector: id => elements.get(id.slice(1)),
        createElement: () => new Element(), head: {appendChild() {}}})
    const store = new Map([["easy-diffusion-native-video-v1", JSON.stringify(state)]])
    const context = vm.createContext({document, console, Event, setTimeout: fn => fn(),
        localStorage: {getItem: k => store.get(k), setItem: (k, v) => store.set(k, v)},
        PLUGINS: {TASK_BUILD: [], TASK_CREATE: []},
        SD: {VideoTask: class VideoTask {constructor(body) {this.body = body}}},
        loadRequiredPluginHTML: () => html})
    context.window = context
    vm.runInContext(source("ui/media/js/searchable-models.js"), context)
    context.catalog = catalog
    vm.runInContext("modelsDB = buildModelsDB(catalog)", context)
    // Unit harness fields dispatch like the real selectors (browser checks use the real DOM components).
    function field(id, initial) {
        let value = initial
        const el = elements.get(id)
        el.dataset.path = initial
        return {get value() {return value}, set value(next) {value = next; el.dataset.path = next; el.dispatchEvent(new Event("change"))},
            addEventListener: (...args) => el.addEventListener(...args)}
    }
    context.stableDiffusionModelField = field("stable_diffusion_model", selected)
    const dropdownFields = new Map()
    context.FixtureDropdown = class {constructor(el) {
        const result = field(el.id, el.dataset.path || "")
        dropdownFields.set(el.id, result)
        return result
    }}
    vm.runInContext("ModelDropdown = FixtureDropdown", context)
    context.vaeModelField = field("vae_model", "image-vae")
    let encoders = ["image-encoder"]
    context.textEncoderModelField = {get modelNames() {return encoders}, set modelNames(next) {
        encoders = next; elements.get("text_encoder_model").dispatchEvent(new Event("change"))
    }}
    vm.runInContext(plugin, context)
    return {context, elements, store,
        select(model) {context.stableDiffusionModelField.value = model},
        selectMotionModule(model) {dropdownFields.get("native-video-motion-module").value = model},
        motionModule() {return dropdownFields.get("native-video-motion-module").value},
        companions() {return JSON.parse(JSON.stringify({vae: context.vaeModelField.value, encoders: context.textEncoderModelField.modelNames}))},
        task(model = context.stableDiffusionModelField.value) {
            const event = {reqBody: {use_stable_diffusion_model: model, use_vae_model: context.vaeModelField.value,
                use_text_encoder_model: context.textEncoderModelField.modelNames.join(","), sampler_name: "euler_a"}}
            context.PLUGINS.TASK_BUILD.forEach(fn => fn(event)); context.PLUGINS.TASK_CREATE.forEach(fn => fn(event))
            return event
        },
    }
}

test("AnimateDiff motion-module selection routes an image checkpoint to video and clamps frames", () => {
    const h = harness(); h.selectMotionModule("motion-module")
    h.elements.get("native-video-frames").value = "99"
    const task = h.task()
    assert.ok(task.instance instanceof h.context.SD.VideoTask)
    assert.equal(task.reqBody.motion_module_model, "motion-module")
    assert.equal(task.reqBody.video_frames, 32)
})

test("clearing the motion module restores still-image routing", () => {
    const h = harness(); h.selectMotionModule("motion-module")
    assert.ok(h.task().instance instanceof h.context.SD.VideoTask)
    h.selectMotionModule("")
    const task = h.task()
    assert.equal(task.instance, undefined)
    assert.equal(task.reqBody.motion_module_model, undefined)
})

test("native video ignores a preserved AnimateDiff selection until an image checkpoint returns", () => {
    const h = harness({selected: "renamed-video"}); h.selectMotionModule("motion-module")
    const task = h.task()
    assert.ok(task.instance instanceof h.context.SD.VideoTask)
    assert.equal(task.reqBody.motion_module_model, undefined)
    assert.equal(h.motionModule(), "motion-module")
})

test("native video clears SD-only sampling optimizations", () => {
    const h = harness({selected: "renamed-video"})
    const event = {reqBody: {use_stable_diffusion_model: "renamed-video", tome_ratio: 0.25,
        negative_guidance_min_sigma: 1}}
    h.context.PLUGINS.TASK_BUILD.forEach(fn => fn(event))
    assert.equal(event.reqBody.tome_ratio, 0)
    assert.equal(event.reqBody.negative_guidance_min_sigma, 0)
})

test("video toggle uses the existing checkpoint selector", () => {
    assert.ok(html.includes('id="native-video-enabled"'))
    for (const id of ["native-video-model", "native-video-vae", "native-video-text-encoder"]) {
        assert.ok(!html.includes(`id="${id}"`), `${id} must be removed`)
    }
})

test("AnimateDiff controls remain available for still-image checkpoints", () => {
    assert.doesNotMatch(plugin, /panel\.classList\.toggle\("displayNone", !family && !motionModule\.value\)/)
})

test("both caches are independently selectable for real SD1.5 and SDXL architecture tags", () => {
    for (const family of ["sd_v1_base", "sd_xl_base"]) {
        const h = harness({selected:"renamed", catalog:[...models, checkpoint("renamed", family)]})
        h.selectMotionModule("motion-module")
        for (const mode of ["easycache", "teacache", "disabled", "easycache"]) {
            h.elements.get("native-video-cache").value = mode
            h.elements.get("native-video-cache").dispatchEvent(new Event("change"))
            const task = h.task()
            assert.ok(task.instance instanceof h.context.SD.VideoTask)
            assert.equal(task.reqBody.cache_mode, mode)
            assert.equal(h.elements.get("native-video-threshold").disabled, mode === "disabled")
        }
    }
})

test("queued AnimateDiff requests preserve cache and motion settings after the UI changes", () => {
    const h = harness(); h.selectMotionModule("motion-module")
    h.elements.get("native-video-cache").value = "teacache"
    h.elements.get("native-video-threshold").value = "0.08"
    const event = {reqBody:{use_stable_diffusion_model:"image"}}
    h.context.PLUGINS.TASK_BUILD.forEach(fn => fn(event))
    h.selectMotionModule("")
    h.select("wan")
    h.elements.get("native-video-cache").value = "easycache"
    h.context.PLUGINS.TASK_CREATE.forEach(fn => fn(event))
    assert.ok(event.instance instanceof h.context.SD.VideoTask)
    assert.equal(event.reqBody.motion_module_model, "motion-module")
    assert.equal(event.reqBody.cache_mode, "teacache")
    assert.equal(event.reqBody.cache_threshold, 0.08)
})

test("turning AnimateDiff off does not attach cache options to still-image requests", () => {
    const h = harness(); h.selectMotionModule("motion-module")
    const toggle = h.elements.get("native-video-enabled")
    toggle.checked = false; toggle.dispatchEvent(new Event("change"))
    assert.equal(h.task().reqBody.cache_mode, undefined)
    assert.equal(h.task().instance, undefined)
})

test("AnimateDiff uses dedicated video steps instead of still-image options", () => {
    const h = harness(); h.selectMotionModule("motion-module")
    h.elements.get("native-video-steps").value = "12"
    const event = {reqBody:{use_stable_diffusion_model:"image", num_inference_steps:4}}
    h.context.PLUGINS.TASK_BUILD.forEach(fn => fn(event))
    h.context.PLUGINS.TASK_CREATE.forEach(fn => fn(event))
    assert.equal(event.reqBody.num_inference_steps,4)
    assert.equal(event.reqBody.video_steps,12)
})

test("Illustrious and Pony SDXL checkpoints remain valid AnimateDiff image bases", () => {
    for (const model of ["illustrious-checkpoint", "pony-checkpoint"]) {
        const h = harness({selected: model, catalog: [...models, checkpoint(model, "sd_xl_base")]})
        h.selectMotionModule("motion-module")
        const task = h.task()
        assert.ok(task.instance instanceof h.context.SD.VideoTask)
        assert.equal(task.reqBody.motion_module_model, "motion-module")
    }
})
test("dedicated video catalog is available in the main dropdown and metadata lookup", () => {
    const context = vm.createContext({document: {querySelector: () => null}, catalog: models})
    vm.runInContext(source("ui/media/js/searchable-models.js"), context)
    assert.ok(vm.runInContext('JSON.stringify(convertToLegacyModelOptions(catalog)["stable-diffusion"]).includes("external")', context))
    assert.equal(vm.runInContext('buildModelsDB(catalog)["stable-diffusion"]["external/mochi"].tags[1]', context), "mochi_v1_preview")
})
test("selecting a renamed Mochi model auto-selects installed companions and routes video", () => {
    const h = harness(); h.select("renamed-video")
    assert.deepEqual(h.companions(), {vae: "mochi/mochi_vae_fp8_e4m3fn", encoders: ["t5xxl_fp8_e4m3fn"]})
    const task = h.task()
    assert.ok(task.instance instanceof h.context.SD.VideoTask)
    assert.equal(task.reqBody.use_stable_diffusion_model, "renamed-video")
    assert.equal(task.reqBody.scheduler_name, "mochi")
    assert.equal(task.reqBody.video_frames, 25)
})
test("video families select their own resources, embedded SVD clears external resources, and image state returns", () => {
    const h = harness(); h.select("renamed-video"); h.select("wan")
    assert.deepEqual(h.companions(), {vae: "wan/wan_2.1_vae", encoders: ["umt5_xxl_fp8_e4m3fn_scaled"]})
    h.select("ltx-distilled")
    assert.deepEqual(h.companions(), {vae: "", encoders: ["t5xxl_fp8_e4m3fn"]})
    assert.equal(h.task().reqBody.video_steps, 8)
    h.select("svd"); assert.deepEqual(h.companions(), {vae: "", encoders: []})
    h.select("image"); assert.deepEqual(h.companions(), {vae: "image-vae", encoders: ["image-encoder"]})
    assert.equal(h.task().instance, undefined)
    assert.equal(h.task().reqBody.sampler_name, "euler_a")
})
test("video companion overrides survive switching and refresh without touching image settings", () => {
    const h = harness(); h.select("renamed-video"); h.context.vaeModelField.value = "mochi/mochi_vae"
    h.context.document.dispatchEvent(new Event("refreshModels"))
    assert.equal(h.companions().vae, "mochi/mochi_vae")
    h.select("image"); h.select("renamed-video")
    assert.equal(h.companions().vae, "mochi/mochi_vae")
})
test("startup uses the main selection, never a stale separately enabled video checkpoint", () => {
    const h = harness({selected: "image", state: {enabled: true, model: "wan"}})
    assert.equal(h.task().instance, undefined)
    const video = harness({selected: "renamed-video"})
    assert.ok(video.task().instance instanceof video.context.SD.VideoTask)
    assert.equal(video.companions().vae, "mochi/mochi_vae_fp8_e4m3fn")
})
test("saved legacy video companions migrate and unavailable entries fall back to installed choices", () => {
    const h = harness({state: {companions: {"renamed-video": {vae: "mochi/mochi_vae", textEncoder: "missing"}}}})
    h.select("renamed-video")
    assert.deepEqual(h.companions(), {vae: "mochi/mochi_vae", encoders: ["t5xxl_fp8_e4m3fn"]})
})
test("missing required resources are reported and placeholders are not selected", () => {
    const h = harness({catalog: models.filter(m => !m.model.includes("t5xxl_fp8"))}); h.select("renamed-video")
    assert.deepEqual(h.companions().encoders, [])
    assert.match(h.elements.get("native-video-status").textContent, /missing.*text encoder/i)
})
test("request routing follows the queued checkpoint, not a later UI selection", () => {
    const h = harness(); h.select("renamed-video")
    assert.equal(h.task("image").instance, undefined)
    h.select("image")
    const task = h.task("wan")
    assert.ok(task.instance instanceof h.context.SD.VideoTask)
    assert.equal(task.reqBody.use_stable_diffusion_model, "wan")
})
test("an empty image checkpoint still restores its companion selections after video", () => {
    const h = harness({selected:""}); h.select("renamed-video"); h.select("")
    assert.deepEqual(h.companions(), {vae:"image-vae",encoders:["image-encoder"]})
})
test("an image architecture is not classified by a video-like filename", () => {
    const h = harness({selected:"mochi-art",catalog:[...models,checkpoint("mochi-art","sdxl")]})
    assert.equal(h.task().instance, undefined)
})
test("task creation preserves built video options and the absent end-frame snapshot", () => {
    const h = harness(); h.select("renamed-video")
    const event = {reqBody:{use_stable_diffusion_model:"renamed-video"}}
    h.context.PLUGINS.TASK_BUILD.forEach(fn => fn(event))
    h.elements.get("native-video-frames").value = "97"
    h.elements.get("native-video-end-preview").attrs.src = "data:image/png;base64,test"
    h.elements.get("native-video-end-preview").src = "data:image/png;base64,test"
    h.context.PLUGINS.TASK_CREATE.forEach(fn => fn(event))
    assert.equal(event.reqBody.video_frames,25)
    assert.equal(event.reqBody.end_image,null)
})
test("late architecture metadata activates video and still restores image resources", () => {
    const h = harness({selected:"renamed-video",catalog:models.map(m => m.model==="renamed-video" ? {model:m.model,tags:["stable-diffusion"]} : m)})
    assert.equal(h.task().instance,undefined)
    vm.runInContext('modelsDB["stable-diffusion"]["renamed-video"].tags.push("mochi_v1_preview")',h.context)
    h.context.document.dispatchEvent(new Event("refreshModels"))
    assert.equal(h.companions().vae,"mochi/mochi_vae_fp8_e4m3fn")
    h.select("image")
    assert.deepEqual(h.companions(),{vae:"image-vae",encoders:["image-encoder"]})
})
test("LTX-2.3 selects all four resources and forwards its audio VAE and connectors", () => {
    const catalog = [...models, checkpoint("renamed-ltx", "ltx2_3"),
        {model:"ltx-2.3-22b-dev_video_vae",tags:["vae"]},
        {model:"ltx-2.3-22b-dev_audio_vae",tags:["vae"]},
        {model:"gemma-3-12b-it-Q4_K_M",tags:["text-encoder"]},
        {model:"ltx-2.3-22b-dev_embeddings_connectors",tags:["text-encoder"]},
        {model:"ltx-2.0_audio_vae",tags:["vae"]},
        {model:"gemma4-12b",tags:["text-encoder"]}]
    const h = harness({catalog}); h.select("renamed-ltx")
    assert.deepEqual(h.companions(), {vae:"ltx-2.3-22b-dev_video_vae",encoders:["gemma-3-12b-it-Q4_K_M"]})
    const event=h.task()
    assert.equal(event.reqBody.audio_vae_model,"ltx-2.3-22b-dev_audio_vae")
    assert.equal(event.reqBody.embeddings_connectors_model,"ltx-2.3-22b-dev_embeddings_connectors")
    assert.equal(event.reqBody.scheduler_name,"ltx2")
    h.select("image")
    assert.equal(h.task().reqBody.audio_vae_model,undefined)
    assert.deepEqual(h.companions(),{vae:"image-vae",encoders:["image-encoder"]})
})
test("LTX-2.3 reports missing auxiliary resources without substituting other versions", () => {
    const h=harness({catalog:[...models,checkpoint("ltx-2.3-dev","ltx2_3"),
        {model:"ltx-2.0_audio_vae",tags:["vae"]}, {model:"ltx-2.5_embeddings_connectors",tags:["text-encoder"]}]})
    h.select("ltx-2.3-dev")
    assert.match(h.elements.get("native-video-status").textContent,/audio VAE/i)
    assert.match(h.elements.get("native-video-status").textContent,/embedding connectors/i)
    assert.throws(()=>h.task(),/LTX-2.3.*missing/i)
})

const ltxCatalog = [...models,
    checkpoint("video/ltx-2-19b-distilled-Q3_K_M", "ltx2"),
    checkpoint("video/ltx-2.3-22b-distilled-Q3_K_M", "ltx2_3"),
    checkpoint("video/ltx-2.5-dev", "ltx2_5"),
    ...["LTX2_video_vae_bf16", "LTX2_audio_vae_bf16",
        "ltx-2.3-22b-dev_video_vae", "ltx-2.3-22b-dev_audio_vae",
        "ltx-2.3-22b-distilled_video_vae", "ltx-2.3-22b-distilled_audio_vae"].map(model => ({model,tags:["vae"]})),
    ...["gemma-3-12b-it-Q4_K_S", "ltx-2-19b-embeddings_connector_dev_bf16",
        "ltx-2-19b-embeddings_connector_distill_bf16", "ltx-2.3-22b-dev_embeddings_connectors",
        "ltx-2.3-22b-distilled_embeddings_connectors"].map(model => ({model,tags:["text-encoder"]})),
]
test("original LTX-2.0 selects its distilled resources, not 2.3, and switches both ways", () => {
    const h = harness({catalog:ltxCatalog})
    for (const version of ["2", "2.3", "2", "2.3"]) {
        const original = version === "2"
        h.select(`video/ltx-${version}-${original ? "19" : "22"}b-distilled-Q3_K_M`)
        const event = h.task()
        assert.equal(event.reqBody.use_vae_model, original ? "LTX2_video_vae_bf16" : "ltx-2.3-22b-distilled_video_vae")
        assert.equal(event.reqBody.audio_vae_model, original ? "LTX2_audio_vae_bf16" : "ltx-2.3-22b-distilled_audio_vae")
        assert.equal(event.reqBody.embeddings_connectors_model, original ? "ltx-2-19b-embeddings_connector_distill_bf16" : "ltx-2.3-22b-distilled_embeddings_connectors")
        assert.equal(event.reqBody.use_text_encoder_model, "gemma-3-12b-it-Q4_K_S")
        assert.equal(event.reqBody.scheduler_name, "ltx2")
        assert.equal(event.reqBody.sampler_name, "euler")
        assert.equal(event.reqBody.video_steps, 8)
        assert.equal(event.reqBody.guidance_scale, 1)
        assert.equal(h.elements.get("native-video-ltx23-resources").classList.contains("displayNone"), false)
    }
    h.select("image")
    assert.deepEqual(h.companions(), {vae:"image-vae",encoders:["image-encoder"]})
    assert.equal(h.task().reqBody.audio_vae_model, undefined)
})
test("LTX-2.0 refuses missing original companions and does not adopt 2.3 resources", () => {
    const h = harness({catalog:ltxCatalog.filter(m => !m.model.startsWith("LTX2_") && !m.model.startsWith("ltx-2-19b-embeddings"))})
    h.select("video/ltx-2-19b-distilled-Q3_K_M")
    assert.throws(() => h.task(), /LTX-2.0.*missing/i)
    assert.equal(h.companions().vae, "")
})
test("LTX-2.5 is not treated as 2.0 or 2.3", () => {
    const h = harness({catalog:ltxCatalog}); h.select("video/ltx-2.5-dev")
    assert.equal(h.elements.get("native-video-ltx23-resources").classList.contains("displayNone"), true)
    assert.equal(h.task().reqBody.embeddings_connectors_model, undefined)
})
