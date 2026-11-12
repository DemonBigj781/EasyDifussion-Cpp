const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")
const {test} = require("node:test")

const source = fs.readFileSync(path.join(__dirname, "../ui/media/js/main.js"), "utf8")
const start = source.indexOf("function getCurrentUserRequest() {")
const end = source.indexOf("\nfunction setEmbeddings(", start)
assert.ok(start >= 0 && end > start, "Locate the production request builder")
const requestBuilder = source.slice(start, end)

function harness() {
    const classes = new Set()
    const container = {
        classList: {contains: name => classes.has(name)},
        style: {display: ""},
    }
    const context = {
        refImageContainer: container, refImages: ["data:image/png;base64,reference"],
        IMAGE_STEP_SIZE: 8, IMAGE_REGEX: /^data:image\//,
        activeTags: [], initImagePreview: {src: ""},
        maskSetting: {checked: false}, imageInpainter: {getImg: () => "painted-mask"},
        testDiffusers: {checked: false}, isFluxModel: () => false,
        schedulerSelectionContainer: {style: {display: "none"}},
        PLUGINS: {TASK_BUILD: []},
        document: {getElementById: () => null},
        colorGuideSetting: {checked: false}, advancedInpainter: {appliedColorGuide: null},
    }
    // Supply inert form fields while executing the entire production builder,
    // including the fork's mask handling and task-build extension point.
    for (const name of new Set(requestBuilder.match(/\b\w+Field\b/g))) {
        context[name] = {value: "", checked: false}
    }
    context.widthField.value = context.heightField.value = "512"
    context.numOutputsTotalField.value = context.numOutputsParallelField.value = "1"
    context.seedField.value = "42"
    vm.createContext(context)
    vm.runInContext(requestBuilder, context)
    return {context, container, classes, build: () => context.getCurrentUserRequest().reqBody}
}

test("visible reference images are copied, not shared with queued requests", () => {
    const h = harness(), request = h.build()
    assert.deepEqual(Array.from(request.ref_images), h.context.refImages)
    assert.notEqual(request.ref_images, h.context.refImages)
    h.context.refImages.push("later-image")
    assert.equal(request.ref_images.length, 1)
})

test("model changes hide references without discarding them; switching back restores submission", () => {
    const h = harness()
    assert.equal(h.build().ref_images.length, 1)
    h.classes.add("displayNone")
    assert.equal(Object.hasOwn(h.build(), "ref_images"), false)
    assert.equal(h.context.refImages.length, 1)
    h.classes.delete("displayNone")
    assert.equal(h.build().ref_images.length, 1)
})

test("inline-hidden and missing reference pickers do not submit stale images", () => {
    const h = harness()
    h.container.style.display = "none"
    assert.equal(Object.hasOwn(h.build(), "ref_images"), false)
    h.context.refImageContainer = null
    assert.equal(Object.hasOwn(h.build(), "ref_images"), false)
})

test("empty references are omitted even when the picker is visible", () => {
    const h = harness()
    h.context.refImages = []
    assert.equal(Object.hasOwn(h.build(), "ref_images"), false)
})

test("hiding references preserves inpainting fields and fork task-build hooks", () => {
    const h = harness()
    h.classes.add("displayNone")
    h.context.initImagePreview.src = "data:image/png;base64,initial"
    h.context.maskSetting.checked = true
    h.context.strictMaskBorderField.checked = true
    h.context.PLUGINS.TASK_BUILD.push(event => {event.reqBody.fork_option = "retained"})
    const request = h.build()
    assert.equal(Object.hasOwn(request, "ref_images"), false)
    assert.equal(request.init_image, h.context.initImagePreview.src)
    assert.equal(request.mask, "painted-mask")
    assert.equal(request.strict_mask_border, true)
    assert.equal(request.fork_option, "retained")
})


test("hidden references preserve Mint color guide and sampling controls", () => {
    const h = harness()
    h.classes.add("displayNone")
    h.context.document.getElementById = id => ({value: id === "tome_ratio" ? "0.25" : "0.5"})
    h.context.initImagePreview.src = "data:image/png;base64,initial"
    h.context.colorGuideSetting.checked = true
    h.context.advancedInpainter.appliedColorGuide = "data:image/png;base64,color-guide"
    h.context.strictMaskBorderField.checked = true
    const request = h.build()
    assert.equal(Object.hasOwn(request, "ref_images"), false)
    assert.equal(request.inpaint_color_guide, h.context.advancedInpainter.appliedColorGuide)
    assert.equal(request.strict_mask_border, true)
    assert.equal(request.tome_ratio, 0.25)
    assert.equal(request.negative_guidance_min_sigma, 0.5)
})
