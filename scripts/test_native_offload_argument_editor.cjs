"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../ui/media/js/parameters.js"), "utf8");

function editor(saved) {
    const controls = new Map();
    function element(id) {
        if (!controls.has(id)) {
            controls.set(id, {
                value: "",
                checked: false,
                listeners: new Map(),
                addEventListener(event, listener) { this.listeners.set(event, listener); },
                setCustomValidity() {},
                checkValidity() { return true; },
            });
        }
        return controls.get(id);
    }

    const hidden = element("backend_commandline_args");
    const additional = element("native-backend-additional-arguments");
    hidden.value = saved;
    element("native-backend-arguments-editor").querySelectorAll = () => [additional];

    // Execute the real editor and its format helpers in a minimal DOM fixture.
    const options = source.slice(source.indexOf("const NATIVE_BACKEND_ATTENTION_MODES"),
                                 source.indexOf("const MODEL_DIRECTORY_OPTIONS"));
    const markup = source.slice(source.indexOf("function nativeBackendArgumentId"),
                                source.indexOf("/** @type {Array.<Parameter>} */"));
    const behavior = source.slice(source.indexOf("function splitNativeBackendArguments"),
                                  source.indexOf("// listen to parameters from plugins"));
    const context = vm.createContext({ document: { getElementById: element } });
    vm.runInContext(options + markup + behavior, context);
    return { context, element, hidden, additional };
}

test("offload editor hides manual VRAM caps but keeps offload/streaming controls", () => {
    const { context } = editor("");
    const markup = vm.runInContext(
        'renderNativeBackendArgumentsEditor({id: "backend_commandline_args"})', context,
    );
    assert.doesNotMatch(markup, /--(?:video-)?max-vram/);
    for (const flag of ["--offload-to-cpu", "--stream-layers",
                        "--video-offload-to-cpu", "--video-stream-layers"]) {
        assert.ok(markup.includes(flag), flag);
    }
});

test("loading and saving an old profile preserves both manual VRAM arguments", () => {
    const saved = "--max-vram 'cuda=6, cpu=0' --video-max-vram=4 --video-offload-to-cpu";
    const { hidden, additional, element, context } = editor(saved);
    assert.equal(hidden.value, saved);
    assert.equal(additional.value, "--max-vram 'cuda=6, cpu=0' --video-max-vram=4");
    const streaming = vm.runInContext('nativeBackendArgumentId("--video-stream-layers")', context);
    element(streaming).checked = true;
    additional.listeners.get("change")();
    assert.match(hidden.value, /--video-stream-layers/);

    // The controls may move their own tokens, but legacy extras stay intact,
    // including the argument with whitespace and the attached-value spelling.
    const savedAgain = vm.runInContext(
        'splitNativeBackendArguments(document.getElementById("backend_commandline_args").value)', context,
    );
    assert.deepEqual(Array.from(savedAgain), [
        "--video-offload-to-cpu", "--video-stream-layers",
        "--max-vram", "cuda=6, cpu=0", "--video-max-vram=4",
    ]);
});
