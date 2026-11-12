# Portable color-guide magic wand

This increment adds **Magic wand (source color)** to the existing portable
Advanced Inpaint guide editor. It does not add an ordinary-mask editor or SAM.

## Existing behavior retained

Both original editors use `ui/media/js/editor-mask-tools.js`:
- Legacy `ui/media/js/image-editor.js`, `magicWandSelect`, samples the background
  and applies current color/alpha to its target layer.
- C++ `source/UI.cpp/Pages/src/Plugin/plugin_scripts/canvas_workspace.js`
  samples its base layer, calls `EditorMaskTools.select`, and fills only the
  selected working layer.

The portable form now bundles that file **verbatim** at the start of generate.js
and calls `window.EditorMaskTools.select`. Both its main DOM test and dedicated
pixel test enforce exact source equality, so canonical changes cannot silently
leave a divergent selection implementation. Update the verbatim block when
updating the canonical helper, then run both tests.

The algorithm selects a four-connected region by squared Euclidean RGB distance
from the clicked **seed color**, not by incremental neighbor-color similarity.
Source alpha is ignored. A buffer expands the result by Manhattan distance.
Coordinates are floored/clamped by the shared helper. The UI bounds sensitivity
to finite 0–255 and the integer buffer to 0–64. Default sensitivity30, buffer0
match the existing editors.

## Portable interaction

The editor decodes a separate source snapshot while loading the optional guide.
Controls are blocked until both are ready. Wand clicks sample only this immutable
snapshot, never already painted guide pixels. A left click fills selected pixels
with brush color and opacity; Erase instead removes guide opacity in that region.
The fill is a single operation, so dragging does not turn it into a brush stroke.
Undo, save and cancel retain the existing guide lifecycle and five-operation
history bound. Cancel during decode invalidates pending callbacks. Original
source and ordinary mask are never changed; the stored guide changes only on Save.
The existing four-megapixel editor limit also bounds selection allocations.

This is a guide-only editor increment. Ordinary inpainting remains mask upload
with independent enable state. Missing selection transforms, full original-editor
parity and SAM are not implied. Existing Legacy and C++ editors are untouched.
SAM's original plugin calls a model-backed `/sam/segment` workflow; this browser
color selection neither loads a segmentation model nor substitutes for it.

## Validation

- `node cosmopolitan/tests/verify_portable_wand.js`: exact shared-source equality
  and 218 real-pixel/differential cases passed. Covers disconnected same colors,
  four-neighbor connectivity, seed-relative colors, exact tolerance boundary,
  row boundary, Manhattan expansion, source-alpha behavior, bounds/nonfinite
  rejection, immutable source, and 200 deterministic comparison cases.
- `node --test scripts/test_editor_mask_tools.cjs`: both canonical tests pass.
- `node cosmopolitan/tests/verify_inference_ui.js`: expanded DOM/mock suite
  passes actual event-handler tests for fill payload color/alpha, erase operation,
  undo, invalid buffer, non-left input, busy/unsupported capability, source/mask
  independence, canceled delayed source decode, and existing generation controls.
  Canvas compositing, browser decoding and network responses remain mocked.
- The pre-wand script fails the new DOM negative control due missing wand controls.
- JavaScript syntax, Python browser-fixture syntax and diff whitespace pass.
- Rendered-page browser fixture now includes wand/undo checks, but was **not run**
  because the previously verified Chromium socket denial remains. No retries.
  No real-browser interaction, GPU, model inference, packaged runtime or SAM
  execution is claimed by these tests.

Evidence and exact incremental patch are under `recovery-evidence/portable-selection`.
