# Portable ordinary-mask editor

The portable form now has a separate **Edit ordinary mask** surface. It does not
replace or share edit storage with Advanced Inpaint. Original Legacy/C++ editors
remain untouched; this is not full original-editor or SAM parity.

## Pixel and request contract

Ordinary mask state is a one-byte grayscale array: white means repaint, black
means preserve. Brush, erase, magic wand, undo, clear, save and cancel operate on
that array, not canvas transparency. The pink-on-source canvas is a preview only.
Save constructs a separate PNG with R=G=B=mask and alpha 255 at every pixel.
An empty/all-black mask is valid and is explicitly saved, not omitted.

Mask inversion and blur remain existing request fields. They are not baked into
editor pixels. Both mask and guide still have independent enable controls;
saving one never alters the other's data or original source. Generation remains
blocked while either editor is open. Closing/canceling leaves stored input data
unchanged; clear/replace source clears both associated inputs as before.

Existing uploaded masks are decoded for editing only if dimensions match the
source and every decoded alpha is 255. RGB luminance conversion exactly uses
native stb's `(77*r + 150*g + 29*b) >> 8`. A transparent or differently-sized
mask is **not modified or discarded** when editing is refused: it can still be
submitted through the unchanged direct upload path. This avoids silently losing
hidden RGB through browser alpha handling or claiming browser resizing matches
native resizing. Supply a source-sized opaque mask to edit it.

Gray blend controls mask values only, not promised proportional denoising.
Downstream diffusion may threshold grayscale masks; native inversion and blur
semantics are unchanged. This editor does not claim soft-mask model quality.

## Selection and stroke behavior

Magic wand reuses the verbatim shared `EditorMaskTools.select`: immutable source
pixels, four-connected seed-relative RGB matching, threshold 0–255 and integer
Manhattan buffer 0–64. It adds white or erases to black with the chosen mask value
blend. Separate clicks are separate operations; wand is not SAM segmentation.

Brush coordinates clamp to the preview's image bounds. The raster helper snaps
endpoints to their containing pixel centers and fills a round-capped segment;
a size 1 click at an integer coordinate reliably paints that pixel. This is a
specified discrete raster rule, not browser-antialias equivalence. Each pointer
stroke unions its coverage and applies blend once against a start snapshot, so
repeated/overlapping move events cannot strengthen the same stroke. Pointer
cancel/lost capture restores that snapshot. Only a successful pointerup enters
undo history, so cancel cannot evict the oldest of five completed operations.

Editors are limited to 4 megapixels and five undo operations; larger source/mask
inputs remain available through uploads. Invalid or stale decoder callbacks do
not repopulate canceled editors. Native/UI resource use and larger interactive
performance are not claimed validated in a browser.

## Validation

- `verify_portable_mask_pixels.js`: 38 exact checks of production pixel helper:
  native RGB luminance, opaque black/gray/white output, transparency rejection,
  pixel-center dots/corners/strokes, erase/clipping, selection blend, validation,
  immutable inputs and exhaustive 256 grayscale roundtrip.
- Expanded `verify_inference_ui.js` DOM/mock suite passes actual ordinary editor
  handlers: gray export bytes, all-black save, white wand, erased/undone/reset
  masks, stroke cancel, repeated-move strength, five-history cancellation,
  source/guide isolation, request-level inversion/blur, busy/unsupported guards,
  opaque/dimension refusals and delayed-decode cancel. Existing Advanced guide,
  shared-wand and generation regressions pass. Canvas/PNG/transport remain mocked.
- Pre-editor production script fails the new negative control because ordinary
  editor controls are missing.
- Existing shared-wand 218 pixel/differential cases still pass.
- Independent integration regression `test_portable_mask_roundtrip.py` performs
  actual production JS export/import -> Pillow PNG -> native `prepareMaskImage`:
  twelve native decode/inversion operations across black, white, gray, stroke and
  color-import cases pass, including exact original-color/native-luminance match.
  It is not browser canvas encoding or model inference.
- JavaScript syntax, Python browser-fixture syntax and diff whitespace pass.
  The real-browser fixture was expanded but **not run**; the existing Chromium
  socket restriction remains, and no retry was attempted. No browser, packaged
  runtime, model-quality, GPU, SAM or full editor parity claim is made.

Evidence and before files: `recovery-evidence/portable-mask-editor`. Native
roundtrip test files are owned by the coordinating integration change and are
not duplicated in the UI-only incremental patch.
