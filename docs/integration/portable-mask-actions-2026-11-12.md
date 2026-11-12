# Portable ordinary-mask whole-image actions

This bounded offline increment adds **Invert mask pixels (255 − value)** and
**Fill entire mask white**, and routes existing clear through **Clear entire
mask black**. These operate only on the ordinary editor's grayscale array.
No source image, Advanced guide, checkbox enable state, request inversion flag,
blur setting or original Legacy/C++ editor is changed.

## Established references and deliberate scope

- Legacy `ui/plugins/ui/inpaint_plugin/inpainting.plugin.js` implements
  `invert_selection` by replacing RGB with `255 - value`, setting alpha 255 and
  recording history. For grayscale ordinary masks the portable operation has
  that same pixel polarity. The Legacy action also enables its mask checkbox;
  portable editing intentionally preserves its independently selected enable
  state instead of silently changing whether the next request uses the mask.
- Legacy `ui/media/js/image-editor.js` has history-tracked `fill_all` and `clear`.
  Its generic fill uses current canvas brush settings. Portable **Fill entire
  mask white** is deliberately unambiguous: exact 255 regardless of brush color,
  erase, or strength. Clear is exact 0, preserving empty black as a valid mask.
- C++ `canvas_workspace.js` clears only its current mode/layer and snapshots
  history. Portable actions likewise do not modify source or guide layers.

Invert is exact `255 - gray` over the full image. It preserves intermediate
levels rather than thresholding; twice restores all original gray values.
Fill/clear ignore brush strength and erase settings. Each changed operation is
one undo entry; filling an already-white or clearing an already-black mask does
not consume history or evict an older operation. Undo is still bounded to five
completed operations. Cancel discards all editor edits; Save exports the existing
opaque grayscale PNG format. Empty black remains valid.

**Editor inversion differs from request inversion.** The first changes saved
pixels. The request flag still applies afterward to the effective generation
mask and is not toggled/reset by these buttons. Without other conditioning or
blur, applying both inverts twice. The UI labels and help distinguish them;
combined Advanced guide behavior and downstream native masking remain unchanged.
No proportional soft-denoising guarantee is added.

Actions are disabled/guarded during decoding, active brush strokes, closed
editors and active generation. They preserve source/guide previews and existing
uploads. This adds no region flood-fill tool, redo, SAM, full editor parity,
model execution, browser validation or packaged-build claim.

## Validation

- Production JS mask helper: 47 exact checks pass, including gray-preserving
  inversion/twice inversion, exact fill/clear, valid empty opaque export,
  no-op change flags and invalid-argument immutability, plus prior raster/import
  and grayscale tests.
- DOM/mock suite passes actual button handlers and exported production pixel
  arrays: invert gray 128 to 127, fill/undo/clear sequences, no-op fill preserving
  oldest undo, brush-setting independence, cancel/no export, loading/stroke/busy
  guards, unchanged source/guide/enabled state and separate request inversion.
  Canvas/browser decoding/transport remain mocked.
- Prechange script fails new DOM negative control because new actions are absent.
- Shared wand 218 pixel/differential cases still pass; JS syntax and diff checks pass.
- Coordinating integration extended its separate native JS→PNG→mask-preparation
  test to all 256 gray values, invert/twice-invert, fills and request inversion;
  its final 20 decode/inversion operations pass. Those backend-facing test files
  are outside this UI-only patch.
- No network, downloads, browser/access retry, GPU, Media, spending, commit or
  push was performed. Existing browser and full application validation limits
  remain unchanged.

Before files, logs and exact incremental patch are in
`recovery-evidence/portable-mask-actions`.
