# Native mask inversion and production-path tests

The application's img2img API now honors `inpainting_mask_invert`, matching the
existing stable-diffusion.cpp example server: false/zero preserves a supplied
mask, true/nonzero inverts it. Booleans and finite whole-number values are
accepted; malformed types and fractional values return HTTP 400 before task
creation/admission. Irrelevant inversion fields on txt2img/video remain ignored.

This is API parity, not a new UI control: the current Easy Diffusion shared WebUI
adapter does not send this option. Neither UI nor either mask-selection method
has been replaced. Main and Cosmo use identical request handling.

The existing production mask-preparation body now lives in image_utils and is
called directly by ImageGenerator. This permits model-free tests of the actual
encoded-mask path instead of only isolated blur helpers. Preparation decodes one
gray channel, resizes with the existing clamp-to-eight behavior, inverts when
requested, then applies the existing optional Gaussian blur. The supplied mask's
white pixels still mean regenerate. Without a supplied mask, the all-white
img2img default remains unchanged even when inversion is requested, matching the
reference server. Failures release decoded mask storage before throwing.

Tests cover encoded PNG decode, real resize, exact inversion, softened mask edges,
full production ordering, absent-mask preservation, explicit white-to-black
inversion, malformed input, resize failure and blur failure. Request tests cover
boolean/integer/default flags and rejection before tasks. Complete Cosmo patch
replay and shared-handler equality remain part of the regression suite.

This does not implement full-resolution inpaint cropping/padding or fill modes.
No pretrained inference, visual-quality claim, GPU use, video LoRA run, or final
Cosmocc/Windows package verification is implied by these model-free tests.

## Verification results

- Full CPU-only application build succeeded.
- All 10 native component/application CTests passed.
- All 127 foundation tests passed, including all 13 zero-fuzz patches and host
  compilation/equality of shared request handlers.
- Real CPU HTTP tests rejected 28 malformed requests with HTTP 400 and confirmed
  each task was absent afterward.
- All 35 Python UI/SAM/inpainting regressions passed.
- Both changed native targets passed ASan/UBSan. LeakSanitizer was disabled
  (`detect_leaks=0`) because of executor compatibility; no leak-check pass is claimed.

The blur keeps the established native Gaussian convention; these tests do not
claim pixel-identical Pillow or Forge behavior.
