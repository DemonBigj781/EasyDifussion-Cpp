# Portable image-to-image and uploaded-mask UI

The portable native Generate form now offers optional PNG/JPEG initial-image
and mask inputs. Browser-local previews appear after decoding. Generate sends
an unprefixed base64 initial image to the existing native img2img endpoint, with
explicit denoising strength. A supplied mask also sends mask blur and inversion.
White means repaint and black means preserve, before the optional inversion.
The native implementation resizes inputs to the selected output dimensions.
This is uploaded-mask inpainting, not a new painted-mask or segmentation editor.

Inputs accept at most 16 MiB per file and 32 megapixels decoded. These are UI
limits, not native API limits. Clearing or replacing the initial image clears
its associated mask. Revision tokens discard delayed file-read/decode results
following replacement or removal. Generation remains disabled during decoding,
and input mutation is disabled while a request is active. File-read/decode
failure clears the replaced input rather than silently retaining old pixels.

Images remain browser-local until Generate and are excluded from saved recipe
settings. Clearing the source restores the existing txt2img route. Request
identity, progress, cancellation and uncertain-network handling are shared with
the existing native generation path. Older servers without img2img_ui=true
hide and disable the new inputs. Mask controls separately require
uploaded_mask_inpainting_ui=true, so an img2img-only server cannot receive an
unsupported mask from this form. The current server advertises img2img_ui and
uploaded_mask_inpainting_ui; this does not advertise advanced editor parity.

The Python UI, its painting/automatic masking methods, and the shared C++ page
source are untouched. Companion model, LoRA, ControlNet and plugin controls in
this portable form remain unavailable. Full-resolution crop/fill modes and
SAM/color-mask UI integration are outside this increment.

## Validation

- C++ UI renderer built successfully and rendered the real Generate page.
- Expanded Node DOM/mock tests pass: image and mask payloads, clear/replace,
  delayed-read races, decode error, older-server gating, pixel-free saved
  defaults, plus existing generation/cancellation/transport-error behavior.
- All 46 Python regressions, all 11 existing CPU native CTests and eight Settings
  DOM/mock cases passed.
- All 128 foundation tests passed, including patch replay and host compilation.
- JavaScript syntax and Python browser-fixture syntax checks passed.
- A real Chromium fixture was added at cosmopolitan/tests/verify_img2img_browser.py.
  It uses the rendered C++ HTML, production generation script and actual PNG
  bytes with mocked API responses. Execution was attempted normally and with
  approved escalation, but Chromium aborted at startup because socket() was
  disallowed by the execution environment. Exact failure:
  `FATAL:chrome/browser/process_singleton_posix.cc:297] Check failed: . socket() failed: Operation not permitted (1)`. No browser rendering, real image
  decoding or browser interaction pass is claimed from that fixture.

Run the browser fixture in a compatible local test environment with:

    python cosmopolitan/tests/verify_img2img_browser.py --renderer /path/to/easy-diffusion-ui-render --chromium /path/to/chromium

No pretrained generation, GPU execution, full Cosmocc packaging or Windows
execution was performed. Successful mocked API tests are not model inference.
