# Portable Advanced Inpaint input and brush controls

The portable native form now exposes separate ordinary-mask and Advanced
Inpaint color-guide inputs. `advanced_inpaint_ui=true` gates the latter.
The canonical request field is `inpaint_color_guide`; this does not alias the
older `inpaint_color_guidance` overlay. Both original UI implementations remain
untouched. This is meaningful input/brush coverage, **not full editor parity**.

## Interaction contract

- Initial image and mask remain PNG/JPEG. Guide upload accepts PNG with an
  explicit RGBA or gray-alpha channel, nonzero alpha, and source-matching
  dimensions. RGB PNGs and empty guides are rejected rather than silently used.
  These are browser checks in addition to native validation.
- Ordinary mask and guide have independent enable controls. Disabling retains
  the layer and preview; Generate omits that input. Guide-only, mask-only, and
  combined requests use the same img2img route. Guide requests require strength
  strictly between zero and one. Native preprocessing combines masks, then
  applies mask inversion/blur; ordinary-mask inversion is offered only while
  the ordinary mask is enabled.
- Previews and inputs stay local until Generate, never in saved defaults.
  Source replacement/clear clears both associated layers. Bad replacement
  clears that input; delayed decode completions cannot resurrect removed data.
  Busy generation disables editing, and source pixels are never painted over.
- A small local guide editor supports pointer brushes, color, opacity, erasing,
  five-stroke undo, clear, save, and cancel. Existing guide pixels initialize the
  overlay; the source is a CSS background, not part of exported pixels. Export
  is RGBA PNG at original source dimensions. Cancel does not change the stored
  guide. Generate stays disabled during editing and guide decoding.
- Upload limits remain 16 MiB/32 megapixels. Brush editing is limited to
  4 megapixels to bound its undo storage; larger prepared guides can be uploaded.
  Pointer painting has upload as its alternative, with labeled controls.

Missing parity remains explicit: automatic masks/SAM, wand/selection tools,
full-resolution crop/fill modes, and the complete original editors are not
implemented by this compact portable form. It does not replace either original
UI or claim those workflows work natively.

## Validation of this change

- Standalone actual C++ renderer compiled with host g++ from its listed CMake
  sources. No CMake executable was installed in this workspace.
- Expanded executable Node DOM/mock suite passes. It exercises actual handlers
  for independent guide/mask requests, guide-only and ordinary-only, alpha,
  empty-guide/dimension/strength errors, capability gating, pixel-free saved
  defaults, busy restrictions, pointer paint/erase, undo/cancel/save, source
  immutability, plus existing generation/progress/cancellation/network tests.
  Canvas pixels, decoding and API transport are mocked in this suite.
- Negative control with the old production generation script fails because
  Advanced guide controls do not exist.
- JavaScript syntax, Python browser-fixture syntax and diff whitespace pass.
- Real rendered-page Chromium regression was expanded with actual PNG decode,
  canvas strokes, undo/cancel/save and RGBA export checks. Execution was attempted
  normally and once with approved escalation. Both failed before browser startup:
  `FATAL:chrome/browser/process_singleton_posix.cc:297] Check failed: . socket() failed: Operation not permitted (1)`.
  Therefore **no real-browser pass is claimed**. Retry only on an environment
  supporting those local sockets. The test intercepts API calls and does not
  perform model inference.

Full application rebuilding and actual native conditioning validation belong to
separate integration evidence. This UI change alone proves neither.
