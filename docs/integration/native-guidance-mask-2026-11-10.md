# Native distilled guidance and mask blur

## Behavior

Both native request transports now honor `distilled_cfg_scale`, already sent by
Easy Diffusion's shared WebUI adapter. The value is forwarded to the existing
stable-diffusion.cpp sampling guidance for image and video requests. An absent
field leaves the native initializer's value intact (currently 3.5); explicit zero
is retained. Finite float values, including negative values, follow the native
runtime's semantics. This does not add a second guidance mechanism or change CFG.

Native img2img now honors `mask_blur`, already sent as 4 by the legacy UI. The
Gaussian sigma is measured in final image pixels and is applied once, after the
existing mask resize. Zero or an omitted field is a byte-preserving no-op. The
kernel is normalized, extends to ceil(2.5 * sigma), and clamps at the edges,
following the existing native detailer's Gaussian convention. This is not a claim
of pixel-identical Pillow/Forge filtering. White remains the region to regenerate;
black/white solid masks and the default all-white img2img mask stay unchanged.
Neither UI nor either inpainting selection method was removed or replaced.

The accepted blur interval is 0–64, including fractional sigma. This bounds
per-pixel work to 321 taps per separable pass; it is an explicit new API limit,
not a clamp. Invalid types, nonfinite/out-of-float guidance, and out-of-range blur
receive HTTP 400 before task creation/admission. Blur validation applies only to
img2img, preserving irrelevant-field behavior for txt2img/video. Mask inversion,
full-resolution crop/padding and fill modes remain outside this increment.

The blur keeps the image allocation/dimensions unchanged. It allocates its
workspace before writing pixels, returns failure without changing pixels on
allocation/validation failure, and the generation path frees the decoded mask
before reporting failure. No model ownership changes are involved.

## Verification

- Native tests initialize actual runtime image/video sampling structures, verify
  default 3.5 remains intact and explicit guidance values reach both structures.
- Request tests cover accepted fractional/zero blur and malformed values.
- Pixel tests cover exact zero no-op, all-black/white/gray preservation, a known
  Gaussian impulse, edge-clamped transitions, invalid input and ownership.
- CPU-only HTTP regression rejects malformed options before creating tasks.
- The complete Cosmo patch sequence is rebased; handler equality and host syntax
  compilation check the same behavior in its shared service.

Observed results: the complete CPU-only `sdkit` build passed, all 10 native
CTests passed, all 127 foundation tests passed, and 35 Python UI/SAM/inpainting
regressions passed. The real CPU HTTP check rejected 25 malformed image requests
(12 prompt cases and 13 option cases), each with HTTP 400 and no created task.
The two updated native option/image-buffer targets also passed AddressSanitizer
and UndefinedBehaviorSanitizer. Leak detection was disabled because LeakSanitizer
is incompatible with this executor's ptrace environment; leak detection is not
claimed.

No pretrained model inference, GPU use, video LoRA tests, or full Cosmocc/Windows
package validation is claimed. Runtime component tests cannot establish visual
quality or model-specific distilled-guidance behavior.
