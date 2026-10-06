# Anima IP-Adapter: integration findings

Status: **native CPU/CUDA inference is implemented, validated, and deployed**.
Both frontend control paths and the matched native library bundle are installed.
The application restarted successfully, the model catalog identifies both
pinned checkpoints correctly, and the current kiosk preference was preserved.

## Using the native adapter

Select an Anima checkpoint, then open **IP-Adapter** in the legacy image settings
or the C++ page's **Generation Plugins** section. Select the character-reference
adapter and **SigLIP2 base patch16-512**, upload an RGB reference image, enable
the adapter, and set its strength and active step range.

The adapter is discovered by its tensors, not just its filename. An ordinary
CLIP encoder is not interchangeable with SigLIP2, even if its embedding width
is also 768. The pickers filter incompatible encoder choices, and the native
loader rejects incompatible checkpoint shapes and unsupported adapter variants.

Clear the reference or disable the checkbox to return to ordinary generation.
Strength zero and steps outside the selected range disable both the image
residual and the bundled cross-attention LoRA. This is deliberately different
from the author's separately installed LoRA wrappers, which can remain active
at zero image strength. At active strengths, both CFG branches receive the
LoRA correction; only the positive branch receives image attention.

Reference-image contents are not persisted in local storage. Kiosk mode blocks
IP-Adapters, including their bundled LoRAs, without changing the saved kiosk
preference. No ComfyUI process or Python inference sidecar is required.

Use CPU or CUDA for both **Diffusion** and **CLIP Vision** device routing. Anima's
per-block adapter weights follow the diffusion device; the separate IP-Adapter
projection device setting is for the older SD1.x/SDXL projection runner.
Vulkan remains available for existing workloads, but the new adapter rejects
Vulkan routing: its current reduced-precision matrix paths do not meet the
encoder's reference tolerance. This is an explicit restriction, not a silent
CPU fallback or a relaxed numerical test.

## Upstream implementation

- Author implementation:
  [LuciferTC9527/ComfyUI-Anima_IP-Adapter](https://github.com/LuciferTC9527/ComfyUI-Anima_IP-Adapter).
- Weights:
  [LuciferTC/Anima-IP-Adapter](https://huggingface.co/LuciferTC/Anima-IP-Adapter),
  revision `99e9c351c9f00bdd188b5added0066a80d3b1de6`.
- The repository currently contains
  `ip_adapter-Character_Reference-10.safetensors` (503,229,576 bytes), rather than
  the generic filename used in the README example.
- The image encoder is `google/siglip2-base-patch16-512`, not the CLIP Vision
  encoder used by the existing SD1.x/SDXL adapter path.
- The implementation is pinned at commit
  `6b77cd0c367d76402174ace2be50d3cb6aa77855`.
- The released checkpoint uses 28 per-block K/V projections and timestep gates.
  It has no compressor, shared projector, IP self-attention, or learned null
  tokens. The native port rejects those other architectures instead of silently
  ignoring their weights.
- The pinned author's inference loader applies only cross-attention LoRA.
  The native port matches that choice: 336 self-attention/MLP LoRA tensors in
  the release are validated as known unused tensors, not applied.
- The encoder revision is `a89f5c5093f902bf39d3cd4d81d2c09867f0724b`.
  Preprocessing uses aspect-preserving Pillow-compatible bilinear resize,
  centered black padding to 512x512, RGB normalization to [-1,1], and the final
  layer-normalized 1024-token vision output.

The author's README states Apache-2.0 for code and CircleStone Labs
Non-Commercial License v1.0 for weights; the Hugging Face card metadata says
`license: unknown`. Resolve that discrepancy before redistribution or commercial
use. Pinned weights were downloaded for local implementation and verification;
they are not redistributed with the source.

## Encoder reference

[SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic
Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786v1)
by Michael Tschannen et al., arXiv version 1, submitted 2025-02-20, provides the
vision-language encoder background. The exact PDF is indexed in the AI Research
skill's existing research collection.

Use the paper for the encoder's training/design discussion. Use the adapter's
code and checkpoint metadata for image preprocessing, feature-layer selection,
compression and Anima conditioning details. The SigLIP 2 paper does not itself
specify this IP-Adapter implementation.

## Native implementation

- `src/model/te/siglip2.hpp` implements the native vision encoder and preprocessing.
- `src/model/adapter/anima_ip_adapter.hpp` validates the checkpoint schema and
  implements K/V projection, timestep gating, and cross-attention LoRA.
- `src/model/diffusion/anima.hpp` captures the normalized cross-attention query
  and adds the gated image residual after the ordinary MLP.
- `src/stable-diffusion.cpp` selects the Anima-specific model-loading and
  conditioning paths while retaining the existing SD1.x/SDXL adapter path.
- `ui/media/js/ip-adapter-compatibility.js` shares compatibility rules between
  the legacy controls and the C++ frontend.

Native paths above are relative to
`source/sdkit3-port-source/stable-diffusion.cpp/` unless prefixed with `ui/`.

The upstream native backend was also checked at commit
`3f8527a46c54ecf4cb4ed6003da8e8982283c73c`. Its README documents IP-Adapter for
SD1.5 and SDXL (including Plus), and its source tree has no SigLIP module.
That earlier upstream inspection motivated the native port rather than a
cosmetic picker change.

The encoder retains F32 tanh GELU rather than the CPU backend's half-precision
lookup table. CUDA explicit-F32 operations bypass TF32 and reduced-precision
small-batch matrix kernels; the previous cuBLAS handle mode is restored after
each explicit-F32 call. This avoids global GPU environment overrides.

## Verification and limits

- Six preprocessing fixtures match Pillow exactly.
- All 786,432 encoder outputs match the pinned Transformers reference within
  maximum absolute error 0.00070 on CUDA and 0.00046 on CPU.
- Adapter attention and all four cross-attention LoRA projections pass PyTorch
  parity checks at blocks 0, 13, and 27. Zero-strength residuals are exact.
- Seven unsupported or malformed checkpoint variants are rejected.
- The complete 28-block denoiser passes positive/negative/disabled/zero tests.
- Real generation with Anima base v1.0, a fixed seed, 256x256 output, and eight
  steps changes when enabled. Disabled-before, disabled-after, and zero-strength
  outputs are byte-identical. This is a correctness smoke test, not a quality
  benchmark.
- Cancellation during conditioned sampling stops the request, and the next
  disabled generation still matches the original baseline exactly.
- Real generation and cancellation/recovery checks pass with Flash Attention
  both disabled and enabled. All nine native regression tests pass.
- Both staged UIs pass payload, compatibility filtering, clear/disable,
  desktop/mobile, and kiosk tests without writing live backend settings.

Only the released per-block character-reference architecture is supported.
Separate image CFG, gray-null encoding, intermediate encoder feature layers,
alternate encoder image sizes, and other upstream adapter architectures are
not exposed. CUDA is verified on the RTX 3060/SM86. Vulkan encoder tests on
RTX 3060 and Intel SG1 failed the full-precision tolerance; those routes are
blocked for this adapter. Other accelerator backends require qualification
before being enabled. Cancellation during the vision-encoding phase has not
been independently stress-tested.
