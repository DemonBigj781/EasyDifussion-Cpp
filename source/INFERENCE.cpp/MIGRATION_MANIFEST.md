# SDKIT3 inference disposition

Audited against the preserved SDKIT3 snapshot on 2026-09-08. “Reimplement”
means reproduce required behavior against INFERENCE.cpp-owned contracts; it
does not authorize copying an engine-owned context or implementation.

Before admitting a feature, update the
[dated Common gap audit](../../Audit/2026-09-08/INFERENCE_COMMON_GAP_AUDIT.md).
That audit determines which DiffUser Common contracts and backend translations
must land before the inference implementation may call them.

Primitive requirements apply throughout inference and are tracked in the
[project-wide primitive audit](../../Audit/2026-09-08/INFERENCE_PRIMITIVE_AUDIT.md),
not as a VAE-only subset.

## Reimplement in INFERENCE.cpp

| SDKIT3 reference area | Intended INFERENCE.cpp ownership |
| --- | --- |
| `src/conditioning/` | prompt and conditioning orchestration |
| `src/tokenizers/` | tokenizer algorithms, vocabularies, and token policies |
| `src/runtime/denoiser.hpp` | denoising schedule and model-call sequencing |
| `src/runtime/guidance.*` | classifier-free and model-family guidance policy |
| `src/runtime/gits_noise.h` | inference noise policy |
| sampler/scheduler portions of `src/stable-diffusion.cpp` | independently defined sampling loop |
| generation portions of `include/stable-diffusion.h` | normalized image/video request and result contracts |
| `src/extensions/generation_extension.h` | model-neutral generation extension points |
| model-family orchestration under `src/model/` | model execution and graph sequencing without backend tensor ownership |
| top-level SDKIT3 `image_generator.*` | request-to-plan and result-assembly behavior |
| image preprocessing/postprocessing and delivery | operation ordering, dimensions, pixels/frames, metadata, and completed results |
| VAE encode/decode semantics | INFERENCE-owned shape, scaling, tiling, and execution policy |

## Exclude because DiffUser owns or will own it

| SDKIT3 reference area | DiffUser concern |
| --- | --- |
| `ggml/` and `src/core/tensor*` | backend tensors, allocation, placement, and device execution |
| `src/core/ggml_extend*` | backend/driver operation implementation |
| `src/core/backend_fit*` | device fit and resource placement |
| `src/core/ggml_graph_cut*` and `layer_split_partition*` | device graph partition and split resource execution |
| `src/model_loader.*`, `src/model_manager.*`, `src/weight_manager.h` | model-resource load, residency, ownership, and unload |
| `src/model_io/` | resource/file decoding that populates backend-owned model data |
| all Flash, xFormers, Sage, Flex, and Split attention code | normalized attention definitions and translations |
| `src/runtime/flex_attention.*` | Flex Attention implementation |
| `src/runtime/*cache*`, `easycache.hpp`, `teacache.hpp`, `ucache.hpp` | backend cache resources and cache operations |
| backend flags, device enumeration, offload, mmap, VRAM planning | hardware capability and resource management |
| direct CUDA, HIP, SYCL, Vulkan, OpenCL, Metal, or other driver calls | backend definitions/translations |

An excluded operation may appear in an inference plan only through its DiffUser
Common contract. Its backend definition, translation, data structure, or
implementation must not.

A new Common contract is not considered consumable merely because its header
exists. DiffUser must also supply the applicable backend definitions,
Common-to-backend translations, capability reporting, and tests. Image/mask
input planning is deferred until those resource routes exist.

## Keep only as reference

- The complete `source/sdkit3-port-source` snapshot, including
  stable-diffusion.cpp and GGML.
- SDKIT3 build scripts and backend-specific workflows.
- SDKIT3 server, Crow, and Asio code; REST behavior migrates separately to
  RestAPI.cpp.
- Easy Diffusion web assets and Python UI code; presentation migrates
  separately to UI.cpp.
- SDKIT3 attention implementations. They may inform test vectors and required
  semantics, but are not migration inputs for INFERENCE.cpp.

## Implementation order

1. Stable request/result and execution-plan contracts.
2. Independent tokenizer layer under `features/token/common/` with
   golden-vector tests. **In progress:** the generic vocabulary/merge BPE
   contract and deterministic unit tests exist;
   CLIP byte encoding, model vocab loaders, T5 Unigram, and family-specific
   golden vectors remain.
3. Conditioning and guidance policy.
4. Scheduler and sampler math using backend-neutral scalar/state contracts.
5. Denoising loop expressed as abstract model operations and resource IDs.
6. Model-family orchestration and image/video result assembly.
7. API.test composition tests that prove INFERENCE.cpp reaches hardware and
   resources only through DiffUser Common.

VAE planning is now represented under `Feature/vae/common`. Backend execution
uses `Feature/vae/[backend]/[device]/[function].cpp`. Execution remains
blocked on DiffUser Common handlers for VAE, image, and latent resources plus
the compute primitives and GPU translations required by the inference-owned
encode/decode implementation.
