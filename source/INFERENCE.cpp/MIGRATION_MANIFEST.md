# SDKIT3 inference disposition

Audited against the preserved SDKIT3 snapshot on 2026-09-08. “Reimplement”
means reproduce required behavior against INFERENCE.cpp-owned contracts; it
does not authorize copying an engine-owned context or implementation.

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
| model-family orchestration under `src/model/` | graph sequencing without backend tensor ownership |
| top-level SDKIT3 `image_generator.*` | request-to-plan and result-assembly behavior |
| image preprocessing/postprocessing policy | operation ordering, dimensions, and metadata only |

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

An excluded operation may appear in an inference plan as an abstract operation
or resource identifier. Its backend data structure or implementation must not.

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
2. Independent tokenizer layer with golden-vector tests.
3. Conditioning and guidance policy.
4. Scheduler and sampler math using backend-neutral scalar/state contracts.
5. Denoising loop expressed as abstract model operations and resource IDs.
6. Model-family orchestration and image/video result assembly.
7. API.test composition tests that supply DiffUser, without introducing a
   production dependency in either direction.
