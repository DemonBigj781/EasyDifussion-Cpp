# Project-wide inference primitive audit

Audited on 2026-09-08 against all first-party sources under the preserved
stable-diffusion.cpp `src/` reference, not only its VAE paths. The scan found
271 distinct called symbols with the `ggml_` prefix. That count includes
resource, graph, metadata, conversion, and logging functions as well as
mathematical primitives; it is evidence for classification, not a requirement
to reproduce GGML's API.

## Ownership boundary

DiffUser prepares primitive work through Common and implements the selected
backend/device/type route. INFERENCE consumes those prepared primitives to run
conditioning, encoders, diffusion models, denoisers, VAE, image/video models,
postprocessing, and result delivery.

The required implementation layout is:

```text
source/DiffUser.cpp/features/primitives/[backend]/[device]/[primitive_type].cpp
```

Normalized contracts and registration live in
`source/DiffUser.cpp/features/primitives/common/`.

## Required primitive families

| Family | Operations observed or required | Current Common state |
| --- | --- | --- |
| resource/storage | allocate, free, buffer properties, tensor set/get/copy, async transfer, synchronization | partial feature-specific handlers only |
| tensor construction | new tensor 1D–4D, duplicate, zeros, ones, full, arange, random fill | absent |
| dtype/conversion | cast, F16/BF16/F32 conversions, quantize/dequantize, type traits | absent |
| shape/layout | view 1D–4D, reshape 1D–4D, transpose, permute, contiguous, slice, chunk, concat, repeat, pad, roll | absent |
| arithmetic | add, subtract, multiply, divide, negate, scale, bias, clamp, square, square root | absent |
| transcendental | exp, log, sine, cosine, tanh, sigmoid | absent |
| activation | SiLU, GELU variants, ReLU, leaky ReLU, SwiGLU variants | absent |
| reduction | mean, sum rows, L2 norm, tensor difference | absent |
| normalization | norm, layer norm, RMS norm, group norm | operation enum only; no contract |
| linear | matrix multiply, batched/id matrix multiply, linear, quantized linear, selected-expert operations | operation enum only; no contract |
| convolution | 1D/2D/3D, depthwise, direct, transpose, im2col | operation enum only; no contract |
| spatial | pooling, interpolation, nearest upscale, circular/extended padding | absent |
| indexing/routing | get rows, gather/view offsets, top-k argsort, expert routing, mask application | absent |
| embeddings | timestep embedding, RoPE variants, positional operations | absent |
| graph execution | graph construction, allocation/reservation, submit, async compute, abort, synchronize | backend handling fragments only |
| attention | QKT/mask/softmax/AV and fused variants | separate attention family; partial CPU/CUDA coverage |
| adapters/composites | LoRA merge, LoKr/Kronecker, model blocks | decide per operation; model composition stays in INFERENCE |

## Primitive-type source inventory

Every applicable family must be evaluated for `strings`, Boolean/bytes, signed
and unsigned integer widths, F16, BF16, F32, F64, and explicitly defined
quantized types. The existence of `float32.cpp` cannot imply `float16.cpp`, and
CPU support cannot imply GPU or NPU support.

Strings are primarily required for host metadata, prompts, tokenizer inputs,
model keys, and diagnostics. They are not automatically a GPU requirement.
Integer32 is required broadly for token IDs, dimensions, indices, schedules,
and routing. Floating types carry model tensors and numerical operations.

## Backend/device matrix

| Backend/device | Required source path | Current status |
| --- | --- | --- |
| CPU/CPU | `cpu/cpu/[primitive_type].cpp` | absent |
| CUDA/GPU | `cuda/gpu/[primitive_type].cpp` | absent |
| ROCm/GPU | `rocm/gpu/[primitive_type].cpp` | absent |
| oneAPI/CPU | `oneapi/cpu/[primitive_type].cpp` | absent |
| oneAPI/GPU | `oneapi/gpu/[primitive_type].cpp` | absent |
| oneAPI/NPU | `oneapi/npu/[primitive_type].cpp` | absent |
| OpenCL CPU/GPU/NPU | `opencl/[device]/[primitive_type].cpp` | absent |
| OpenVINO CPU/GPU/NPU | `openvino/[device]/[primitive_type].cpp` | absent |
| Vulkan CPU/GPU | `vulkan/[device]/[primitive_type].cpp` | absent |
| OpenGL/GPU | `opengl/gpu/[primitive_type].cpp` | absent |
| Mesa CPU/GPU | `mesa/[device]/[primitive_type].cpp` | absent |
| DirectML/GPU/NPU | `directml/[device]/[primitive_type].cpp` | absent |

## Implementation order

1. Common tensor/resource descriptor and capability registry.
2. CPU `strings`, `integer32`, `float32`, F16, and BF16 storage/conversion.
3. CUDA GPU integer and floating storage/transfer/conversion translations.
4. Layout and elementwise operations across the types actually used by model
   graphs.
5. Matrix, normalization, activation, convolution, and spatial families.
6. Indexing, embeddings, quantized, expert-routing, and specialized operations.
7. Expand to the remaining backend/device matrix using the same contracts.

Each step requires Common dispatch, the exact backend/device/type file, build
coverage, deterministic tests, and applicable GPU runtime evidence before its
status advances.
