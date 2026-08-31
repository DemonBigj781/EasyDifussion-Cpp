# SageAttention — ROCm

## Status

**Stub / Fallback**.

## Implementation

The common SageAttention symbols exist for HIP/ROCm, but the native gfx/MFMA SageAttention kernel is not wired yet. The current ROCm implementation deliberately reports unsupported so the shared dispatcher can choose an existing HIP attention path instead of referencing CUDA-only symbols.

## Hardware targets

No ROCm GPU target should currently be treated as having native SageAttention support through this API.

## Precision / data types

Not applicable to a native SageAttention kernel yet. The effective restrictions are those of the fallback HIP attention implementation selected by the dispatcher.

## Compile-time requirements

- HIP/ROCm backend enabled.
- Common SageAttention dispatch source compiled for HIP.

## Runtime capability check

`ggml_sage_attn_supported(device, dst)` currently returns `false` for the ROCm path.

## Fallback

The shared attention dispatcher should select the existing supported HIP attention implementation when SageAttention is requested but unavailable.

## Known limitations

- No native gfx/MFMA SageAttention implementation is currently wired.
- Calling `ggml_sage_attn()` directly on ROCm is not a valid fallback path; the implementation aborts because native SageAttention is unavailable.

## Validation

Compile integration exists, but native ROCm SageAttention runtime validation is not applicable until a native kernel is implemented.

## Relevant source paths

- `source/sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda/sage/sage-attention.cuh`
- `source/sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda/sage/sage-attention.cu`
