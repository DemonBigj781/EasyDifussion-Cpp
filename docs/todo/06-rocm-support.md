# 06 — ROCm Support

## Objective
Add AMD GPU acceleration as a first-class Easy Diffusion backend using HIP/ROCm.

## Implementation
1. Enable the ROCm/HIP backend already provided by GGML/stable-diffusion.cpp where available.
2. Add installer detection for ROCm runtime, compiler/toolchain, supported GPU visibility, and library paths.
3. Route device enumeration through the same backend capability interface used by CUDA and SYCL.
4. Validate model allocation, matmul, convolution/attention operations, UNet/DiT, CLIP, VAE, LoRA, and ControlNet separately before claiming full-pipeline support.
5. Implement VRAM accounting, offload/streaming, and OOM recovery using actual HIP allocation behavior.
6. Gate specialized attention/kernels by GPU architecture rather than assuming CUDA feature parity.
7. Add explicit user device selection and an `auto` mode that can choose ROCm when appropriate.

## Dependencies
Backend abstraction and consistent model-placement logic make this substantially easier.

## Failure behavior
If a required kernel is unsupported, either use a documented CPU/native fallback or fail before starting the expensive generation phase. Never silently switch the whole workload to CPU without reporting it.

## Validation
Test a supported AMD GPU end-to-end, model switching, LoRA, ControlNet, VAE tiling, low-VRAM mode, repeated jobs, and cleanup after cancellation/OOM.

## Complete when
A supported AMD GPU can generate images reliably and backend/device/memory usage are visible in logs and UI.
