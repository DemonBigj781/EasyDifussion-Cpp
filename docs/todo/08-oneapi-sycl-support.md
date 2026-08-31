# 08 — oneAPI / SYCL Support

## Objective
Make oneAPI/SYCL a first-class compute backend for the Easy Diffusion generation stack.

## Existing foundation
The vendored stable-diffusion.cpp already contains `ggml-sycl`. The preferred work is to enable, expose, test, and extend that implementation, not create a competing tensor engine.

## Implementation
1. Add a supported SYCL build preset using Intel oneAPI/DPC++ or another compatible toolchain required by the vendored backend.
2. Enumerate SYCL platforms/devices at runtime and expose name, vendor, global memory, capability, and stable selector/index.
3. Integrate device selection into sdkit startup and backend configuration.
4. Validate graph operations needed by CLIP/text encoding, UNet/DiT, VAE, ControlNet, LoRA application, and attention.
5. Add correct device-local allocation, USM/buffer usage, transfer synchronization, and VRAM accounting.
6. Make unsupported operations explicit and provide controlled CPU fallback where correct.
7. Add multi-device scheduling that treats each physical GPU's memory independently.
8. Ensure conversion utilities may use SYCL acceleration but do not require it.
9. Keep CPU/CUDA/ROCm builds intact and make SYCL optional at compile time.

## Related design
See `docs/easy-diffusion-oneapi-plan.md` and `docs/native-hf-lora-gguf-plan.md`.

## Validation
End-to-end generation on a supported Intel GPU, model switch, VAE, text encoder, LoRA, ControlNet, low-memory mode, repeated jobs, and output comparison against CPU/reference.

## Complete when
Easy Diffusion can run the full supported generation pipeline on SYCL with clear device/memory reporting and reliable fallback.
