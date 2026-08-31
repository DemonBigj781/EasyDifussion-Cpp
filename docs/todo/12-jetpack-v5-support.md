# 12 — JetPack v5 Support

## Objective
Support NVIDIA Jetson systems running JetPack 5 while respecting its older CUDA/L4T/toolchain constraints.

## Implementation
1. Build on generic aarch64 support so JetPack logic only handles Jetson-specific differences.
2. Detect L4T/JetPack version and select compatible CUDA, cuDNN, compiler, and CMake options.
3. Avoid kernels that require CUDA/toolchain features newer than JetPack 5 provides.
4. Add a JetPack 5 build preset and installer path using platform packages rather than desktop x86 CUDA installers.
5. Account for Jetson shared system/GPU memory; model-placement logic must not treat reported GPU memory like a discrete PCIe card.
6. Validate VAE/text-encoder CPU placement, offload/streaming, and low-memory modes.
7. Gate FlashAttention/fused kernels by actual SM and compiler support.

## Dependencies
Generic aarch64 support first. CUDA backend code should already have capability-based kernel selection.

## Validation
Native build on a JetPack 5 target, startup, small image generation, model reload, LoRA, VAE decode, OOM handling, and temperature/sustained-load sanity checks.

## Complete when
A JetPack 5 device can build and execute a supported Easy Diffusion generation without unsupported desktop-CUDA assumptions.
