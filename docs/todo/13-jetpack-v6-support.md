# 13 — JetPack v6 Support

## Objective
Support Jetson systems running JetPack 6 while sharing as much architecture as possible with generic ARM64 and JetPack 5.

## Implementation
1. Detect JetPack/L4T version and keep a capability table rather than scattering version comparisons throughout code.
2. Use JetPack 6 platform CUDA/cuDNN/toolchain packages.
3. Enable newer CUDA features only when both hardware SM and toolkit support them.
4. Validate unified/shared-memory behavior, pinned allocations, model offload, and streaming.
5. Add a dedicated build/test preset while reusing the same source paths as JetPack 5.
6. Verify text encoders, UNet/DiT, VAE, LoRA, ControlNet, and attention selection.
7. Test newer Jetson devices independently because a successful x86 CUDA build does not prove ARM CUDA correctness.

## Dependencies
Generic aarch64 support; capability-driven CUDA kernel selection.

## Validation
Clean native build, end-to-end generation, low-memory mode, model switching, cancellation, repeated jobs, and optional video path where hardware capacity allows.

## Complete when
JetPack 6 has a reproducible build/install path and at least one fully-supported generation workflow with documented limitations.
