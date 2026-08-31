# 04 — Pascal / Volta / Turing Compatible Flash Attention

## Objective
Provide memory-efficient attention on NVIDIA generations that cannot execute Ampere-oriented FlashAttention kernels.

## Implementation
1. Detect CUDA compute capability and expose an attention capability table instead of one global boolean.
2. Keep current high-performance kernels for architectures they support.
3. For SM60/61 Pascal, SM70/72 Volta, and SM75 Turing, implement or reuse a tiled online-softmax attention path that limits the QK attention matrix footprint.
4. Guard tensor-core-specific instructions by architecture. Pascal may require conventional CUDA cores and FP32 accumulation; Volta/Turing can use generation-appropriate tensor operations where numerically safe.
5. Reuse GGML/stable-diffusion.cpp attention primitives if they already solve portions of this rather than creating a parallel API.
6. Separate functional compatibility from branding: the older-GPU path can be FlashAttention-like/memory-efficient even if it is not the same upstream kernel.
7. Add runtime fallback to standard attention when shape, dtype, head dimension, or architecture is unsupported.

## Dependencies
CUDA backend must report device capability. Attention selection should be backend-aware and shared by image/video modules where possible.

## Validation
Compare output against reference attention within dtype-appropriate tolerance. Benchmark peak VRAM, speed, odd sequence lengths, head dimensions, batch sizes, and cancellation/reload behavior.

## Complete when
Representative Pascal, Volta, and Turing devices can use a validated lower-memory attention path, with reliable fallback and no Ampere-only instruction faults.
