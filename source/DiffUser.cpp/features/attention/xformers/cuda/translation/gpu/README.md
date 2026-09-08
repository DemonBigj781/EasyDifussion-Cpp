# xFormers-compatible CUDA attention

This directory contains the sole production CUDA backend translation for
xFormers-compatible attention. It is inference-only and has no PyTorch/ATen
dependency.

## Common API bridge

`forward.cu` is the only CUDA translation registered with `edcpp::api::attention::xformers::common`. It:

1. receives a normalized `AttentionRequest` from Common;
2. translates it to `cuda::definition::gpu::NativeRequest`;
3. runs the CUDA definition validation;
4. validates the active NVIDIA device and CUDA-visible pointer address spaces;
5. launches one online-softmax fused kernel; and
6. synchronizes and reports success or failure through the Common Boolean result.

This path advertises fused forward only. The superseded materialized
`qkt`/`mask`/`softmax`/`av` CUDA files were moved to `../../../prototype/` and
are excluded from builds. The fused kernel satisfies those semantics internally
without a materialized score buffer.

The translation accepts strided CUDA device or managed-memory F32/F16/BF16
Q/K/V, F32 output, additive F32/F16/BF16 masks with broadcasting, causal mode,
explicit or maximum-bias ALiBi, soft-cap, attention sinks, GQA/MQA, head/batch
broadcasting, and value dimensions through 512. It rejects pageable host
tensor pointers, invalid strides/broadcasting, non-divisible mappings, and
pre-Pascal NVIDIA devices.

## Application integration

The stable-diffusion.cpp `ggml-cuda/xformers-attention.*` adapter maps
`GGML_OP_FLASH_ATTN_EXT` tensors and parameters into `AttentionRequest`, then
calls only Common validation/forward. Unsupported requests return false so the
application can select its normal fallback. Direct GGML-to-CUDA implementation
code is excluded under `prototype/legacy-ggml-adapter.*`.
