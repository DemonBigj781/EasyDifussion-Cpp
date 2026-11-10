# xFormers-compatible CUDA attention

This directory contains the sole production CUDA backend translation for
xFormers-compatible attention. It is inference-only and has no PyTorch/ATen
dependency.

## Common API bridge

`forward.cu` is the only CUDA translation registered with `edcpp::api::attention::xformers::common`. It:

1. receives a normalized `AttentionRequest` from Common;
2. translates it to `cuda::definition::gpu::NativeRequest`;
3. calls the CUDA definition for native validation and execution; and
4. translates the native result back to Common.

The active-device checks, CUDA-visible pointer validation, fused kernel,
launch, and synchronization are owned by
`cuda/definition/gpu/forward.cu`, not this translation.

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

## Historical application integration

The former stable-diffusion.cpp `ggml-cuda/xformers-attention.*` adapter mapped
`GGML_OP_FLASH_ATTN_EXT` tensors and parameters into `AttentionRequest`, then
called only Common validation/forward. Unsupported requests returned false so
the application could select its normal fallback. That wider GGML adapter was
removed by the `d883d6e` isolation change and is not part of the current
runtime path. Direct GGML-to-CUDA implementation code remains excluded under
`prototype/legacy-ggml-adapter.*`.
