# CUDA xFormers backend

The CUDA backend implements the normalized Common xFormers contract for NVIDIA GPUs.

`definition/gpu/` owns the CUDA request representation, capability declaration, and stage-by-stage validation rules. `translation/gpu/forward.cu` converts a Common `AttentionRequest` to that representation, verifies the active device and pointer address spaces, launches the fused kernel, converts CUDA errors to a Common result, and self-registers the CUDA translation.

The Common route supports strided device or managed-memory F32/F16/BF16
Q/K/V tensors and F32 output, additive F32/F16/BF16 masks, causal masking,
explicit or maximum-bias ALiBi, soft-cap, attention sinks, GQA/MQA, and head/
batch broadcasting. Value dimensions are limited to 512. Unsupported layouts
and pageable host tensor pointers are rejected. NVIDIA Pascal or newer is
required.

The request carries an opaque execution stream and synchronization policy.
CUDA translation alone interprets that handle as a CUDA stream; Common never
includes CUDA types.

The stable-diffusion.cpp `ggml-cuda/xformers-attention.*` source converts GGML
tensors into this Common request. The former direct GGML implementation is
retained only under `prototype/legacy-ggml-adapter.*`.
