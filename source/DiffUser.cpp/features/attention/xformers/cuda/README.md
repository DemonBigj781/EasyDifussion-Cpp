# CUDA xFormers backend

The CUDA backend implements the normalized Common xFormers contract for NVIDIA GPUs.

`definition/gpu/` owns the CUDA request representation, capability rules,
active-device and pointer validation, fused kernel, launch, synchronization,
and native error results. `translation/gpu/forward.cu` converts a Common
`AttentionRequest` to that representation, invokes the definition, converts
the result, and self-registers the CUDA translation.

The Common route supports strided device or managed-memory F32/F16/BF16
Q/K/V tensors and F32 output, additive F32/F16/BF16 masks, causal masking,
explicit or maximum-bias ALiBi, soft-cap, attention sinks, GQA/MQA, and head/
batch broadcasting. Value dimensions are limited to 512. Unsupported layouts
and pageable host tensor pointers are rejected. NVIDIA Pascal or newer is
required.

The request carries an opaque execution stream and synchronization policy.
Only the CUDA definition interprets that handle as a CUDA stream; Common and
the translation never include CUDA runtime types.

The historical stable-diffusion.cpp `ggml-cuda/xformers-attention.*` source was
a wider GGML compatibility adapter that converted tensors into this Common
request. It remained separate from this normalized translation and from the
former direct implementation under `prototype/legacy-ggml-adapter.*`. Commit
`d883d6e` removed the engine adapter while isolating DiffUser, so current runtime
claims come from API.test's Common route rather than application integration.
