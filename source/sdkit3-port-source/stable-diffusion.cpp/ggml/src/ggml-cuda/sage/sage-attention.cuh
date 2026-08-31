#pragma once

#include "../common.cuh"

// Backend-neutral optimized-attention interface used by the shared CUDA/HIP
// dispatcher. Backend-specific implementations stay behind this API so callers
// never need to know about SM, gfx, CUDA, or HIP implementation names.
enum ggml_attention_impl {
    GGML_ATTN_IMPL_NONE = 0,
    GGML_ATTN_IMPL_SAGE,
    GGML_ATTN_IMPL_XFORMERS,
};

// SageAttention common API.
bool ggml_sage_attn_supported(int device, const ggml_tensor * dst);
void ggml_sage_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// xFormers common API. Backend-specific implementations may be added without
// changing the high-level dispatcher. Until a native xFormers implementation
// is wired for a backend, supported() returns false so callers safely fall back.
bool ggml_xformers_attn_supported(int device, const ggml_tensor * dst);
void ggml_xformers_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// FlashAttention is compatibility-only. It is not part of automatic optimized
// attention selection and should only be requested by an API that explicitly
// requires FlashAttention-compatible behavior. ROCm/Volta implementations may
// satisfy this contract independently. Pascal support is intentionally left as
// a separate compatibility implementation to be added later.
bool ggml_flash_compat_supported(int device, const ggml_tensor * dst);
void ggml_flash_compat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Query an automatically selectable optimized implementation by type.
bool ggml_attention_impl_supported(ggml_attention_impl impl, int device, const ggml_tensor * dst);
