#pragma once

#include "../../../../common/ggml-attention-common.h"
#include "../../../../flash/cuda/translation/gpu/common.cuh"

// Backend-neutral optimized-attention interface used by the shared CUDA/HIP
// dispatcher. Backend-specific implementations stay behind this API so callers
// never need to know about SM, gfx, CUDA, or HIP implementation names.

// SageAttention common API.
bool ggml_sage_attn_supported(int device, const ggml_tensor * dst);
void ggml_sage_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// xFormers common API. CUDA delegates to the ggml-native implementation in the
// sibling xformers directory. Other backends return false so callers safely
// fall back without changing the high-level dispatcher.
bool ggml_xformers_attn_supported(int device, const ggml_tensor * dst);
void ggml_xformers_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// FlashAttention is compatibility-only. It is not part of automatic optimized
// attention selection and should only be requested by an API that explicitly
// requires FlashAttention-compatible behavior. CPU uses the adapter declared in
// ggml-attention-common.h; ROCm/Volta implementations satisfy the same contract
// here. Pascal support remains a separate compatibility implementation.
bool ggml_flash_compat_supported(int device, const ggml_tensor * dst);
void ggml_flash_compat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Query an automatically selectable optimized implementation by type.
bool ggml_attention_impl_supported(ggml_attention_impl impl, int device, const ggml_tensor * dst);
