#pragma once

#include "ggml.h"

// Common optimized-attention identifiers. FlashAttention is intentionally not
// listed here: Flash retention is an explicit API-compatibility contract, not
// an automatically selected optimization.
enum ggml_attention_impl {
    GGML_ATTN_IMPL_NONE = 0,
    GGML_ATTN_IMPL_SAGE,
    GGML_ATTN_IMPL_XFORMERS,
};

// CPU FlashAttention compatibility adapter.
//
// The CPU backend already implements GGML_OP_FLASH_ATTN_EXT through
// ggml_compute_forward_flash_attn_ext(). Keep that implementation as the source
// of truth and expose it through the same common compatibility concept used by
// CUDA/HIP. This is a correctness/API-retention path; it does not claim GPU-like
// FlashAttention performance characteristics.
struct ggml_compute_params;

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_flash_attn_ext(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst);

static inline bool ggml_cpu_flash_compat_supported(const struct ggml_tensor * dst) {
    return dst != NULL && dst->op == GGML_OP_FLASH_ATTN_EXT;
}

static inline void ggml_cpu_flash_compat(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    GGML_ASSERT(params != NULL);
    GGML_ASSERT(ggml_cpu_flash_compat_supported(dst));
    ggml_compute_forward_flash_attn_ext(params, dst);
}

#ifdef __cplusplus
}
#endif
