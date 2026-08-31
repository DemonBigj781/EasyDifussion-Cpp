#include "sage-attention.cuh"
#include "../flash/fattn.cuh"
#include "../xformers/xformers-attention.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

// Non-CUDA implementation hook. Until native ROCm/MUSA Sage kernels are wired,
// report unsupported so the shared dispatcher selects an existing attention
// path instead of referencing CUDA-only symbols.
bool ggml_sage_attn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

void ggml_sage_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("SageAttention is not available for this target");
}

#else

#include "sage-attention-sm80.cuh"

bool ggml_sage_attn_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_sage_attn_sm80_supported(device, dst);
}

void ggml_sage_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_sage_attn_sm80(ctx, dst);
}

#endif

bool ggml_xformers_attn_supported(int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    return ggml_cuda_xformers_attn_supported(device, dst);
#endif
}

void ggml_xformers_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("xFormers attention is not available for this target");
#else
    GGML_ASSERT(ggml_xformers_attn_supported(ctx.device, dst));
    ggml_cuda_xformers_attn(ctx, dst);
#endif
}

// FlashAttention retention is an API-compatibility contract, not an automatic
// optimization choice. The actual CUDA/HIP kernels remain in fattn.cu; this
// common API owns backend/architecture policy and delegates execution to them.
bool ggml_flash_compat_supported(int device, const ggml_tensor * dst) {
    if (dst == nullptr) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[device].cc;

#if defined(GGML_USE_HIP)
    // ROCm/HIP uses the shared fattn implementation. Keep architecture-specific
    // kernel selection inside fattn.cu (MFMA/WMMA/tile/vector) and expose only
    // the compatibility contract here.
    if (!GGML_CUDA_CC_IS_AMD(cc)) {
        return false;
    }
    return ggml_cuda_flash_attn_ext_supported(device, dst);
#else
    // Pascal gets a dedicated opcode/native implementation later. Do not allow
    // the generic compatibility API to silently treat Pascal as Volta+.
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_VOLTA) {
        return false;
    }
    return ggml_cuda_flash_attn_ext_supported(device, dst);
#endif
}

void ggml_flash_compat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_flash_compat_supported(ctx.device, dst));
    ggml_cuda_flash_attn_ext(ctx, dst);
}

bool ggml_attention_impl_supported(ggml_attention_impl impl, int device, const ggml_tensor * dst) {
    switch (impl) {
        case GGML_ATTN_IMPL_SAGE:
            return ggml_sage_attn_supported(device, dst);
        case GGML_ATTN_IMPL_XFORMERS:
            return ggml_xformers_attn_supported(device, dst);
        case GGML_ATTN_IMPL_NONE:
        default:
            return false;
    }
}
