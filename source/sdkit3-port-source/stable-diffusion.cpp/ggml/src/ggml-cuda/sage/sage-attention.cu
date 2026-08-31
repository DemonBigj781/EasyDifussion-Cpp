#include "sage-attention.cuh"

#if defined(GGML_USE_HIP)

// ROCm implementation hook. Until the native gfx/MFMA Sage kernel is wired,
// report unsupported so the shared FlashAttention dispatcher selects the
// existing HIP attention kernels instead of referencing CUDA-only symbols.
bool ggml_sage_attn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

void ggml_sage_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("ROCm SageAttention implementation is not available for this target");
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

// Native xFormers-style kernels are not wired into this ggml tree yet. Keep a
// stable backend-neutral ABI now so CUDA, HIP/ROCm, Volta/Xavier, and future
// implementations can provide their own kernels without changing callers.
bool ggml_xformers_attn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

void ggml_xformers_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("xFormers attention implementation is not available for this backend");
}

bool ggml_attention_impl_supported(ggml_attention_impl impl, int device, const ggml_tensor * dst) {
    switch (impl) {
        case GGML_ATTN_IMPL_SAGE:
            return ggml_sage_attn_supported(device, dst);
        case GGML_ATTN_IMPL_XFORMERS:
            return ggml_xformers_attn_supported(device, dst);
        case GGML_ATTN_IMPL_FLASH:
            // FlashAttention is currently selected by the existing fattn
            // capability logic. It can be moved fully behind this API once its
            // backend-specific entry point is separated from fattn.cu.
            return false;
        case GGML_ATTN_IMPL_NONE:
        default:
            return false;
    }
}
