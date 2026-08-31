#include "sage-attention.cuh"

#if defined(GGML_USE_HIP)

// ROCm implementation hook. Until the native gfx/MFMA Sage kernel is wired,
// report unsupported so the shared dispatcher selects the existing HIP
// attention kernels instead of referencing CUDA-only symbols.
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

// FlashAttention retention is an API-compatibility contract, not an automatic
// optimization choice. ROCm is the first non-CUDA backend intended to satisfy
// this contract. Volta/Xavier can use its legacy implementation when wired into
// this tree; Pascal will get a distinct compatibility implementation later.
bool ggml_flash_compat_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
#if defined(GGML_USE_HIP)
    // The existing HIP FlashAttention path remains the compatibility provider.
    // Shape/device validation stays in fattn.cu until that path is fully lifted
    // behind this interface.
    return true;
#else
    // CUDA compatibility remains available through the existing fattn path.
    // Pascal-specific validation/implementation will be added separately.
    return true;
#endif
}

void ggml_flash_compat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("FlashAttention compatibility dispatch has not yet been fully lifted from fattn.cu");
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
