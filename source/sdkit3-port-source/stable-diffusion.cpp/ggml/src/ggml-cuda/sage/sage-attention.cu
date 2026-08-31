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
