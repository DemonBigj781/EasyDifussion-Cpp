#include "sage-attention-sm80.cuh"

bool ggml_cuda_sage_attn_sm80_supported(int, const ggml_tensor *) {
    return false;
}

void ggml_cuda_sage_attn_sm80(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("SageAttention was not compiled for this CUDA target");
}
