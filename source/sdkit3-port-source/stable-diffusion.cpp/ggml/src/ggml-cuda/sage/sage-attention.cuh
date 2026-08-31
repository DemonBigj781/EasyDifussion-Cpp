#pragma once

#include "../common.cuh"

// Backend-neutral SageAttention interface used by the shared CUDA/HIP
// FlashAttention dispatcher. Backend-specific implementations stay behind
// this API so fattn.cu never needs to know about SM or gfx kernel names.
bool ggml_sage_attn_supported(int device, const ggml_tensor * dst);
void ggml_sage_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
