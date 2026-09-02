/*
 * Native ggml integration for the SageAttention SM80 INT8-QK/FP16-PV kernel.
 *
 * The underlying attention kernel and helper headers are Copyright (c) 2024
 * by the SageAttention team and licensed under Apache-2.0.
 */

#pragma once

#include "../../../../flash/cuda/translation/gpu/common.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#include "sage-attention.cuh"

// fattn.cu is shared by CUDA, HIP, and MUSA. Keep the legacy symbol spelling
// as a compatibility alias at this boundary, but route non-CUDA targets
// through the neutral SageAttention API. New code should call
// ggml_sage_attn_* directly.
#define ggml_cuda_sage_attn_sm80_supported ggml_sage_attn_supported
#define ggml_cuda_sage_attn_sm80           ggml_sage_attn
#else
bool ggml_cuda_sage_attn_sm80_supported(int device, const ggml_tensor * dst);
void ggml_cuda_sage_attn_sm80(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
