#pragma once

#include "common.cuh"

// GGML's application-side adapter for the normalized Flash Common route. It
// translates GGML tensors and parameters but never calls the CUDA translation
// or definition directly.
bool ggml_cuda_flash_common_attn_supported(
    int device, const ggml_tensor* dst);
void ggml_cuda_flash_common_attn(
    ggml_backend_cuda_context& context, ggml_tensor* dst);
uint64_t ggml_cuda_flash_common_attn_launch_count() noexcept;
