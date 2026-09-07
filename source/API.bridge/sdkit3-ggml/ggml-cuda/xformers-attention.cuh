#pragma once

#include "common.cuh"

// GGML's application-side adapter. It converts GGML tensors into the
// backend-neutral xFormers Common request; it never calls a backend
// translation or definition directly.
bool ggml_cuda_xformers_attn_supported(int device, const ggml_tensor* dst);
void ggml_cuda_xformers_attn(
    ggml_backend_cuda_context& context, ggml_tensor* dst);
uint64_t ggml_cuda_xformers_attn_launch_count() noexcept;
