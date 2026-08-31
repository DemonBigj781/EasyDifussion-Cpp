#pragma once

#include "../../../../sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda/common.cuh"

// Native inference-only memory-efficient attention forward path. The API is
// intentionally ggml-native: no PyTorch/ATen dependency and no backward pass.
bool ggml_cuda_xformers_attn_supported(int device, const ggml_tensor * dst);
void ggml_cuda_xformers_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
