#pragma once

#include "../../../../flash/cuda/translation/gpu/common.cuh"

// Native inference-only memory-efficient attention forward path. The API is
// intentionally ggml-native: no PyTorch/ATen dependency and no backward pass.
// It implements the complete GGML_OP_FLASH_ATTN_EXT forward contract for
// unquantized CUDA tensors, including broadcasted GQA/MQA inputs, additive
// masks/ALiBi, logit soft-capping, attention sinks, and independent K/V head
// dimensions. Unsupported layouts return false so the normal ggml attention
// dispatcher can select another implementation without changing semantics.
bool ggml_cuda_xformers_attn_supported(int device, const ggml_tensor * dst);
void ggml_cuda_xformers_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
