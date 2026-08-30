/*
 * Native ggml integration for the SageAttention SM80 INT8-QK/FP16-PV kernel.
 *
 * The underlying attention kernel and helper headers are Copyright (c) 2024
 * by the SageAttention team and licensed under Apache-2.0.
 */

#pragma once

#include "../common.cuh"

bool ggml_cuda_sage_attn_sm80_supported(int device, const ggml_tensor * dst);
void ggml_cuda_sage_attn_sm80(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
