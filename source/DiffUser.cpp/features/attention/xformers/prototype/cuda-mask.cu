#include <cuda_runtime.h>
#include <cstddef>
#include <limits>

namespace easyapi::attention::xformers::cuda {

__global__ void mask_kernel(float* scores, const float* additive_mask,
                            std::size_t q_tokens, std::size_t kv_tokens,
                            bool causal) {
    const std::size_t qi = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t ki = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= q_tokens || ki >= kv_tokens) {
        return;
    }
    const std::size_t index = qi * kv_tokens + ki;
    if (causal && ki > qi) {
        scores[index] = -CUDART_INF_F;
        return;
    }
    if (additive_mask) {
        scores[index] += additive_mask[index];
    }
}

bool apply_mask(float* scores, const float* additive_mask,
                std::size_t q_tokens, std::size_t kv_tokens,
                bool causal) {
    if (!scores || q_tokens == 0 || kv_tokens == 0) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid((kv_tokens + block.x - 1) / block.x,
                    (q_tokens + block.y - 1) / block.y);
    mask_kernel<<<grid, block>>>(scores, additive_mask, q_tokens, kv_tokens, causal);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace easyapi::attention::xformers::cuda
