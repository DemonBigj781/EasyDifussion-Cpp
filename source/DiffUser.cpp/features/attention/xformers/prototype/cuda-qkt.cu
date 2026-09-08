#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

namespace easyapi::attention::xformers::cuda {

__global__ void qkt_kernel(const float* q, const float* k, float* scores,
                           std::size_t q_tokens, std::size_t kv_tokens,
                           std::size_t head_dim, float scale) {
    const std::size_t qi = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t ki = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= q_tokens || ki >= kv_tokens) {
        return;
    }

    float dot = 0.0f;
    const float* q_row = q + qi * head_dim;
    const float* k_row = k + ki * head_dim;
    for (std::size_t d = 0; d < head_dim; ++d) {
        dot += q_row[d] * k_row[d];
    }
    scores[qi * kv_tokens + ki] = dot * scale;
}

bool qkt(const float* q, const float* k, float* scores,
         std::size_t q_tokens, std::size_t kv_tokens,
         std::size_t head_dim, float scale) {
    if (!q || !k || !scores || q_tokens == 0 || kv_tokens == 0 || head_dim == 0) {
        return false;
    }
    const float applied_scale = scale == 0.0f
        ? 1.0f / std::sqrt(static_cast<float>(head_dim))
        : scale;
    const dim3 block(16, 16);
    const dim3 grid((kv_tokens + block.x - 1) / block.x,
                    (q_tokens + block.y - 1) / block.y);
    qkt_kernel<<<grid, block>>>(q, k, scores, q_tokens, kv_tokens, head_dim, applied_scale);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace easyapi::attention::xformers::cuda
