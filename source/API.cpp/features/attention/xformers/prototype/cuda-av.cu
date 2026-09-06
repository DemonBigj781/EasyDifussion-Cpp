#include <cuda_runtime.h>
#include <cstddef>

namespace easyapi::attention::xformers::cuda {

__global__ void av_kernel(const float* weights, const float* v, float* output,
                          std::size_t q_tokens, std::size_t kv_tokens,
                          std::size_t head_dim) {
    const std::size_t qi = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= q_tokens || d >= head_dim) {
        return;
    }

    float sum = 0.0f;
    for (std::size_t ki = 0; ki < kv_tokens; ++ki) {
        sum += weights[qi * kv_tokens + ki] * v[ki * head_dim + d];
    }
    output[qi * head_dim + d] = sum;
}

bool av(const float* weights, const float* v, float* output,
        std::size_t q_tokens, std::size_t kv_tokens,
        std::size_t head_dim) {
    if (!weights || !v || !output || q_tokens == 0 || kv_tokens == 0 || head_dim == 0) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid((head_dim + block.x - 1) / block.x,
                    (q_tokens + block.y - 1) / block.y);
    av_kernel<<<grid, block>>>(weights, v, output, q_tokens, kv_tokens, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace easyapi::attention::xformers::cuda
