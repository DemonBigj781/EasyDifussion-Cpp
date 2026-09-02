#include <cuda_runtime.h>
#include <cstddef>

namespace easyapi::attention::xformers::cuda {

bool qkt(const float*, const float*, float*, std::size_t, std::size_t, std::size_t, float);
bool apply_mask(float*, const float*, std::size_t, std::size_t, bool);
bool softmax(float*, std::size_t, std::size_t);
bool av(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);

bool forward(const float* q, const float* k, const float* v,
             const float* additive_mask, float* output,
             std::size_t q_tokens, std::size_t kv_tokens,
             std::size_t head_dim, bool causal, float scale) {
    if (!q || !k || !v || !output || q_tokens == 0 ||
        kv_tokens == 0 || head_dim == 0) {
        return false;
    }

    float* scores = nullptr;
    const std::size_t score_bytes = q_tokens * kv_tokens * sizeof(float);
    if (cudaMalloc(&scores, score_bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const bool ok = qkt(q, k, scores, q_tokens, kv_tokens, head_dim, scale) &&
                    apply_mask(scores, additive_mask, q_tokens, kv_tokens, causal) &&
                    softmax(scores, q_tokens, kv_tokens) &&
                    av(scores, v, output, q_tokens, kv_tokens, head_dim) &&
                    cudaDeviceSynchronize() == cudaSuccess;

    const cudaError_t free_status = cudaFree(scores);
    if (free_status != cudaSuccess) {
        cudaGetLastError();
    }
    return ok && free_status == cudaSuccess;
}

} // namespace easyapi::attention::xformers::cuda
