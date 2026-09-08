#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

namespace easyapi::attention::xformers::cuda {

__global__ void softmax_kernel(float* scores, std::size_t rows, std::size_t cols) {
    const std::size_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) {
        return;
    }

    float* values = scores + row * cols;
    float max_value = -CUDART_INF_F;
    for (std::size_t col = 0; col < cols; ++col) {
        max_value = fmaxf(max_value, values[col]);
    }

    float sum = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
        values[col] = expf(values[col] - max_value);
        sum += values[col];
    }
    if (!(sum > 0.0f) || !isfinite(sum)) {
        for (std::size_t col = 0; col < cols; ++col) {
            values[col] = 0.0f;
        }
        return;
    }
    const float inv_sum = 1.0f / sum;
    for (std::size_t col = 0; col < cols; ++col) {
        values[col] *= inv_sum;
    }
}

bool softmax(float* scores, std::size_t rows, std::size_t cols) {
    if (!scores || rows == 0 || cols == 0) {
        return false;
    }
    softmax_kernel<<<static_cast<unsigned int>(rows), 1>>>(scores, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace easyapi::attention::xformers::cuda
