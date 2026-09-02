#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace easyapi::attention::xformers::cpu {

void softmax(float* scores, std::size_t rows, std::size_t cols) {
    if (!scores || cols == 0) {
        return;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        float* values = scores + row * cols;
        float max_value = -std::numeric_limits<float>::infinity();
        for (std::size_t col = 0; col < cols; ++col) {
            max_value = std::max(max_value, values[col]);
        }

        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            values[col] = std::exp(values[col] - max_value);
            sum += values[col];
        }

        if (sum == 0.0f || !std::isfinite(sum)) {
            for (std::size_t col = 0; col < cols; ++col) {
                values[col] = 0.0f;
            }
            continue;
        }

        const float inv_sum = 1.0f / sum;
        for (std::size_t col = 0; col < cols; ++col) {
            values[col] *= inv_sum;
        }
    }
}

} // namespace easyapi::attention::xformers::cpu
