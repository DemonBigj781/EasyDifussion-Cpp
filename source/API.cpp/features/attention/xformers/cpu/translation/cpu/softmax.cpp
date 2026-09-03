#include "features/attention/xformers/common/xformers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace edcpp::api::attention::xformers::cpu::translation {

bool softmax(const AttentionRequest&, ScoreBuffer& scores) {
    if (scores.key_tokens <= 0 || scores.query_tokens <= 0 || scores.heads <= 0 || scores.batch <= 0) {
        return false;
    }

    const std::size_t cols = static_cast<std::size_t>(scores.key_tokens);
    const std::size_t rows = static_cast<std::size_t>(scores.batch * scores.heads * scores.query_tokens);
    if (scores.values.size() != rows * cols) return false;

    for (std::size_t row = 0; row < rows; ++row) {
        float* values = scores.values.data() + row * cols;
        float max_value = -std::numeric_limits<float>::infinity();
        for (std::size_t col = 0; col < cols; ++col) {
            max_value = std::max(max_value, values[col]);
        }

        if (max_value == -std::numeric_limits<float>::infinity()) {
            std::fill(values, values + cols, 0.0f);
            continue;
        }

        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            values[col] = std::exp(values[col] - max_value);
            sum += values[col];
        }

        if (sum == 0.0f || !std::isfinite(sum)) {
            std::fill(values, values + cols, 0.0f);
            continue;
        }

        const float inv_sum = 1.0f / sum;
        for (std::size_t col = 0; col < cols; ++col) {
            values[col] *= inv_sum;
        }
    }
    return true;
}

} // namespace edcpp::api::attention::xformers::cpu::translation
