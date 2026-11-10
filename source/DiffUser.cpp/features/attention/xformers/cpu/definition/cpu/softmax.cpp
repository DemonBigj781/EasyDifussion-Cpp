#include "xformers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace edcpp::api::attention::xformers::cpu::definition::cpu {

bool softmax(const NativeRequest&, NativeScoreBuffer& scores) {
    if (scores.key_tokens <= 0 || scores.query_tokens <= 0 ||
        scores.heads <= 0 || scores.batch <= 0) {
        return false;
    }

    const std::size_t columns =
        static_cast<std::size_t>(scores.key_tokens);
    const std::size_t rows = static_cast<std::size_t>(
        scores.batch * scores.heads * scores.query_tokens);
    if (scores.values.size() != rows * columns) return false;

    for (std::size_t row = 0; row < rows; ++row) {
        float* values = scores.values.data() + row * columns;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t column = 0; column < columns; ++column) {
            maximum = std::max(maximum, values[column]);
        }

        if (maximum == -std::numeric_limits<float>::infinity()) {
            std::fill(values, values + columns, 0.0f);
            continue;
        }

        float sum = 0.0f;
        for (std::size_t column = 0; column < columns; ++column) {
            values[column] = std::exp(values[column] - maximum);
            sum += values[column];
        }
        if (sum == 0.0f || !std::isfinite(sum)) {
            std::fill(values, values + columns, 0.0f);
            continue;
        }

        const float inverse_sum = 1.0f / sum;
        for (std::size_t column = 0; column < columns; ++column) {
            values[column] *= inverse_sum;
        }
    }
    return true;
}

} // namespace edcpp::api::attention::xformers::cpu::definition::cpu
