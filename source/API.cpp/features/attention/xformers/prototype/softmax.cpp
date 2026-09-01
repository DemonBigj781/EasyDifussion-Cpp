#include "xformers.hpp"

#include <cmath>
#include <limits>

namespace xformers::prototype {
namespace {

std::size_t score_index(const ScoreBuffer & s,
                        std::int64_t b,
                        std::int64_t h,
                        std::int64_t q,
                        std::int64_t k) {
    return static_cast<std::size_t>((((b * s.heads) + h) * s.query_tokens + q) * s.key_tokens + k);
}

} // namespace

bool softmax(ScoreBuffer & scores) {
    for (std::int64_t b = 0; b < scores.batch; ++b) {
        for (std::int64_t h = 0; h < scores.heads; ++h) {
            for (std::int64_t q = 0; q < scores.query_tokens; ++q) {
                float max_value = -std::numeric_limits<float>::infinity();
                for (std::int64_t k = 0; k < scores.key_tokens; ++k) {
                    const float v = scores.values[score_index(scores, b, h, q, k)];
                    if (v > max_value) {
                        max_value = v;
                    }
                }

                float sum = 0.0f;
                for (std::int64_t k = 0; k < scores.key_tokens; ++k) {
                    float & v = scores.values[score_index(scores, b, h, q, k)];
                    if (!std::isfinite(v) && v < 0.0f) {
                        v = 0.0f;
                        continue;
                    }
                    v = std::exp(v - max_value);
                    sum += v;
                }

                if (sum <= 0.0f || !std::isfinite(sum)) {
                    return false;
                }

                const float inv_sum = 1.0f / sum;
                for (std::int64_t k = 0; k < scores.key_tokens; ++k) {
                    scores.values[score_index(scores, b, h, q, k)] *= inv_sum;
                }
            }
        }
    }

    return true;
}

} // namespace xformers::prototype
