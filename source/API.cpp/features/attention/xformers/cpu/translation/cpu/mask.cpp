#include "features/attention/xformers/common/xformers.hpp"

#include <cmath>
#include <limits>

namespace edcpp::api::attention::xformers::cpu::translation {
namespace {
std::size_t score_index(const ScoreBuffer& s, std::int64_t b, std::int64_t h,
                        std::int64_t q, std::int64_t k) {
    return static_cast<std::size_t>((((b * s.heads) + h) * s.query_tokens + q) * s.key_tokens + k);
}
std::size_t mask_index(const MaskView& m, std::int64_t b, std::int64_t h,
                       std::int64_t q, std::int64_t k) {
    const auto mb = m.batch == 1 ? 0 : b;
    const auto mh = m.heads == 1 ? 0 : h;
    const auto mq = m.query_tokens == 1 ? 0 : q;
    const auto mk = m.key_tokens == 1 ? 0 : k;
    return static_cast<std::size_t>((((mb * m.heads) + mh) * m.query_tokens + mq) * m.key_tokens + mk);
}
}

bool apply_mask_and_bias(const AttentionRequest& request, ScoreBuffer& scores) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (std::int64_t b = 0; b < scores.batch; ++b) {
        for (std::int64_t h = 0; h < scores.heads; ++h) {
            const float alibi_slope = request.alibi.enabled ? request.alibi.slopes[h] : 0.0f;
            for (std::int64_t q = 0; q < scores.query_tokens; ++q) {
                for (std::int64_t k = 0; k < scores.key_tokens; ++k) {
                    float& value = scores.values[score_index(scores, b, h, q, k)];
                    if (request.causal && k > q) {
                        value = neg_inf;
                        continue;
                    }
                    if (request.mask.data) {
                        value += request.mask.data[mask_index(request.mask, b, h, q, k)];
                    }
                    if (request.alibi.enabled) {
                        value += alibi_slope * static_cast<float>(k - q);
                    }
                    if (request.softcap > 0.0f && std::isfinite(value)) {
                        value = request.softcap * std::tanh(value / request.softcap);
                    }
                }
            }
        }
    }
    return true;
}

} // namespace edcpp::api::attention::xformers::cpu::translation
