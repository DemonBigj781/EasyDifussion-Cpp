#include "features/attention/xformers/common/xformers.hpp"

#include <cstddef>

namespace edcpp::api::attention::xformers::cpu::translation {
namespace {
std::size_t tensor_index(const Tensor4D& t, std::int64_t b, std::int64_t h,
                         std::int64_t token, std::int64_t d) {
    return static_cast<std::size_t>((((b * t.heads) + h) * t.tokens + token) * t.head_dim + d);
}

std::size_t mutable_tensor_index(const MutableTensor4D& t, std::int64_t b, std::int64_t h,
                                 std::int64_t token, std::int64_t d) {
    return static_cast<std::size_t>((((b * t.heads) + h) * t.tokens + token) * t.head_dim + d);
}

std::size_t score_index(const ScoreBuffer& s, std::int64_t b, std::int64_t h,
                        std::int64_t q, std::int64_t k) {
    return static_cast<std::size_t>((((b * s.heads) + h) * s.query_tokens + q) * s.key_tokens + k);
}

std::int64_t map_kv_head(std::int64_t q_head, std::int64_t q_heads, std::int64_t kv_heads) {
    if (kv_heads == q_heads) return q_head;
    const std::int64_t group = q_heads / kv_heads;
    return group > 0 ? q_head / group : 0;
}
}

bool av(const AttentionRequest& request, const ScoreBuffer& probabilities) {
    if (request.dtype != DType::f32) return false;

    const auto* v_data = static_cast<const float*>(request.v.data);
    auto* out_data = static_cast<float*>(request.out.data);
    if (v_data == nullptr || out_data == nullptr) return false;

    for (std::int64_t b = 0; b < request.q.batch; ++b) {
        for (std::int64_t h = 0; h < request.q.heads; ++h) {
            const std::int64_t v_head = map_kv_head(h, request.q.heads, request.v.heads);
            for (std::int64_t qi = 0; qi < request.q.tokens; ++qi) {
                for (std::int64_t d = 0; d < request.v.head_dim; ++d) {
                    float value = 0.0f;
                    for (std::int64_t ki = 0; ki < request.v.tokens; ++ki) {
                        value += probabilities.values[score_index(probabilities, b, h, qi, ki)] *
                                 v_data[tensor_index(request.v, b, v_head, ki, d)];
                    }
                    out_data[mutable_tensor_index(request.out, b, h, qi, d)] = value;
                }
            }
        }
    }
    return true;
}

} // namespace edcpp::api::attention::xformers::cpu::translation
