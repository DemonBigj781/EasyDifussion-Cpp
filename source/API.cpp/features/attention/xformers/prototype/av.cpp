#include "xformers.hpp"

#include <algorithm>

namespace xformers::prototype {
namespace {

std::size_t tensor_index(const Tensor4D & t,
                         std::int64_t b,
                         std::int64_t h,
                         std::int64_t token,
                         std::int64_t d) {
    return static_cast<std::size_t>((((b * t.heads) + h) * t.tokens + token) * t.head_dim + d);
}

std::size_t mutable_index(const MutableTensor4D & t,
                          std::int64_t b,
                          std::int64_t h,
                          std::int64_t token,
                          std::int64_t d) {
    return static_cast<std::size_t>((((b * t.heads) + h) * t.tokens + token) * t.head_dim + d);
}

std::size_t score_index(const ScoreBuffer & s,
                        std::int64_t b,
                        std::int64_t h,
                        std::int64_t q,
                        std::int64_t k) {
    return static_cast<std::size_t>((((b * s.heads) + h) * s.query_tokens + q) * s.key_tokens + k);
}

std::int64_t map_kv_head(std::int64_t q_head,
                         std::int64_t q_heads,
                         std::int64_t kv_heads) {
    if (kv_heads == q_heads) {
        return q_head;
    }
    const std::int64_t group = q_heads / kv_heads;
    return group > 0 ? q_head / group : 0;
}

} // namespace

bool av(const AttentionRequest & request, const ScoreBuffer & probabilities) {
    const auto & v = request.v;
    auto & out = const_cast<MutableTensor4D &>(request.out);

    std::fill(out.data,
              out.data + static_cast<std::size_t>(out.batch * out.heads * out.tokens * out.head_dim),
              0.0f);

    for (std::int64_t b = 0; b < probabilities.batch; ++b) {
        for (std::int64_t h = 0; h < probabilities.heads; ++h) {
            const std::int64_t kv_head = map_kv_head(h, probabilities.heads, v.heads);
            for (std::int64_t q = 0; q < probabilities.query_tokens; ++q) {
                for (std::int64_t k = 0; k < probabilities.key_tokens; ++k) {
                    const float weight = probabilities.values[score_index(probabilities, b, h, q, k)];
                    for (std::int64_t d = 0; d < v.head_dim; ++d) {
                        out.data[mutable_index(out, b, h, q, d)] +=
                            weight * v.data[tensor_index(v, b, kv_head, k, d)];
                    }
                }
            }
        }
    }

    return true;
}

} // namespace xformers::prototype
