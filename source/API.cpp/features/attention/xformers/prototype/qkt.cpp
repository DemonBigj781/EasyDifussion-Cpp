#include "xformers.hpp"

#include <cmath>

namespace xformers::prototype {
namespace {

std::size_t tensor_index(const Tensor4D & t,
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
    if (kv_heads <= 0 || q_heads <= 0) {
        return 0;
    }
    if (kv_heads == q_heads) {
        return q_head;
    }
    const std::int64_t group = q_heads / kv_heads;
    return group > 0 ? q_head / group : 0;
}

} // namespace

bool qkt(const AttentionRequest & request, ScoreBuffer & scores) {
    const auto & q = request.q;
    const auto & k = request.k;

    scores.batch = q.batch;
    scores.heads = q.heads;
    scores.query_tokens = q.tokens;
    scores.key_tokens = k.tokens;
    scores.values.assign(static_cast<std::size_t>(q.batch * q.heads * q.tokens * k.tokens), 0.0f);

    const float scale = request.scale > 0.0f
        ? request.scale
        : 1.0f / std::sqrt(static_cast<float>(q.head_dim));

    for (std::int64_t b = 0; b < q.batch; ++b) {
        for (std::int64_t h = 0; h < q.heads; ++h) {
            const std::int64_t kv_head = map_kv_head(h, q.heads, k.heads);
            for (std::int64_t qi = 0; qi < q.tokens; ++qi) {
                for (std::int64_t ki = 0; ki < k.tokens; ++ki) {
                    float dot = 0.0f;
                    for (std::int64_t d = 0; d < q.head_dim; ++d) {
                        dot += q.data[tensor_index(q, b, h, qi, d)] *
                               k.data[tensor_index(k, b, kv_head, ki, d)];
                    }
                    scores.values[score_index(scores, b, h, qi, ki)] = dot * scale;
                }
            }
        }
    }

    return true;
}

} // namespace xformers::prototype
