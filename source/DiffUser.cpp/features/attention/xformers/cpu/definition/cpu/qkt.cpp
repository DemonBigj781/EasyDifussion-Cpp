#include "xformers.hpp"

#include <cmath>
#include <cstddef>

namespace edcpp::api::attention::xformers::cpu::definition::cpu {
namespace {

std::size_t tensor_index(const NativeTensor4D& tensor, std::int64_t batch,
                         std::int64_t head, std::int64_t token,
                         std::int64_t dimension) {
    return static_cast<std::size_t>(
        (((batch * tensor.heads) + head) * tensor.tokens + token) *
            tensor.head_dim +
        dimension);
}

std::size_t score_index(const NativeScoreBuffer& scores, std::int64_t batch,
                        std::int64_t head, std::int64_t query,
                        std::int64_t key) {
    return static_cast<std::size_t>(
        (((batch * scores.heads) + head) * scores.query_tokens + query) *
            scores.key_tokens +
        key);
}

std::int64_t map_kv_head(
    std::int64_t query_head, std::int64_t query_heads,
    std::int64_t key_value_heads) {
    if (key_value_heads == query_heads) return query_head;
    const std::int64_t group = query_heads / key_value_heads;
    return group > 0 ? query_head / group : 0;
}

} // namespace

bool qkt(const NativeRequest& request, NativeScoreBuffer& scores) {
    if (request.dtype != NativeDType::f32) return false;

    const auto* query = static_cast<const float*>(request.q.data);
    const auto* key = static_cast<const float*>(request.k.data);
    if (query == nullptr || key == nullptr) return false;

    scores.batch = request.q.batch;
    scores.heads = request.q.heads;
    scores.query_tokens = request.q.tokens;
    scores.key_tokens = request.k.tokens;
    scores.values.assign(
        static_cast<std::size_t>(
            request.q.batch * request.q.heads * request.q.tokens *
            request.k.tokens),
        0.0f);
    scores.extra_denominator_logits.clear();

    const float scale = request.scale > 0.0f
        ? request.scale
        : 1.0f / std::sqrt(static_cast<float>(request.q.head_dim));

    for (std::int64_t batch = 0; batch < request.q.batch; ++batch) {
        for (std::int64_t head = 0; head < request.q.heads; ++head) {
            const std::int64_t key_head =
                map_kv_head(head, request.q.heads, request.k.heads);
            for (std::int64_t query_token = 0;
                 query_token < request.q.tokens; ++query_token) {
                for (std::int64_t key_token = 0;
                     key_token < request.k.tokens; ++key_token) {
                    float dot = 0.0f;
                    for (std::int64_t dimension = 0;
                         dimension < request.q.head_dim; ++dimension) {
                        dot += query[tensor_index(
                                   request.q, batch, head, query_token,
                                   dimension)] *
                               key[tensor_index(
                                   request.k, batch, key_head, key_token,
                                   dimension)];
                    }
                    scores.values[score_index(
                        scores, batch, head, query_token, key_token)] =
                        dot * scale;
                }
            }
        }
    }
    return true;
}

} // namespace edcpp::api::attention::xformers::cpu::definition::cpu
