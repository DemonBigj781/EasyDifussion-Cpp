#include "xformers.hpp"

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

std::size_t tensor_index(
    const NativeMutableTensor4D& tensor, std::int64_t batch,
    std::int64_t head, std::int64_t token, std::int64_t dimension) {
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

bool av(
    const NativeRequest& request,
    const NativeScoreBuffer& probabilities) {
    if (request.dtype != NativeDType::f32) return false;

    const auto* value = static_cast<const float*>(request.v.data);
    auto* output = static_cast<float*>(request.out.data);
    if (value == nullptr || output == nullptr) return false;

    for (std::int64_t batch = 0; batch < request.q.batch; ++batch) {
        for (std::int64_t head = 0; head < request.q.heads; ++head) {
            const std::int64_t value_head =
                map_kv_head(head, request.q.heads, request.v.heads);
            for (std::int64_t query = 0;
                 query < request.q.tokens; ++query) {
                for (std::int64_t dimension = 0;
                     dimension < request.v.head_dim; ++dimension) {
                    float result = 0.0f;
                    for (std::int64_t key = 0;
                         key < request.v.tokens; ++key) {
                        result += probabilities.values[score_index(
                                      probabilities, batch, head, query,
                                      key)] *
                                  value[tensor_index(
                                      request.v, batch, value_head, key,
                                      dimension)];
                    }
                    output[tensor_index(
                        request.out, batch, head, query, dimension)] =
                        result;
                }
            }
        }
    }
    return true;
}

} // namespace edcpp::api::attention::xformers::cpu::definition::cpu
