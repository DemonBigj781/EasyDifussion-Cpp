#include "xformers.hpp"

#include <cmath>
#include <limits>

namespace edcpp::api::attention::xformers::cpu::definition::cpu {
namespace {

std::size_t score_index(const NativeScoreBuffer& scores, std::int64_t batch,
                        std::int64_t head, std::int64_t query,
                        std::int64_t key) {
    return static_cast<std::size_t>(
        (((batch * scores.heads) + head) * scores.query_tokens + query) *
            scores.key_tokens +
        key);
}

std::size_t mask_index(const NativeMaskView& mask, std::int64_t batch,
                       std::int64_t head, std::int64_t query,
                       std::int64_t key) {
    const auto mask_batch = mask.batch == 1 ? 0 : batch;
    const auto mask_head = mask.heads == 1 ? 0 : head;
    const auto mask_query = mask.query_tokens == 1 ? 0 : query;
    const auto mask_key = mask.key_tokens == 1 ? 0 : key;
    return static_cast<std::size_t>(
        (((mask_batch * mask.heads) + mask_head) * mask.query_tokens +
          mask_query) *
            mask.key_tokens +
        mask_key);
}

} // namespace

bool apply_mask_and_bias(
    const NativeRequest& request, NativeScoreBuffer& scores) {
    const float negative_infinity =
        -std::numeric_limits<float>::infinity();
    const auto* mask = static_cast<const float*>(request.mask.data);
    for (std::int64_t batch = 0; batch < scores.batch; ++batch) {
        for (std::int64_t head = 0; head < scores.heads; ++head) {
            const float slope =
                request.alibi.enabled ? request.alibi.slopes[head] : 0.0f;
            for (std::int64_t query = 0;
                 query < scores.query_tokens; ++query) {
                for (std::int64_t key = 0;
                     key < scores.key_tokens; ++key) {
                    float& value =
                        scores.values[score_index(
                            scores, batch, head, query, key)];
                    if (request.softcap > 0.0f && std::isfinite(value)) {
                        value = request.softcap *
                            std::tanh(value / request.softcap);
                    }
                    if (request.causal && key > query) {
                        value = negative_infinity;
                        continue;
                    }
                    if (mask != nullptr) {
                        value += mask[mask_index(
                            request.mask, batch, head, query, key)];
                    }
                    if (request.alibi.enabled) {
                        value += slope * static_cast<float>(key - query);
                    }
                }
            }
        }
    }
    return true;
}

} // namespace edcpp::api::attention::xformers::cpu::definition::cpu
