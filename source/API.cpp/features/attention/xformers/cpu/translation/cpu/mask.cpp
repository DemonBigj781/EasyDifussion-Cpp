#include <cstddef>
#include <limits>

namespace easyapi::attention::xformers::cpu {

void apply_mask(float* scores, const float* additive_mask,
                std::size_t q_tokens, std::size_t kv_tokens,
                bool causal) {
    if (!scores) {
        return;
    }

    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (std::size_t qi = 0; qi < q_tokens; ++qi) {
        for (std::size_t ki = 0; ki < kv_tokens; ++ki) {
            const std::size_t index = qi * kv_tokens + ki;
            if (causal && ki > qi) {
                scores[index] = neg_inf;
                continue;
            }
            if (additive_mask) {
                scores[index] += additive_mask[index];
            }
        }
    }
}

} // namespace easyapi::attention::xformers::cpu
