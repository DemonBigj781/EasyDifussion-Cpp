#include <algorithm>
#include <cstddef>
#include <cmath>

namespace easyapi::attention::xformers::cpu {

void qkt(const float* q, const float* k, float* scores,
         std::size_t q_tokens, std::size_t kv_tokens,
         std::size_t head_dim, float scale) {
    if (!q || !k || !scores || head_dim == 0) {
        return;
    }

    const float applied_scale = scale == 0.0f
        ? 1.0f / std::sqrt(static_cast<float>(head_dim))
        : scale;

    for (std::size_t qi = 0; qi < q_tokens; ++qi) {
        const float* q_row = q + qi * head_dim;
        for (std::size_t ki = 0; ki < kv_tokens; ++ki) {
            const float* k_row = k + ki * head_dim;
            float dot = 0.0f;
            for (std::size_t d = 0; d < head_dim; ++d) {
                dot += q_row[d] * k_row[d];
            }
            scores[qi * kv_tokens + ki] = dot * applied_scale;
        }
    }
}

} // namespace easyapi::attention::xformers::cpu
