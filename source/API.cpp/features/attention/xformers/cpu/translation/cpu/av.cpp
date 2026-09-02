#include <cstddef>

namespace easyapi::attention::xformers::cpu {

void av(const float* weights, const float* v, float* output,
        std::size_t q_tokens, std::size_t kv_tokens,
        std::size_t head_dim) {
    if (!weights || !v || !output || head_dim == 0) {
        return;
    }

    for (std::size_t qi = 0; qi < q_tokens; ++qi) {
        float* out_row = output + qi * head_dim;
        for (std::size_t d = 0; d < head_dim; ++d) {
            out_row[d] = 0.0f;
        }
        for (std::size_t ki = 0; ki < kv_tokens; ++ki) {
            const float weight = weights[qi * kv_tokens + ki];
            const float* v_row = v + ki * head_dim;
            for (std::size_t d = 0; d < head_dim; ++d) {
                out_row[d] += weight * v_row[d];
            }
        }
    }
}

} // namespace easyapi::attention::xformers::cpu
