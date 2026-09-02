#include <cstddef>
#include <vector>

namespace easyapi::attention::xformers::cpu {

void qkt(const float*, const float*, float*, std::size_t, std::size_t, std::size_t, float);
void apply_mask(float*, const float*, std::size_t, std::size_t, bool);
void softmax(float*, std::size_t, std::size_t);
void av(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);

bool forward(const float* q, const float* k, const float* v,
             const float* additive_mask, float* output,
             std::size_t q_tokens, std::size_t kv_tokens,
             std::size_t head_dim, bool causal, float scale) {
    if (!q || !k || !v || !output || q_tokens == 0 ||
        kv_tokens == 0 || head_dim == 0) {
        return false;
    }

    std::vector<float> scores(q_tokens * kv_tokens);
    qkt(q, k, scores.data(), q_tokens, kv_tokens, head_dim, scale);
    apply_mask(scores.data(), additive_mask, q_tokens, kv_tokens, causal);
    softmax(scores.data(), q_tokens, kv_tokens);
    av(scores.data(), v, output, q_tokens, kv_tokens, head_dim);
    return true;
}

} // namespace easyapi::attention::xformers::cpu
