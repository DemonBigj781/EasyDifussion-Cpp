#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace edcpp::api::attention::xformers::cpu::definition::cpu {

enum class NativeDType : std::uint8_t {
    f32,
    f16,
    bf16,
};

struct NativeTensor4D {
    const void* data = nullptr;
    std::int64_t batch = 0;
    std::int64_t heads = 0;
    std::int64_t tokens = 0;
    std::int64_t head_dim = 0;
    NativeDType dtype = NativeDType::f32;
    std::array<std::size_t, 4> byte_strides{};
};

struct NativeMutableTensor4D {
    void* data = nullptr;
    std::int64_t batch = 0;
    std::int64_t heads = 0;
    std::int64_t tokens = 0;
    std::int64_t head_dim = 0;
    NativeDType dtype = NativeDType::f32;
    std::array<std::size_t, 4> byte_strides{};
};

struct NativeMaskView {
    const void* data = nullptr;
    std::int64_t batch = 1;
    std::int64_t heads = 1;
    std::int64_t query_tokens = 1;
    std::int64_t key_tokens = 1;
    NativeDType dtype = NativeDType::f32;
    std::array<std::size_t, 4> byte_strides{};
};

struct NativeAlibiConfig {
    bool enabled = false;
    const float* slopes = nullptr;
    std::int64_t slope_count = 0;
    float max_bias = 0.0f;
};

struct NativeAttentionSinkConfig {
    bool enabled = false;
    const float* values = nullptr;
    std::int64_t value_count = 0;
};

struct NativeRequest {
    NativeTensor4D q{};
    NativeTensor4D k{};
    NativeTensor4D v{};
    NativeMutableTensor4D out{};
    NativeDType dtype = NativeDType::f32;
    float scale = 0.0f;
    float softcap = 0.0f;
    bool causal = false;
    NativeMaskView mask{};
    NativeAlibiConfig alibi{};
    NativeAttentionSinkConfig sinks{};
};

struct NativeCapabilities {
    bool forward = false;
    bool separate_stages = false;
    bool additive_mask = false;
    bool causal_mask = false;
    bool alibi = false;
    bool softcap = false;
    bool attention_sinks = false;
    bool gqa = false;
    bool mqa = false;
    bool f32 = false;
    bool f16 = false;
    bool bf16 = false;
};

struct NativeResult {
    bool ok = false;
    const char* message = nullptr;
};

struct NativeScoreBuffer {
    std::vector<float> values{};
    std::int64_t batch = 0;
    std::int64_t heads = 0;
    std::int64_t query_tokens = 0;
    std::int64_t key_tokens = 0;
    std::vector<float> extra_denominator_logits{};
};

NativeCapabilities capabilities() noexcept;
NativeResult validate(const NativeRequest& request) noexcept;
bool qkt(const NativeRequest& request, NativeScoreBuffer& scores);
bool apply_mask_and_bias(const NativeRequest& request, NativeScoreBuffer& scores);
bool softmax(const NativeRequest& request, NativeScoreBuffer& scores);
bool av(const NativeRequest& request, const NativeScoreBuffer& probabilities);
bool forward(const NativeRequest& request);

} // namespace edcpp::api::attention::xformers::cpu::definition::cpu
