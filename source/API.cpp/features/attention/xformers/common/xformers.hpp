#pragma once

#include "common/api.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edcpp::api::attention::xformers {

enum class DType : std::uint8_t {
    f32,
    f16,
    bf16,
};

struct Tensor4D {
    const void* data = nullptr;
    std::int64_t batch = 0;
    std::int64_t heads = 0;
    std::int64_t tokens = 0;
    std::int64_t head_dim = 0;
    DType dtype = DType::f32;
    // Byte strides in head_dim, token, head, batch order. All-zero means
    // contiguous storage.
    std::array<std::size_t, 4> byte_strides{};
};

struct MutableTensor4D {
    void* data = nullptr;
    std::int64_t batch = 0;
    std::int64_t heads = 0;
    std::int64_t tokens = 0;
    std::int64_t head_dim = 0;
    DType dtype = DType::f32;
    std::array<std::size_t, 4> byte_strides{};
};

struct MaskView {
    const void* data = nullptr;
    std::int64_t batch = 1;
    std::int64_t heads = 1;
    std::int64_t query_tokens = 1;
    std::int64_t key_tokens = 1;
    DType dtype = DType::f32;
    // Byte strides in key-token, query-token, head, batch order.
    std::array<std::size_t, 4> byte_strides{};
};

struct AlibiConfig {
    bool enabled = false;
    const float* slopes = nullptr;
    std::int64_t slope_count = 0;
    // Optional per-head mask slope. When positive, the translation scales mask
    // values per head instead of consuming an explicit slopes buffer.
    float max_bias = 0.0f;
};

struct AttentionSinkConfig {
    bool enabled = false;
    const float* values = nullptr;
    std::int64_t value_count = 0;
};

struct ExecutionContext {
    // Backend translations interpret this opaque handle. CUDA uses it as a
    // cudaStream_t; Common never exposes that native type.
    void* stream = nullptr;
    bool synchronize = true;
    int device_index = -1;
};

struct AttentionRequest {
    Tensor4D q{};
    Tensor4D k{};
    Tensor4D v{};
    MutableTensor4D out{};
    DType dtype = DType::f32;
    float scale = 0.0f;
    float softcap = 0.0f;
    bool causal = false;
    MaskView mask{};
    AlibiConfig alibi{};
    AttentionSinkConfig sinks{};
    ExecutionContext execution{};
};

struct Capabilities {
    bool forward = false;
    bool qkt = false;
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

struct ValidationResult {
    bool ok = false;
    std::string message{};
};

struct ScoreBuffer {
    std::vector<float> values{};
    std::int64_t batch = 0;
    std::int64_t heads = 0;
    std::int64_t query_tokens = 0;
    std::int64_t key_tokens = 0;
    std::vector<float> extra_denominator_logits{};
};

using ValidateFn = ValidationResult (*)(const AttentionRequest&);
using QktFn = bool (*)(const AttentionRequest&, ScoreBuffer&);
using MaskFn = bool (*)(const AttentionRequest&, ScoreBuffer&);
using SoftmaxFn = bool (*)(const AttentionRequest&, ScoreBuffer&);
using AvFn = bool (*)(const AttentionRequest&, const ScoreBuffer&);
using ForwardFn = bool (*)(const AttentionRequest&);

struct Translation {
    Backend backend = Backend::none;
    const char* name = "unknown";
    Capabilities capabilities{};
    ValidateFn validate = nullptr;
    QktFn qkt = nullptr;
    MaskFn mask = nullptr;
    SoftmaxFn softmax = nullptr;
    AvFn av = nullptr;
    ForwardFn forward = nullptr;
};

bool register_translation(const Translation* translation) noexcept;
const Translation* translation_for(Backend backend) noexcept;
Capabilities capabilities(Backend backend) noexcept;
ValidationResult validate(Backend backend, const AttentionRequest& request);
bool qkt(Backend backend, const AttentionRequest& request, ScoreBuffer& scores);
bool apply_mask_and_bias(Backend backend, const AttentionRequest& request, ScoreBuffer& scores);
bool softmax(Backend backend, const AttentionRequest& request, ScoreBuffer& scores);
bool av(Backend backend, const AttentionRequest& request, const ScoreBuffer& probabilities);
bool forward(Backend backend, const AttentionRequest& request);

} // namespace edcpp::api::attention::xformers
