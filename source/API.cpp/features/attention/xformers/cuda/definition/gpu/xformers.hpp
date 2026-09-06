#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace edcpp::api::attention::xformers::cuda::definition::gpu {

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

struct NativeExecutionContext {
    void* stream = nullptr;
    bool synchronize = true;
    int device_index = -1;
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
    NativeExecutionContext execution{};
};

struct NativeCapabilities {
    bool fused_forward = false;
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
    std::int64_t maximum_value_dimension = 0;
    int minimum_compute_major = 0;
};

struct NativeValidationResult {
    bool ok = false;
    const char* message = nullptr;
};

NativeCapabilities capabilities() noexcept;
NativeValidationResult validate_qkt(const NativeRequest& request) noexcept;
NativeValidationResult validate_mask(const NativeRequest& request) noexcept;
NativeValidationResult validate_softmax(const NativeRequest& request) noexcept;
NativeValidationResult validate_av(const NativeRequest& request) noexcept;
NativeValidationResult validate_forward(const NativeRequest& request) noexcept;

} // namespace edcpp::api::attention::xformers::cuda::definition::gpu
