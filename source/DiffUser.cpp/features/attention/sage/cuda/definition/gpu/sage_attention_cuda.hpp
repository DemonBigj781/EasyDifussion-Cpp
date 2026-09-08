#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace edcpp::api::attention::sage::cuda::definition::gpu {

enum class NativeDType : std::uint8_t { f32, f16, bf16 };

struct NativeTensor4D {
    const void* data = nullptr;
    NativeDType dtype = NativeDType::f32;
    std::int64_t head_dim = 0;
    std::int64_t tokens = 0;
    std::int64_t heads = 0;
    std::int64_t batch = 0;
    std::array<std::size_t, 4> byte_strides{};
};

struct NativeMutableTensor4D {
    void* data = nullptr;
    NativeDType dtype = NativeDType::f32;
    std::int64_t head_dim = 0;
    std::int64_t tokens = 0;
    std::int64_t heads = 0;
    std::int64_t batch = 0;
    std::array<std::size_t, 4> byte_strides{};
};

struct NativeSupportRequest {
    NativeTensor4D query{};
    NativeTensor4D key{};
    NativeTensor4D value{};
    NativeMutableTensor4D output{};
    int device_index = -1;
    int compute_major = 0;
    int compute_minor = 0;
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    bool additive_mask = false;
    bool attention_sinks = false;
};

struct NativeCapabilities {
    bool support = false;
    bool forward = false;
    bool int8_qk = false;
    bool fp16_pv = false;
    bool grouped_query = false;
    int minimum_compute_major = 0;
    int maximum_compute_major_exclusive = 0;
};

struct NativeResult {
    bool ok = false;
    const char* message = nullptr;
};

NativeCapabilities capabilities() noexcept;
NativeResult support(const NativeSupportRequest& request) noexcept;

} // namespace edcpp::api::attention::sage::cuda::definition::gpu
