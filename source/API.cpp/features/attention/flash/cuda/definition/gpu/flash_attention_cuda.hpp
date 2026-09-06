#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace edcpp::api::attention::flash::cuda::definition::gpu {

enum class NativeDType : std::uint8_t {
    f32,
    f16,
    bf16,
};

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

struct NativeExecutionContext {
    void* stream = nullptr;
    bool synchronize = true;
    int device_index = -1;
};

struct NativeRequest {
    NativeTensor4D query{};
    NativeTensor4D key{};
    NativeTensor4D value{};
    NativeTensor4D mask{};
    NativeMutableTensor4D output{};
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    NativeExecutionContext execution{};
};

struct NativeCapabilities {
    bool fused_forward = false;
    bool additive_mask = false;
    bool alibi_bias = false;
    bool logit_softcap = false;
    bool grouped_query = false;
    bool f32_accumulation = false;
    bool f32 = false;
    bool f16 = false;
    bool bf16 = false;
    std::int64_t maximum_value_dimension = 0;
    int minimum_compute_major = 0;
};

struct NativeResult {
    bool ok = false;
    const char* message = nullptr;
};

NativeCapabilities capabilities() noexcept;
NativeResult validate_forward(const NativeRequest& request) noexcept;

} // namespace edcpp::api::attention::flash::cuda::definition::gpu
