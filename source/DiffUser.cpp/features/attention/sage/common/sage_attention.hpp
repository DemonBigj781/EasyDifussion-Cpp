#pragma once

#include "common/api.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace edcpp::api::attention::sage {

enum class DType : std::uint8_t { f32, f16, bf16 };

struct Tensor4D {
    const void* data = nullptr;
    DType dtype = DType::f32;
    std::int64_t head_dim = 0;
    std::int64_t tokens = 0;
    std::int64_t heads = 0;
    std::int64_t batch = 0;
    // Byte strides in head_dim, token, head, batch order. All-zero denotes
    // contiguous storage for the tensor's dtype.
    std::array<std::size_t, 4> byte_strides{};
};

struct MutableTensor4D {
    void* data = nullptr;
    DType dtype = DType::f32;
    std::int64_t head_dim = 0;
    std::int64_t tokens = 0;
    std::int64_t heads = 0;
    std::int64_t batch = 0;
    std::array<std::size_t, 4> byte_strides{};
};

struct Device {
    int index = -1;
    int compute_major = 0;
    int compute_minor = 0;
};

struct SupportRequest {
    Tensor4D query{};
    Tensor4D key{};
    Tensor4D value{};
    MutableTensor4D output{};
    Device device{};
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    bool additive_mask = false;
    bool attention_sinks = false;
};

struct Capabilities {
    bool support = false;
    bool forward = false;
    bool int8_qk = false;
    bool fp16_pv = false;
    bool grouped_query = false;
    bool additive_mask = false;
    bool attention_sinks = false;
    int minimum_compute_major = 0;
    int maximum_compute_major_exclusive = 0;
};

struct Result {
    bool ok = false;
    const char* message = nullptr;
};

using SupportFn = Result (*)(const SupportRequest&) noexcept;

struct Translation {
    Backend backend = Backend::none;
    const char* name = "unknown";
    Capabilities capabilities{};
    SupportFn support = nullptr;
};

bool register_translation(const Translation* translation) noexcept;
const Translation* translation_for(Backend backend) noexcept;
Capabilities capabilities(Backend backend) noexcept;
Result support(Backend backend, const SupportRequest& request) noexcept;

} // namespace edcpp::api::attention::sage
