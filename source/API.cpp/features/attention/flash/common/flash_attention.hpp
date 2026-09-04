#pragma once

#include "common/api.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace edcpp::api::attention::flash {

enum class DType : std::uint8_t { f32, f16, bf16 };

struct Tensor4D {
    const void* data = nullptr;
    DType dtype = DType::f32;
    std::int64_t head_dim = 0;
    std::int64_t tokens = 0;
    std::int64_t heads = 0;
    std::int64_t batch = 0;
    // Byte strides in head_dim, token, head, batch order. All-zero means contiguous.
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

struct ExecutionContext {
    std::int32_t thread_index = 0;
    std::int32_t thread_count = 1;
    void* workspace = nullptr;
    std::size_t workspace_size = 0;
    bool reference = false;
};

struct Request {
    Tensor4D query{};
    Tensor4D key{};
    Tensor4D value{};
    Tensor4D mask{}; // Optional when data is null.
    MutableTensor4D output{};
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    ExecutionContext execution{};
};

struct Capabilities {
    bool forward = false;
    bool additive_mask = false;
    bool alibi_bias = false;
    bool logit_softcap = false;
    bool grouped_query = false;
    bool f32_accumulation = false;
};

struct Result {
    bool ok = false;
    const char* message = nullptr;
};

using ValidateFn = Result (*)(const Request&) noexcept;
using ForwardFn = Result (*)(const Request&) noexcept;

struct Translation {
    Backend backend = Backend::none;
    const char* name = "unknown";
    Capabilities capabilities{};
    ValidateFn validate = nullptr;
    ForwardFn forward = nullptr;
};

bool register_translation(const Translation* translation) noexcept;
const Translation* translation_for(Backend backend) noexcept;
Capabilities capabilities(Backend backend) noexcept;
Result validate(Backend backend, const Request& request) noexcept;
Result forward(Backend backend, const Request& request) noexcept;

} // namespace edcpp::api::attention::flash
