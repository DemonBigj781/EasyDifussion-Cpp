#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace edcpp::api {

enum class Backend : std::uint8_t {
    none = 0,
    cpu,
    cuda,
    rocm,
    oneapi,
    opencl,
    openvino,
    opengl,
    vulkan,
    mesa,
    directml,
};

enum class Operation : std::uint16_t {
    unknown = 0,
    attention,
    flash_attention,
    sage_attention,
    xformers_attention,
    matmul,
    convolution,
    normalization,
    sampling,
};

enum class Capability : std::uint64_t {
    none               = 0,
    attention          = 1ull << 0,
    flash_attention    = 1ull << 1,
    sage_attention     = 1ull << 2,
    xformers_attention = 1ull << 3,
    fp16               = 1ull << 4,
    bf16               = 1ull << 5,
    fp32               = 1ull << 6,
    int8               = 1ull << 7,
    async_execution    = 1ull << 8,
    unified_memory     = 1ull << 9,
};

constexpr Capability operator|(Capability a, Capability b) noexcept {
    return static_cast<Capability>(
        static_cast<std::uint64_t>(a) | static_cast<std::uint64_t>(b));
}

constexpr Capability operator&(Capability a, Capability b) noexcept {
    return static_cast<Capability>(
        static_cast<std::uint64_t>(a) & static_cast<std::uint64_t>(b));
}

constexpr bool has_capability(Capability set, Capability value) noexcept {
    return (static_cast<std::uint64_t>(set) & static_cast<std::uint64_t>(value)) ==
           static_cast<std::uint64_t>(value);
}

struct DeviceInfo {
    Backend backend = Backend::none;
    std::uint32_t device_index = 0;
    std::string_view name{};
    std::string_view architecture{};
    std::uint64_t total_memory = 0;
    Capability capabilities = Capability::none;
};

struct DispatchContext {
    DeviceInfo device{};
    void* backend_context = nullptr;
    void* stream = nullptr;
};

using DispatchFn = bool (*)(DispatchContext&, Operation, void*) noexcept;

struct BackendInterface {
    Backend backend = Backend::none;
    std::string_view name{};
    Capability capabilities = Capability::none;
    DispatchFn dispatch = nullptr;
};

const char* backend_name(Backend backend) noexcept;
const char* operation_name(Operation operation) noexcept;

} // namespace edcpp::api
