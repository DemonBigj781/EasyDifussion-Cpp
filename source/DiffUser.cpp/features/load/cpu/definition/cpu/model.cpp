#include "features/load/cpu/definition/cpu/model.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <cstddef>

namespace edcpp::api::load::cpu::definition::cpu {

NativeResult load_model(const void* data, std::uint64_t size, std::uint64_t max_bytes) {
    NativeResult result;
    if (data == nullptr || size == 0) {
        result.diagnostic = "CPU model load requires a non-empty source";
        return result;
    }
    if (size > max_bytes || size > std::numeric_limits<std::size_t>::max() ||
        size > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        result.diagnostic = "CPU byte allocation exceeds its budget or addressable size";
        return result;
    }
    result.handle = std::malloc(static_cast<std::size_t>(size));
    if (result.handle == nullptr) {
        result.diagnostic = "CPU model allocation failed";
        return result;
    }
    std::memcpy(result.handle, data, static_cast<std::size_t>(size));
    result.loaded = true;
    result.size = size;
    return result;
}

void discard_model(void* handle) noexcept { std::free(handle); }

} // namespace edcpp::api::load::cpu::definition::cpu
