#include "features/load/cpu/definition/cpu/model.hpp"

#include "native/cpu/byte_storage.hpp"
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
    auto allocation = edcpp::native::cpu::copy_bytes(data, static_cast<std::size_t>(size));
    result.handle = allocation.data;
    result.allocation_id = allocation.identity;
    result.loaded = true;
    result.size = size;
    return result;
}

bool discard_model(void* handle, std::uint64_t allocation_id) noexcept {
    return edcpp::native::cpu::release_bytes(handle, allocation_id);
}

} // namespace edcpp::api::load::cpu::definition::cpu
