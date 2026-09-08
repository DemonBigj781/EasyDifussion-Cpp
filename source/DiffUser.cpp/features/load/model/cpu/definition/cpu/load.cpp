#include "load.hpp"

#include <cstdlib>
#include <cstring>

namespace edcpp::api::load::model::cpu::definition::cpu {

NativeResult load(const void* data, std::uint64_t size) {
    NativeResult result;
    if (data == nullptr || size == 0) {
        result.diagnostic = "CPU model load requires a non-empty source";
        return result;
    }

    void* allocation = std::malloc(static_cast<std::size_t>(size));
    if (allocation == nullptr) {
        result.diagnostic = "CPU model allocation failed";
        return result;
    }

    std::memcpy(allocation, data, static_cast<std::size_t>(size));
    result.loaded = true;
    result.handle = allocation;
    result.size = size;
    return result;
}

} // namespace edcpp::api::load::model::cpu::definition::cpu
