#include "features/load/cpu/definition/cpu/model.hpp"

#include <cstdlib>
#include <cstring>

namespace edcpp::api::load::cpu::definition::cpu {

NativeResult load_model(const void* data, std::uint64_t size) {
    NativeResult result;
    if (data == nullptr || size == 0) {
        result.diagnostic = "CPU model load requires a non-empty source";
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

} // namespace edcpp::api::load::cpu::definition::cpu
