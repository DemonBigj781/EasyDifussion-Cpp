#include "release.hpp"

#include <cstdlib>

namespace edcpp::api::overflow::cpu::definition::cpu {

NativeRelease release(void* handle) noexcept {
    NativeRelease result;
    if (handle == nullptr) {
        result.diagnostic = "CPU overflow release received an empty handle";
        return result;
    }
    std::free(handle);
    result.released = true;
    return result;
}

} // namespace edcpp::api::overflow::cpu::definition::cpu
