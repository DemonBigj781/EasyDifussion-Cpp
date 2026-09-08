#include "unload.hpp"

#include <cstdlib>

namespace edcpp::api::unload::model::cpu::definition::cpu {

NativeResult unload(void* handle) noexcept {
    NativeResult result;
    if (handle == nullptr) {
        result.diagnostic = "CPU model unload received an empty handle";
        return result;
    }

    std::free(handle);
    result.unloaded = true;
    return result;
}

} // namespace edcpp::api::unload::model::cpu::definition::cpu
