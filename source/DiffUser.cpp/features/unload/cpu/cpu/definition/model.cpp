#include "features/unload/cpu/cpu/definition/model.hpp"

#include <cstdlib>

namespace edcpp::api::unload::cpu::cpu::definition {

NativeResult unload_model(void* handle) noexcept {
    if (handle == nullptr) return {false, "CPU model unload received an empty handle"};
    std::free(handle);
    return {true, {}};
}

} // namespace edcpp::api::unload::cpu::cpu::definition
