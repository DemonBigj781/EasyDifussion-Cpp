#include "features/unload/cpu/cpu/definition/model.hpp"

#include "native/cpu/byte_storage.hpp"

namespace edcpp::api::unload::cpu::cpu::definition {

NativeResult unload_model(void* handle, std::uint64_t allocation_id) noexcept {
    if (handle == nullptr) return {false, "CPU model unload received an empty handle"};
    if (!edcpp::native::cpu::release_bytes(handle, allocation_id))
        return {false, "CPU resource is stale, foreign or already released"};
    return {true, {}};
}

} // namespace edcpp::api::unload::cpu::cpu::definition
