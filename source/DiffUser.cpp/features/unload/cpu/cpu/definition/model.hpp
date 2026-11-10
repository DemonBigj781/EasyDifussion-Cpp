#pragma once

#include <string>
#include <cstdint>

namespace edcpp::api::unload::cpu::cpu::definition {

struct NativeResult {
    bool unloaded = false;
    std::string diagnostic{};
};

NativeResult unload_model(void* handle, std::uint64_t allocation_id) noexcept;

} // namespace edcpp::api::unload::cpu::cpu::definition
