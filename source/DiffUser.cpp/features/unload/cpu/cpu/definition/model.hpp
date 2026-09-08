#pragma once

#include <string>

namespace edcpp::api::unload::cpu::cpu::definition {

struct NativeResult {
    bool unloaded = false;
    std::string diagnostic{};
};

NativeResult unload_model(void* handle) noexcept;

} // namespace edcpp::api::unload::cpu::cpu::definition
