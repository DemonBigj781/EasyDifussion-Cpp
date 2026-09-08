#pragma once

#include <string>

namespace edcpp::api::unload::model::cpu::definition::cpu {

struct NativeResult {
    bool unloaded = false;
    std::string diagnostic;
};

NativeResult unload(void* handle) noexcept;

} // namespace edcpp::api::unload::model::cpu::definition::cpu
