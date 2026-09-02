#pragma once

#include <string>

namespace edcpp::api::overflow::cpu::definition::cpu {

struct NativeRelease {
    bool released = false;
    std::string diagnostic;
};

NativeRelease release(void* handle) noexcept;

} // namespace edcpp::api::overflow::cpu::definition::cpu
