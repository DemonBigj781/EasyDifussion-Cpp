#pragma once

#include <cstdint>
#include <string>

namespace edcpp::api::overflow::cpu::definition::cpu {

struct NativeAllocation {
    bool allocated = false;
    void* handle = nullptr;
    std::uint64_t size = 0;
    std::string diagnostic;
};

NativeAllocation allocate(
    const void* data, std::uint64_t size, std::uint64_t host_reserve);

} // namespace edcpp::api::overflow::cpu::definition::cpu
