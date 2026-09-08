#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace edcpp::api::detect::cpu::definition::cpu {

struct NativeDevice {
    int index = 0;
    std::string name;
    std::string architecture;
    std::uint64_t total_memory = 0;
    std::uint64_t free_memory = 0;
    bool available = false;
};

struct NativeResult {
    bool available = false;
    std::vector<NativeDevice> devices;
    std::string diagnostic;
};

// Native host meaning: report the CPU execution device, compile-target
// architecture, and an OS memory snapshot when the platform exposes one.
NativeResult detect();

} // namespace edcpp::api::detect::cpu::definition::cpu
