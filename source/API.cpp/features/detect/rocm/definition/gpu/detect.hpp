#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace edcpp::api::detect::rocm::definition::gpu {

struct NativeDevice {
    int index = -1;
    std::string name;
    std::string architecture;
    int compute_major = 0;
    int compute_minor = 0;
    std::uint64_t total_memory = 0;
    std::uint64_t free_memory = 0;
    bool available = false;
};

struct NativeResult {
    bool available = false;
    std::vector<NativeDevice> devices;
    std::string diagnostic;
};

NativeResult detect();

} // namespace edcpp::api::detect::rocm::definition::gpu
