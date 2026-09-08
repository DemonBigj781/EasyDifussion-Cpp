#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace edcpp::api::detect::directml::definition::gpu {
struct NativeDevice {
    int index = -1;
    std::string name;
    std::string architecture;
    std::uint64_t total_memory = 0;
    bool available = false;
};
struct NativeResult {
    bool available = false;
    std::vector<NativeDevice> devices;
    std::string diagnostic;
};
NativeResult detect();
} // namespace edcpp::api::detect::directml::definition::gpu
