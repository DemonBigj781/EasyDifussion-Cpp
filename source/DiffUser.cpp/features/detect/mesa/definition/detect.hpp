#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace edcpp::api::detect::mesa::definition {
enum class NativeClass : std::uint8_t { cpu, gpu };
struct NativeDevice {
    int index = 0;
    NativeClass device_class = NativeClass::gpu;
    std::string name;
    std::string architecture;
    bool available = false;
};
struct NativeResult {
    bool available = false;
    std::vector<NativeDevice> devices;
    std::string diagnostic;
};
namespace cpu { NativeResult detect(); }
namespace gpu { NativeResult detect(); }
} // namespace edcpp::api::detect::mesa::definition
