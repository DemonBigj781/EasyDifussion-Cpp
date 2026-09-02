#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace edcpp::api::detect::openvino::definition {

enum class NativeClass : std::uint8_t { cpu, gpu, npu };

struct NativeDevice {
    int index = -1;
    NativeClass device_class = NativeClass::cpu;
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
namespace npu { NativeResult detect(); }

} // namespace edcpp::api::detect::openvino::definition
