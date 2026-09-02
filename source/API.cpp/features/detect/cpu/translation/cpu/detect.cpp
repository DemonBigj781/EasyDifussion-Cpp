#include "detect.hpp"

#include "features/detect/cpu/definition/cpu/detect.hpp"

#include <utility>

namespace edcpp::api::detect::cpu::translation::cpu {

Result detect() {
    auto native = definition::cpu::detect();

    Result result;
    result.backend = Backend::cpu;
    result.backend_name = "cpu";
    result.backend_available = native.available;
    result.diagnostic = std::move(native.diagnostic);
    result.devices.reserve(native.devices.size());

    for (auto& native_device : native.devices) {
        Device device;
        device.backend = Backend::cpu;
        device.device_class = DeviceClass::cpu;
        device.index = native_device.index < 0
            ? 0u
            : static_cast<std::uint32_t>(native_device.index);
        device.name = std::move(native_device.name);
        device.architecture = std::move(native_device.architecture);
        device.total_memory = native_device.total_memory;
        device.free_memory = native_device.free_memory;
        device.available = native_device.available;
        result.devices.push_back(std::move(device));
    }

    return result;
}

} // namespace edcpp::api::detect::cpu::translation::cpu
