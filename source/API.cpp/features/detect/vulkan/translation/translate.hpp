#pragma once
#include "detect.hpp"
#include "features/detect/vulkan/definition/detect.hpp"
#include <utility>
namespace edcpp::api::detect::vulkan::translation::detail {
inline Result translate(definition::NativeResult native, DeviceClass device_class) {
    Result result;
    result.backend = Backend::vulkan;
    result.backend_name = "vulkan";
    result.backend_available = native.available;
    result.diagnostic = std::move(native.diagnostic);
    result.devices.reserve(native.devices.size());
    for (auto& native_device : native.devices) {
        Device device;
        device.backend = Backend::vulkan;
        device.device_class = device_class;
        device.index = native_device.index < 0 ? 0u : static_cast<std::uint32_t>(native_device.index);
        device.name = std::move(native_device.name);
        device.architecture = std::move(native_device.architecture);
        device.total_memory = native_device.total_memory;
        device.available = native_device.available;
        result.devices.push_back(std::move(device));
    }
    return result;
}
} // namespace edcpp::api::detect::vulkan::translation::detail
