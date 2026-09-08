#pragma once
#include "detect.hpp"
#include "features/detect/mesa/definition/detect.hpp"
#include <utility>
namespace edcpp::api::detect::mesa::translation::detail {
inline Result translate(definition::NativeResult native, DeviceClass device_class) {
    Result result;
    result.backend = Backend::mesa;
    result.backend_name = "mesa";
    result.backend_available = native.available;
    result.diagnostic = std::move(native.diagnostic);
    result.devices.reserve(native.devices.size());
    for (auto& native_device : native.devices) {
        Device device;
        device.backend = Backend::mesa;
        device.device_class = device_class;
        device.index = native_device.index < 0 ? 0u : static_cast<std::uint32_t>(native_device.index);
        device.name = std::move(native_device.name);
        device.architecture = std::move(native_device.architecture);
        device.available = native_device.available;
        result.devices.push_back(std::move(device));
    }
    return result;
}
} // namespace edcpp::api::detect::mesa::translation::detail
