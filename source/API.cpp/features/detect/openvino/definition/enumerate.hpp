#pragma once

#include "detect.hpp"

#ifndef EDCPP_DETECT_OPENVINO
#define EDCPP_DETECT_OPENVINO 0
#endif

#if EDCPP_DETECT_OPENVINO
#include <openvino/openvino.hpp>
#endif

#include <exception>
#include <string>
#include <utility>

namespace edcpp::api::detect::openvino::definition::detail {

inline const char* class_prefix(NativeClass wanted) noexcept {
    switch (wanted) {
        case NativeClass::cpu: return "CPU";
        case NativeClass::gpu: return "GPU";
        case NativeClass::npu: return "NPU";
    }
    return "";
}

inline bool matches(const std::string& device_name, NativeClass wanted) {
    const std::string prefix = class_prefix(wanted);
    return device_name == prefix ||
           (device_name.size() > prefix.size() &&
            device_name.compare(0, prefix.size(), prefix) == 0 &&
            (device_name[prefix.size()] == '.' || device_name[prefix.size()] == ':'));
}

inline NativeResult enumerate(NativeClass wanted) {
    NativeResult result;

#if EDCPP_DETECT_OPENVINO
    try {
        ov::Core core;
        for (const auto& openvino_device : core.get_available_devices()) {
            if (!matches(openvino_device, wanted)) {
                continue;
            }
            NativeDevice device;
            device.index = static_cast<int>(result.devices.size());
            device.device_class = wanted;
            device.name = openvino_device;
            device.architecture = class_prefix(wanted);
            device.available = true;
            result.devices.push_back(std::move(device));
        }
        result.available = !result.devices.empty();
        if (!result.available) {
            result.diagnostic = "OpenVINO reported no devices for the requested class";
        }
    } catch (const std::exception& error) {
        result.diagnostic = error.what();
    }
#else
    (void) wanted;
    result.diagnostic = "OpenVINO detection was not enabled for this compiler target";
#endif
    return result;
}

} // namespace edcpp::api::detect::openvino::definition::detail
