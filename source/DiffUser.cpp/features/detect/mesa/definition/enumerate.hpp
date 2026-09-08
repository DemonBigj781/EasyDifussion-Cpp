#pragma once
#include "detect.hpp"

#ifndef EDCPP_DETECT_MESA
#define EDCPP_DETECT_MESA 0
#endif
#if EDCPP_DETECT_MESA
#include <GL/gl.h>
#endif

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

namespace edcpp::api::detect::mesa::definition::detail {
inline std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline NativeResult enumerate(NativeClass wanted) {
    NativeResult result;
#if EDCPP_DETECT_MESA
    const auto* renderer_value = glGetString(GL_RENDERER);
    const auto* vendor_value = glGetString(GL_VENDOR);
    const auto* version_value = glGetString(GL_VERSION);
    if (renderer_value == nullptr) {
        result.diagnostic = "Mesa detection requires a current OpenGL context";
        return result;
    }

    const std::string renderer = reinterpret_cast<const char*>(renderer_value);
    const std::string vendor = vendor_value == nullptr
        ? std::string{}
        : reinterpret_cast<const char*>(vendor_value);
    const std::string version = version_value == nullptr
        ? std::string{}
        : reinterpret_cast<const char*>(version_value);
    const std::string identity = lowercase(renderer + ' ' + vendor + ' ' + version);
    if (identity.find("mesa") == std::string::npos) {
        result.diagnostic = "the current OpenGL context is not provided by Mesa";
        return result;
    }

    const bool software = identity.find("llvmpipe") != std::string::npos ||
                          identity.find("softpipe") != std::string::npos ||
                          identity.find("swrast") != std::string::npos;
    if ((wanted == NativeClass::cpu) != software) {
        result.diagnostic = "Mesa renderer does not match the requested device class";
        return result;
    }

    NativeDevice device;
    device.device_class = wanted;
    device.name = renderer;
    device.architecture = vendor;
    if (!version.empty()) {
        if (!device.architecture.empty()) device.architecture += '/';
        device.architecture += version;
    }
    device.available = true;
    result.available = true;
    result.devices.push_back(std::move(device));
#else
    (void) wanted;
    result.diagnostic = "Mesa detection was not enabled for this compiler target";
#endif
    return result;
}
} // namespace edcpp::api::detect::mesa::definition::detail
