#include "detect.hpp"

#ifndef EDCPP_DETECT_OPENGL
#define EDCPP_DETECT_OPENGL 0
#endif
#if EDCPP_DETECT_OPENGL
#include <GL/gl.h>
#endif

#include <string>
#include <utility>

namespace edcpp::api::detect::opengl::definition::gpu {

NativeResult detect() {
    NativeResult result;
#if EDCPP_DETECT_OPENGL
    const auto* renderer = glGetString(GL_RENDERER);
    const auto* vendor = glGetString(GL_VENDOR);
    const auto* version = glGetString(GL_VERSION);
    if (renderer == nullptr) {
        result.diagnostic = "OpenGL detection requires a current context";
        return result;
    }

    NativeDevice device;
    device.name = reinterpret_cast<const char*>(renderer);
    if (vendor != nullptr) {
        device.architecture = reinterpret_cast<const char*>(vendor);
    }
    if (version != nullptr) {
        if (!device.architecture.empty()) device.architecture += '/';
        device.architecture += reinterpret_cast<const char*>(version);
    }
    device.available = true;
    result.available = true;
    result.devices.push_back(std::move(device));
#else
    result.diagnostic = "OpenGL detection was not enabled for this compiler target";
#endif
    return result;
}

} // namespace edcpp::api::detect::opengl::definition::gpu
