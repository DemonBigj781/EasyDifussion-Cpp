#include "detect.hpp"
#include "features/detect/opengl/definition/gpu/detect.hpp"
#include <utility>

namespace edcpp::api::detect::opengl::translation::gpu {
Result detect() {
    auto native = definition::gpu::detect();
    Result result;
    result.backend = Backend::opengl;
    result.backend_name = "opengl";
    result.backend_available = native.available;
    result.diagnostic = std::move(native.diagnostic);
    result.devices.reserve(native.devices.size());
    for (auto& native_device : native.devices) {
        Device device;
        device.backend = Backend::opengl;
        device.device_class = DeviceClass::gpu;
        device.index = native_device.index < 0 ? 0u : static_cast<std::uint32_t>(native_device.index);
        device.name = std::move(native_device.name);
        device.architecture = std::move(native_device.architecture);
        device.available = native_device.available;
        result.devices.push_back(std::move(device));
    }
    return result;
}
} // namespace edcpp::api::detect::opengl::translation::gpu
