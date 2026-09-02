#include "detect.hpp"

#include "features/detect/cuda/definition/gpu/detect.hpp"

#include <string>
#include <utility>

namespace edcpp::api::detect::cuda::translation::gpu {

Result detect() {
    auto native = definition::gpu::detect();

    Result result;
    result.backend = Backend::cuda;
    result.backend_name = "cuda";
    result.backend_available = native.available;
    result.diagnostic = std::move(native.diagnostic);
    result.devices.reserve(native.devices.size());

    for (auto& native_device : native.devices) {
        Device device;
        device.backend = Backend::cuda;
        device.device_class = DeviceClass::gpu;
        device.index = native_device.index < 0
            ? 0u
            : static_cast<std::uint32_t>(native_device.index);
        device.name = std::move(native_device.name);
        device.compute_major = native_device.compute_major;
        device.compute_minor = native_device.compute_minor;
        device.architecture = "sm_" + std::to_string(device.compute_major) +
                              std::to_string(device.compute_minor);
        device.total_memory = native_device.total_memory;
        device.free_memory = native_device.free_memory;
        device.available = native_device.available;
        result.devices.push_back(std::move(device));
    }

    return result;
}

} // namespace edcpp::api::detect::cuda::translation::gpu
