#include "detect.hpp"

#ifndef EDCPP_DETECT_ROCM
#define EDCPP_DETECT_ROCM 0
#endif

#if EDCPP_DETECT_ROCM
#include <hip/hip_runtime_api.h>
#endif

#include <cstddef>
#include <utility>

namespace edcpp::api::detect::rocm::definition::gpu {

NativeResult detect() {
    NativeResult result;

#if EDCPP_DETECT_ROCM
    int count = 0;
    const hipError_t count_rc = hipGetDeviceCount(&count);
    if (count_rc != hipSuccess) {
        result.diagnostic = hipGetErrorString(count_rc);
        hipGetLastError();
        return result;
    }

    result.available = count > 0;
    result.devices.reserve(static_cast<std::size_t>(count));

    int previous_device = -1;
    const bool have_previous_device = hipGetDevice(&previous_device) == hipSuccess;
    if (!have_previous_device) {
        hipGetLastError();
    }

    for (int index = 0; index < count; ++index) {
        hipDeviceProp_t properties{};
        const hipError_t properties_rc = hipGetDeviceProperties(&properties, index);
        if (properties_rc != hipSuccess) {
            hipGetLastError();
            continue;
        }

        NativeDevice device;
        device.index = index;
        device.name = properties.name;
        device.architecture = properties.gcnArchName;
        device.compute_major = properties.major;
        device.compute_minor = properties.minor;
        device.total_memory = static_cast<std::uint64_t>(properties.totalGlobalMem);
        device.available = true;

        if (hipSetDevice(index) == hipSuccess) {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
                device.free_memory = static_cast<std::uint64_t>(free_bytes);
                device.total_memory = static_cast<std::uint64_t>(total_bytes);
            } else {
                hipGetLastError();
            }
        } else {
            hipGetLastError();
        }

        result.devices.push_back(std::move(device));
    }

    if (have_previous_device && previous_device >= 0) {
        if (hipSetDevice(previous_device) != hipSuccess) {
            hipGetLastError();
        }
    }

    if (result.available && result.devices.empty()) {
        result.available = false;
        result.diagnostic = "ROCm reported devices but their properties could not be read";
    }
#else
    result.diagnostic = "ROCm detection was not enabled for this compiler target";
#endif

    return result;
}

} // namespace edcpp::api::detect::rocm::definition::gpu
