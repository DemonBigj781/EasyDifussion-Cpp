#include "detect.hpp"

#ifndef EDCPP_DETECT_CUDA
#define EDCPP_DETECT_CUDA 0
#endif

#if EDCPP_DETECT_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cstddef>
#include <utility>

namespace edcpp::api::detect::cuda::definition::gpu {

NativeResult detect() {
    NativeResult result;

#if EDCPP_DETECT_CUDA
    int count = 0;
    const cudaError_t count_rc = cudaGetDeviceCount(&count);
    if (count_rc != cudaSuccess) {
        result.diagnostic = cudaGetErrorString(count_rc);
        cudaGetLastError();
        return result;
    }

    result.available = count > 0;
    result.devices.reserve(static_cast<std::size_t>(count));

    int previous_device = -1;
    const bool have_previous_device = cudaGetDevice(&previous_device) == cudaSuccess;
    if (!have_previous_device) {
        cudaGetLastError();
    }

    for (int index = 0; index < count; ++index) {
        cudaDeviceProp properties{};
        const cudaError_t properties_rc = cudaGetDeviceProperties(&properties, index);
        if (properties_rc != cudaSuccess) {
            cudaGetLastError();
            continue;
        }

        NativeDevice device;
        device.index = index;
        device.name = properties.name;
        device.compute_major = properties.major;
        device.compute_minor = properties.minor;
        device.total_memory = static_cast<std::uint64_t>(properties.totalGlobalMem);
        device.available = true;

        if (cudaSetDevice(index) == cudaSuccess) {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                device.free_memory = static_cast<std::uint64_t>(free_bytes);
                device.total_memory = static_cast<std::uint64_t>(total_bytes);
            } else {
                cudaGetLastError();
            }
        } else {
            cudaGetLastError();
        }

        result.devices.push_back(std::move(device));
    }

    if (have_previous_device && previous_device >= 0) {
        if (cudaSetDevice(previous_device) != cudaSuccess) {
            cudaGetLastError();
        }
    }

    if (result.available && result.devices.empty()) {
        result.available = false;
        result.diagnostic = "CUDA reported devices but their properties could not be read";
    }
#else
    result.diagnostic = "CUDA detection was not enabled for this compiler target";
#endif

    return result;
}

} // namespace edcpp::api::detect::cuda::definition::gpu
