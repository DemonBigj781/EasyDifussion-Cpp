#pragma once

#include "detect.hpp"

#ifndef EDCPP_DETECT_OPENCL
#define EDCPP_DETECT_OPENCL 0
#endif

#if EDCPP_DETECT_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace edcpp::api::detect::opencl::definition::detail {

#if EDCPP_DETECT_OPENCL
inline std::string device_string(cl_device_id device, cl_device_info field) {
    std::size_t size = 0;
    if (clGetDeviceInfo(device, field, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    if (clGetDeviceInfo(device, field, size, value.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    while (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

inline cl_device_type native_type(NativeClass wanted) noexcept {
    switch (wanted) {
        case NativeClass::cpu: return CL_DEVICE_TYPE_CPU;
        case NativeClass::gpu: return CL_DEVICE_TYPE_GPU;
        case NativeClass::npu: return CL_DEVICE_TYPE_ACCELERATOR;
    }
    return CL_DEVICE_TYPE_DEFAULT;
}
#endif

inline NativeResult enumerate(NativeClass wanted) {
    NativeResult result;

#if EDCPP_DETECT_OPENCL
    cl_uint platform_count = 0;
    const cl_int platform_rc = clGetPlatformIDs(0, nullptr, &platform_count);
    if (platform_rc != CL_SUCCESS) {
        result.diagnostic = "clGetPlatformIDs failed with code " + std::to_string(platform_rc);
        return result;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    if (platform_count != 0 &&
        clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
        result.diagnostic = "OpenCL platform enumeration failed";
        return result;
    }

    for (const auto platform : platforms) {
        cl_uint device_count = 0;
        const cl_int count_rc = clGetDeviceIDs(
            platform, native_type(wanted), 0, nullptr, &device_count);
        if (count_rc == CL_DEVICE_NOT_FOUND) {
            continue;
        }
        if (count_rc != CL_SUCCESS) {
            if (!result.diagnostic.empty()) {
                result.diagnostic += "; ";
            }
            result.diagnostic += "clGetDeviceIDs failed with code " + std::to_string(count_rc);
            continue;
        }

        std::vector<cl_device_id> devices(device_count);
        if (device_count != 0 &&
            clGetDeviceIDs(platform, native_type(wanted), device_count,
                           devices.data(), nullptr) != CL_SUCCESS) {
            continue;
        }

        for (const auto opencl_device : devices) {
            NativeDevice device;
            device.index = static_cast<int>(result.devices.size());
            device.device_class = wanted;
            device.name = device_string(opencl_device, CL_DEVICE_NAME);
            const auto vendor = device_string(opencl_device, CL_DEVICE_VENDOR);
            const auto driver = device_string(opencl_device, CL_DRIVER_VERSION);
            device.architecture = vendor;
            if (!driver.empty()) {
                device.architecture += "/" + driver;
            }

            cl_ulong total_memory = 0;
            clGetDeviceInfo(opencl_device, CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(total_memory), &total_memory, nullptr);
            device.total_memory = static_cast<std::uint64_t>(total_memory);

            cl_bool available = CL_FALSE;
            clGetDeviceInfo(opencl_device, CL_DEVICE_AVAILABLE,
                            sizeof(available), &available, nullptr);
            device.available = available == CL_TRUE;
            result.devices.push_back(std::move(device));
        }
    }

    result.available = std::any_of(
        result.devices.begin(), result.devices.end(),
        [](const NativeDevice& device) { return device.available; });
    if (!result.available && result.diagnostic.empty()) {
        result.diagnostic = "OpenCL reported no available devices for the requested class";
    }
#else
    (void) wanted;
    result.diagnostic = "OpenCL detection was not enabled for this compiler target";
#endif

    return result;
}

} // namespace edcpp::api::detect::opencl::definition::detail
