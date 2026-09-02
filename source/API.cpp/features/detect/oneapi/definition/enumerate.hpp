#pragma once

#include "detect.hpp"

#ifndef EDCPP_DETECT_ONEAPI
#define EDCPP_DETECT_ONEAPI 0
#endif

#if EDCPP_DETECT_ONEAPI
#include <sycl/sycl.hpp>
#endif

#include <cstddef>
#include <exception>
#include <string>
#include <utility>

namespace edcpp::api::detect::oneapi::definition::detail {

#if EDCPP_DETECT_ONEAPI
inline bool matches(const sycl::device& device, NativeClass wanted) {
    switch (wanted) {
        case NativeClass::cpu: return device.is_cpu();
        case NativeClass::gpu: return device.is_gpu();
        case NativeClass::npu: return device.is_accelerator();
    }
    return false;
}
#endif

inline NativeResult enumerate(NativeClass wanted) {
    NativeResult result;

#if EDCPP_DETECT_ONEAPI
    try {
        const auto devices = sycl::device::get_devices();
        for (const auto& sycl_device : devices) {
            if (!matches(sycl_device, wanted)) {
                continue;
            }

            NativeDevice device;
            device.index = static_cast<int>(result.devices.size());
            device.device_class = wanted;
            device.name = sycl_device.get_info<sycl::info::device::name>();
            const auto vendor = sycl_device.get_info<sycl::info::device::vendor>();
            const auto platform = sycl_device.get_platform().get_info<sycl::info::platform::name>();
            device.architecture = vendor;
            if (!platform.empty() && platform != vendor) {
                device.architecture += "/" + platform;
            }
            device.total_memory = static_cast<std::uint64_t>(
                sycl_device.get_info<sycl::info::device::global_mem_size>());
            device.available = true;
            result.devices.push_back(std::move(device));
        }
        result.available = !result.devices.empty();
        if (!result.available) {
            result.diagnostic = "oneAPI reported no devices for the requested device class";
        }
    } catch (const sycl::exception& error) {
        result.diagnostic = error.what();
    } catch (const std::exception& error) {
        result.diagnostic = error.what();
    }
#else
    (void) wanted;
    result.diagnostic = "oneAPI detection was not enabled for this compiler target";
#endif

    return result;
}

} // namespace edcpp::api::detect::oneapi::definition::detail
