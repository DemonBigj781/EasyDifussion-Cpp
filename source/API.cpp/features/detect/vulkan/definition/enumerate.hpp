#pragma once
#include "detect.hpp"

#ifndef EDCPP_DETECT_VULKAN
#define EDCPP_DETECT_VULKAN 0
#endif
#if EDCPP_DETECT_VULKAN
#include <vulkan/vulkan.h>
#endif

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace edcpp::api::detect::vulkan::definition::detail {

#if EDCPP_DETECT_VULKAN
inline bool matches(VkPhysicalDeviceType type, NativeClass wanted) noexcept {
    if (wanted == NativeClass::cpu) return type == VK_PHYSICAL_DEVICE_TYPE_CPU;
    return type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
           type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
           type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
}
#endif

inline NativeResult enumerate(NativeClass wanted) {
    NativeResult result;
#if EDCPP_DETECT_VULKAN
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "Easy Diffusion API.cpp detect";
    application.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    application.pEngineName = "API.cpp";
    application.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    application.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &application;

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult create_rc = vkCreateInstance(&create_info, nullptr, &instance);
    if (create_rc != VK_SUCCESS) {
        result.diagnostic = "vkCreateInstance failed with code " + std::to_string(create_rc);
        return result;
    }

    std::uint32_t device_count = 0;
    VkResult enumerate_rc = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (enumerate_rc != VK_SUCCESS) {
        result.diagnostic = "vkEnumeratePhysicalDevices failed with code " +
                            std::to_string(enumerate_rc);
        vkDestroyInstance(instance, nullptr);
        return result;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    if (device_count != 0) {
        enumerate_rc = vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    }
    if (enumerate_rc == VK_SUCCESS) {
        for (const auto physical_device : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physical_device, &properties);
            if (!matches(properties.deviceType, wanted)) continue;

            NativeDevice device;
            device.index = static_cast<int>(result.devices.size());
            device.device_class = wanted;
            device.name = properties.deviceName;
            std::ostringstream identity;
            identity << "vendor_" << std::hex << std::setw(4) << std::setfill('0')
                     << properties.vendorID << "/device_" << std::setw(4)
                     << properties.deviceID;
            device.architecture = identity.str();

            VkPhysicalDeviceMemoryProperties memory{};
            vkGetPhysicalDeviceMemoryProperties(physical_device, &memory);
            for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) {
                if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                    device.total_memory += memory.memoryHeaps[heap].size;
                }
            }
            device.available = true;
            result.devices.push_back(std::move(device));
        }
    } else {
        result.diagnostic = "Vulkan physical-device list changed during enumeration";
    }

    vkDestroyInstance(instance, nullptr);
    result.available = !result.devices.empty();
    if (!result.available && result.diagnostic.empty()) {
        result.diagnostic = "Vulkan reported no devices for the requested class";
    }
#else
    (void) wanted;
    result.diagnostic = "Vulkan detection was not enabled for this compiler target";
#endif
    return result;
}
} // namespace edcpp::api::detect::vulkan::definition::detail
