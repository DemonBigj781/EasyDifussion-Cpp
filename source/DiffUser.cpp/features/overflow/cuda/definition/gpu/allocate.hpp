#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace edcpp::api::overflow::cuda::definition::gpu {

struct NativeDevice {
    int device_index = -1;
    std::uint64_t free_memory = 0;
    std::uint64_t total_memory = 0;
};

struct NativeCapabilities {
    bool available = false;
    bool managed_memory = false;
    bool mapped_host_memory = false;
    int compute_major = 0;
    int compute_minor = 0;
    std::vector<NativeDevice> secondary_devices;
    std::string diagnostic;
};

enum class NativeMemoryKind : std::uint8_t {
    none = 0,
    device,
    managed,
    mapped_host,
};

struct NativeAllocation {
    bool allocated = false;
    bool out_of_memory = false;
    NativeMemoryKind memory_kind = NativeMemoryKind::none;
    void* host_handle = nullptr;
    void* device_handle = nullptr;
    std::uint64_t size = 0;
    int owner_device_index = -1;
    int access_device_index = -1;
    std::string diagnostic;
};

NativeCapabilities detect_capabilities(int active_device_index);
NativeAllocation allocate_device(
    const void* data, std::uint64_t size, int device_index);
NativeAllocation allocate_managed(
    const void* data, std::uint64_t size, int device_index,
    std::uint64_t host_reserve);
NativeAllocation allocate_mapped_host(
    const void* data, std::uint64_t size, int device_index,
    std::uint64_t host_reserve);

} // namespace edcpp::api::overflow::cuda::definition::gpu
