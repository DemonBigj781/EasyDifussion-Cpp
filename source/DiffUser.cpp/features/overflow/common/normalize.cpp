#include "overflow.hpp"

namespace edcpp::api::overflow {

bool Resource::loaded() const noexcept {
    if (backend == Backend::none || tier == Tier::none ||
        memory_kind == MemoryKind::none || size == 0) {
        return false;
    }

    switch (memory_kind) {
        case MemoryKind::cpu_heap:
            return host_handle != nullptr;
        case MemoryKind::cuda_device:
            return device_handle != nullptr && owner_device_index >= 0;
        case MemoryKind::cuda_managed:
            return host_handle != nullptr && device_handle != nullptr &&
                   access_device_index >= 0;
        case MemoryKind::cuda_mapped_host:
            return host_handle != nullptr && device_handle != nullptr &&
                   access_device_index >= 0;
        case MemoryKind::none:
            return false;
    }
    return false;
}

bool Resource::gpu_accessible() const noexcept {
    return loaded() &&
           (memory_kind == MemoryKind::cuda_device ||
            memory_kind == MemoryKind::cuda_managed ||
            memory_kind == MemoryKind::cuda_mapped_host);
}

Result normalize(Result result) {
    if (!result.allocated || !result.resource.loaded()) {
        result.allocated = false;
        result.overflowed = false;
        result.resource = {};
        if (result.diagnostic.empty()) {
            result.diagnostic = "overflow allocation did not produce a valid owned resource";
        }
        return result;
    }

    result.overflowed = result.resource.tier != Tier::active_vram;
    return result;
}

ReleaseResult normalize(ReleaseResult result, const Resource& resource) {
    if (result.released) {
        if (!resource.loaded()) {
            result.released = false;
            if (result.diagnostic.empty()) {
                result.diagnostic = "overflow release accepted an invalid resource";
            }
        }
    } else if (result.diagnostic.empty()) {
        result.diagnostic = "overflow resource was not released";
    }
    return result;
}

const char* tier_name(Tier tier) noexcept {
    switch (tier) {
        case Tier::none: return "none";
        case Tier::active_vram: return "active-vram";
        case Tier::secondary_vram: return "secondary-vram";
        case Tier::gpu_zram: return "gpu-zram";
        case Tier::system_ram: return "system-ram";
        case Tier::zram: return "zram";
        case Tier::swap: return "swap";
    }
    return "unknown";
}

const char* memory_kind_name(MemoryKind kind) noexcept {
    switch (kind) {
        case MemoryKind::none: return "none";
        case MemoryKind::cpu_heap: return "cpu-heap";
        case MemoryKind::cuda_device: return "cuda-device";
        case MemoryKind::cuda_managed: return "cuda-managed";
        case MemoryKind::cuda_mapped_host: return "cuda-mapped-host";
    }
    return "unknown";
}

} // namespace edcpp::api::overflow
