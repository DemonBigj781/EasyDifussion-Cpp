#include "api/overflow.hpp"

#include "features/overflow/common/overflow.hpp"

namespace easyapi {

bool OverflowResource::loaded() const noexcept {
    if (tier == OverflowTier::none || storage == OverflowStorage::none || size == 0) {
        return false;
    }
    if (storage == OverflowStorage::cpu_heap) {
        return host_handle != nullptr;
    }
    if (storage == OverflowStorage::cuda_device) {
        return device_handle != nullptr && owner_device_index >= 0;
    }
    return host_handle != nullptr && device_handle != nullptr &&
           access_device_index >= 0;
}

bool OverflowResource::gpu_accessible() const noexcept {
    return loaded() && storage != OverflowStorage::cpu_heap;
}

const char* overflow_tier_name(OverflowTier tier) noexcept {
    return edcpp::api::overflow::tier_name(
        static_cast<edcpp::api::overflow::Tier>(tier));
}

const char* overflow_storage_name(OverflowStorage storage) noexcept {
    return edcpp::api::overflow::memory_kind_name(
        static_cast<edcpp::api::overflow::MemoryKind>(storage));
}

} // namespace easyapi
