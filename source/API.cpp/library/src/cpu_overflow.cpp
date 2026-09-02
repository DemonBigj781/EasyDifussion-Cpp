#include "api/overflow.hpp"

#include "features/overflow/common/overflow.hpp"
#include "features/overflow/cpu/translation/cpu/allocate.hpp"
#include "features/overflow/cpu/translation/cpu/release.hpp"

#include <cstdint>
#include <utility>

namespace easyapi {
namespace {

OverflowTier to_public(edcpp::api::overflow::Tier tier) noexcept {
    return static_cast<OverflowTier>(tier);
}

OverflowStorage to_public(edcpp::api::overflow::MemoryKind kind) noexcept {
    return static_cast<OverflowStorage>(kind);
}

edcpp::api::overflow::Tier to_internal(OverflowTier tier) noexcept {
    return static_cast<edcpp::api::overflow::Tier>(tier);
}

edcpp::api::overflow::MemoryKind to_internal(OverflowStorage storage) noexcept {
    return static_cast<edcpp::api::overflow::MemoryKind>(storage);
}

OverflowResource to_public(const edcpp::api::overflow::Resource& resource) {
    OverflowResource result;
    result.tier = to_public(resource.tier);
    result.storage = to_public(resource.memory_kind);
    result.host_handle = resource.host_handle;
    result.device_handle = resource.device_handle;
    result.size = static_cast<std::size_t>(resource.size);
    result.owner_device_index = resource.owner_device_index;
    result.access_device_index = resource.access_device_index;
    return result;
}

edcpp::api::overflow::Resource to_internal(const OverflowResource& resource) {
    edcpp::api::overflow::Resource result;
    result.backend = edcpp::api::Backend::cpu;
    result.tier = to_internal(resource.tier);
    result.memory_kind = to_internal(resource.storage);
    result.host_handle = resource.host_handle;
    result.device_handle = resource.device_handle;
    result.size = static_cast<std::uint64_t>(resource.size);
    result.owner_device_index = resource.owner_device_index;
    result.access_device_index = resource.access_device_index;
    return result;
}

} // namespace

const char* CpuOverflowHandler::name() const noexcept {
    return "cpu";
}

OverflowResult CpuOverflowHandler::allocate(const OverflowRequest& request) const {
    edcpp::api::overflow::Request internal_request;
    internal_request.data = request.data;
    internal_request.size = static_cast<std::uint64_t>(request.size);
    internal_request.host_reserve = static_cast<std::uint64_t>(request.host_reserve);

    auto internal = edcpp::api::overflow::normalize(
        edcpp::api::overflow::cpu::translation::cpu::allocate(internal_request));
    OverflowResult result;
    result.success = internal.allocated;
    result.overflowed = internal.overflowed;
    result.primary_out_of_memory = internal.primary_out_of_memory;
    result.diagnostic = std::move(internal.diagnostic);
    if (internal.allocated) {
        result.resource = to_public(internal.resource);
    }
    return result;
}

OverflowReleaseResult CpuOverflowHandler::release(OverflowResource& resource) const {
    OverflowReleaseResult result;
    if (resource.storage != OverflowStorage::cpu_heap) {
        result.diagnostic = "CPU overflow cannot release a resource owned by another allocator";
        return result;
    }

    auto internal_resource = to_internal(resource);
    auto internal = edcpp::api::overflow::normalize(
        edcpp::api::overflow::cpu::translation::cpu::release(internal_resource),
        internal_resource);
    result.success = internal.released;
    result.diagnostic = std::move(internal.diagnostic);
    if (internal.released) {
        resource = {};
    }
    return result;
}

} // namespace easyapi
