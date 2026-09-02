#include "api/overflow.hpp"

#include "features/overflow/common/overflow.hpp"
#include "features/overflow/cuda/translation/gpu/allocate.hpp"
#include "features/overflow/cuda/translation/gpu/release.hpp"

#include <cstdint>
#include <utility>

namespace easyapi {
namespace {

OverflowTier cuda_to_public(edcpp::api::overflow::Tier tier) noexcept {
    return static_cast<OverflowTier>(tier);
}

OverflowStorage cuda_to_public(edcpp::api::overflow::MemoryKind kind) noexcept {
    return static_cast<OverflowStorage>(kind);
}

edcpp::api::overflow::Tier cuda_to_internal(OverflowTier tier) noexcept {
    return static_cast<edcpp::api::overflow::Tier>(tier);
}

edcpp::api::overflow::MemoryKind cuda_to_internal(OverflowStorage storage) noexcept {
    return static_cast<edcpp::api::overflow::MemoryKind>(storage);
}

OverflowResource cuda_to_public(const edcpp::api::overflow::Resource& resource) {
    OverflowResource result;
    result.tier = cuda_to_public(resource.tier);
    result.storage = cuda_to_public(resource.memory_kind);
    result.host_handle = resource.host_handle;
    result.device_handle = resource.device_handle;
    result.size = static_cast<std::size_t>(resource.size);
    result.owner_device_index = resource.owner_device_index;
    result.access_device_index = resource.access_device_index;
    return result;
}

edcpp::api::overflow::Resource cuda_to_internal(
    const OverflowResource& resource) {
    edcpp::api::overflow::Resource result;
    result.backend = edcpp::api::Backend::cuda;
    result.tier = cuda_to_internal(resource.tier);
    result.memory_kind = cuda_to_internal(resource.storage);
    result.host_handle = resource.host_handle;
    result.device_handle = resource.device_handle;
    result.size = static_cast<std::uint64_t>(resource.size);
    result.owner_device_index = resource.owner_device_index;
    result.access_device_index = resource.access_device_index;
    return result;
}

} // namespace

const char* CudaOverflowHandler::name() const noexcept {
    return "cuda";
}

OverflowResult CudaOverflowHandler::allocate(const OverflowRequest& request) const {
    edcpp::api::overflow::Request internal_request;
    internal_request.data = request.data;
    internal_request.size = static_cast<std::uint64_t>(request.size);
    internal_request.device_index = request.device_index;
    internal_request.force_overflow = request.force_overflow;
    internal_request.host_reserve = static_cast<std::uint64_t>(request.host_reserve);

    auto internal = edcpp::api::overflow::normalize(
        edcpp::api::overflow::cuda::translation::gpu::allocate(internal_request));
    OverflowResult result;
    result.success = internal.allocated;
    result.overflowed = internal.overflowed;
    result.primary_out_of_memory = internal.primary_out_of_memory;
    result.diagnostic = std::move(internal.diagnostic);
    if (internal.allocated) {
        result.resource = cuda_to_public(internal.resource);
    }
    return result;
}

OverflowReleaseResult CudaOverflowHandler::release(OverflowResource& resource) const {
    OverflowReleaseResult result;
    if (resource.storage != OverflowStorage::cuda_device &&
        resource.storage != OverflowStorage::cuda_managed &&
        resource.storage != OverflowStorage::cuda_mapped_host) {
        result.diagnostic = "CUDA overflow cannot release a resource owned by another allocator";
        return result;
    }

    auto internal_resource = cuda_to_internal(resource);
    auto internal = edcpp::api::overflow::normalize(
        edcpp::api::overflow::cuda::translation::gpu::release(internal_resource),
        internal_resource);
    result.success = internal.released;
    result.diagnostic = std::move(internal.diagnostic);
    if (internal.released) {
        resource = {};
    }
    return result;
}

} // namespace easyapi
