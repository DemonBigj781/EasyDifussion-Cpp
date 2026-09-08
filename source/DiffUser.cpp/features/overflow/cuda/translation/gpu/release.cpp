#include "release.hpp"

#include "features/overflow/cuda/definition/gpu/release.hpp"

#include <utility>

namespace edcpp::api::overflow::cuda::translation::gpu {
namespace {

definition::gpu::NativeMemoryKind translate_kind(MemoryKind kind) noexcept {
    switch (kind) {
        case MemoryKind::cuda_device:
            return definition::gpu::NativeMemoryKind::device;
        case MemoryKind::cuda_managed:
            return definition::gpu::NativeMemoryKind::managed;
        case MemoryKind::cuda_mapped_host:
            return definition::gpu::NativeMemoryKind::mapped_host;
        case MemoryKind::none:
        case MemoryKind::cpu_heap:
            return definition::gpu::NativeMemoryKind::none;
    }
    return definition::gpu::NativeMemoryKind::none;
}

} // namespace

ReleaseResult release(const Resource& resource) noexcept {
    ReleaseResult result;
    if (resource.backend != Backend::cuda || !resource.loaded()) {
        result.diagnostic = "CUDA overflow release received an invalid or foreign resource";
        return result;
    }

    auto native = definition::gpu::release(
        translate_kind(resource.memory_kind), resource.host_handle,
        resource.device_handle, resource.owner_device_index);
    result.released = native.released;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

} // namespace edcpp::api::overflow::cuda::translation::gpu
