#include "allocate.hpp"
#include "release.hpp"

#include "features/overflow/cuda/definition/gpu/allocate.hpp"

#include <utility>

namespace edcpp::api::overflow::cuda::translation::gpu {
namespace {

MemoryKind translate_kind(
    definition::gpu::NativeMemoryKind kind) noexcept {
    using Native = definition::gpu::NativeMemoryKind;
    switch (kind) {
        case Native::device: return MemoryKind::cuda_device;
        case Native::managed: return MemoryKind::cuda_managed;
        case Native::mapped_host: return MemoryKind::cuda_mapped_host;
        case Native::none: return MemoryKind::none;
    }
    return MemoryKind::none;
}

Result translate_allocation(
    definition::gpu::NativeAllocation native,
    const Candidate& candidate) {
    Result result;
    result.allocated = native.allocated;
    result.overflowed =
        native.allocated && candidate.tier != Tier::active_vram;
    result.primary_out_of_memory =
        candidate.tier == Tier::active_vram && native.out_of_memory;
    result.resource.backend = Backend::cuda;
    result.resource.tier = candidate.tier;
    result.resource.memory_kind = translate_kind(native.memory_kind);
    result.resource.host_handle = native.host_handle;
    result.resource.device_handle = native.device_handle;
    result.resource.size = native.size;
    result.resource.owner_device_index = native.owner_device_index;
    result.resource.access_device_index = native.access_device_index;
    result.diagnostic = std::move(native.diagnostic);

    if (result.allocated) {
        if (candidate.tier == Tier::active_vram) {
            result.diagnostic = "allocated in active CUDA VRAM";
        } else if (candidate.tier == Tier::secondary_vram) {
            result.diagnostic =
                "active CUDA VRAM unavailable; allocated in secondary GPU VRAM";
        } else if (candidate.kind == CandidateKind::managed) {
            result.diagnostic =
                "CUDA VRAM unavailable; using cudaMallocManaged with CPU-preferred residency";
        } else if (candidate.kind == CandidateKind::mapped_host) {
            result.diagnostic =
                "CUDA VRAM and managed allocation unavailable; using CUDA-mapped host RAM";
        }
    }
    return result;
}

} // namespace

PlanningResult prepare(const Request& request) {
    PlanningResult result;
    if (request.size == 0 || request.device_index < 0) {
        result.diagnostic =
            "CUDA overflow allocation requires a valid device and non-zero size";
        return result;
    }

    const auto capabilities =
        definition::gpu::detect_capabilities(request.device_index);
    if (!capabilities.available) {
        result.diagnostic = capabilities.diagnostic.empty()
            ? "CUDA overflow capabilities are unavailable"
            : capabilities.diagnostic;
        return result;
    }

    result.ready = true;
    result.input.active_backend = Backend::cuda;
    result.input.active_device_index = request.device_index;
    result.input.include_active_vram = !request.force_overflow;
    result.input.managed_memory_available =
        capabilities.managed_memory;
    result.input.mapped_host_available =
        capabilities.mapped_host_memory;
    for (const auto& native_device : capabilities.secondary_devices) {
        result.input.secondary_devices.push_back(
            {Backend::cuda, native_device.device_index,
             native_device.free_memory});
    }
    return result;
}

Result allocate(
    const Request& request, const Candidate& candidate) {
    if (candidate.backend != Backend::cuda) {
        Result result;
        result.diagnostic =
            "CUDA overflow translation received a non-CUDA Common candidate";
        return result;
    }

    definition::gpu::NativeAllocation native;
    switch (candidate.kind) {
        case CandidateKind::device:
            native = definition::gpu::allocate_device(
                request.data, request.size, candidate.device_index);
            break;
        case CandidateKind::managed:
            native = definition::gpu::allocate_managed(
                request.data, request.size, request.device_index,
                request.host_reserve);
            break;
        case CandidateKind::mapped_host:
            native = definition::gpu::allocate_mapped_host(
                request.data, request.size, request.device_index,
                request.host_reserve);
            break;
        case CandidateKind::none:
        case CandidateKind::host:
        case CandidateKind::gpu_zram:
        case CandidateKind::zram:
        case CandidateKind::swap: {
            Result result;
            result.diagnostic =
                "CUDA overflow translation received an unsupported Common candidate";
            return result;
        }
    }
    return translate_allocation(std::move(native), candidate);
}

namespace {

const Translation cuda_translation = {
    Backend::cuda,
    "cuda",
    &prepare,
    &allocate,
    &release,
};

[[maybe_unused]] const bool registered =
    register_translation(&cuda_translation);

} // namespace

} // namespace edcpp::api::overflow::cuda::translation::gpu
