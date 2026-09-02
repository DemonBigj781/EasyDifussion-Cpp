#include "allocate.hpp"

#include "features/overflow/cuda/definition/gpu/allocate.hpp"

#include <string>
#include <utility>

namespace edcpp::api::overflow::cuda::translation::gpu {
namespace {

MemoryKind translate_kind(definition::gpu::NativeMemoryKind kind) noexcept {
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
    definition::gpu::NativeAllocation native, Tier tier,
    bool primary_out_of_memory) {
    Result result;
    result.allocated = native.allocated;
    result.overflowed = native.allocated && tier != Tier::active_vram;
    result.primary_out_of_memory = primary_out_of_memory;
    result.resource.backend = Backend::cuda;
    result.resource.tier = tier;
    result.resource.memory_kind = translate_kind(native.memory_kind);
    result.resource.host_handle = native.host_handle;
    result.resource.device_handle = native.device_handle;
    result.resource.size = native.size;
    result.resource.owner_device_index = native.owner_device_index;
    result.resource.access_device_index = native.access_device_index;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

void append_failure(std::string& failures, const std::string& message) {
    if (message.empty()) {
        return;
    }
    if (!failures.empty()) {
        failures += "; ";
    }
    failures += message;
}

} // namespace

Result allocate(const Request& request) {
    Result result;
    if (request.size == 0 || request.device_index < 0) {
        result.diagnostic = "CUDA overflow allocation requires a valid device and non-zero size";
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

    PlanInput plan_input;
    plan_input.active_backend = Backend::cuda;
    plan_input.active_device_index = request.device_index;
    plan_input.include_active_vram = !request.force_overflow;
    plan_input.managed_memory_available = capabilities.managed_memory;
    plan_input.mapped_host_available = capabilities.mapped_host_memory;
    for (const auto& native_device : capabilities.secondary_devices) {
        plan_input.secondary_devices.push_back(
            {Backend::cuda, native_device.device_index, native_device.free_memory});
    }

    bool primary_out_of_memory = false;
    std::string failures;
    for (const auto& candidate : plan(plan_input, request.size)) {
        if (candidate.backend != Backend::cuda) {
            append_failure(failures, "secondary GPU requires another backend translation");
            continue;
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
            case CandidateKind::gpu_zram:
            case CandidateKind::zram:
            case CandidateKind::swap:
                continue;
        }

        if (native.allocated) {
            auto translated = translate_allocation(
                std::move(native), candidate.tier, primary_out_of_memory);
            if (candidate.tier == Tier::active_vram) {
                translated.diagnostic = "allocated in active CUDA VRAM";
            } else if (candidate.tier == Tier::secondary_vram) {
                translated.diagnostic = "active CUDA VRAM unavailable; allocated in secondary GPU VRAM";
            } else if (candidate.kind == CandidateKind::managed) {
                translated.diagnostic = "CUDA VRAM unavailable; using cudaMallocManaged with CPU-preferred residency";
            } else if (candidate.kind == CandidateKind::mapped_host) {
                translated.diagnostic = "CUDA VRAM and managed allocation unavailable; using CUDA-mapped host RAM";
            }
            return translated;
        }

        append_failure(failures, native.diagnostic);
        if (candidate.tier == Tier::active_vram) {
            primary_out_of_memory = native.out_of_memory;
            if (!native.out_of_memory) {
                result.diagnostic = failures;
                return result;
            }
        }
    }

    result.primary_out_of_memory = primary_out_of_memory;
    result.diagnostic = failures.empty()
        ? "no implemented CUDA overflow tier is available"
        : failures;
    return result;
}

} // namespace edcpp::api::overflow::cuda::translation::gpu
