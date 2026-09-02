#pragma once

#include "features/overflow/cuda/definition/gpu/allocate.hpp"

#include <string>

namespace edcpp::api::overflow::cuda::definition::gpu {

struct NativeRelease {
    bool released = false;
    std::string diagnostic;
};

NativeRelease release(
    NativeMemoryKind memory_kind, void* host_handle, void* device_handle,
    int owner_device_index) noexcept;

} // namespace edcpp::api::overflow::cuda::definition::gpu
