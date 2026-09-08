#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace easyapi {

enum class OverflowTier : std::uint8_t {
    none = 0,
    active_vram,
    secondary_vram,
    gpu_zram,
    system_ram,
    zram,
    swap,
};

enum class OverflowStorage : std::uint8_t {
    none = 0,
    cpu_heap,
    cuda_device,
    cuda_managed,
    cuda_mapped_host,
};

struct OverflowRequest {
    const void* data = nullptr;
    std::size_t size = 0;
    int device_index = 0;
    bool force_overflow = false;
    std::size_t host_reserve = 1024ull * 1024ull * 1024ull;
};

struct OverflowResource {
    OverflowTier tier = OverflowTier::none;
    OverflowStorage storage = OverflowStorage::none;
    void* host_handle = nullptr;
    void* device_handle = nullptr;
    std::size_t size = 0;
    int owner_device_index = -1;
    int access_device_index = -1;

    bool loaded() const noexcept;
    bool gpu_accessible() const noexcept;
};

struct OverflowResult {
    bool success = false;
    bool overflowed = false;
    bool primary_out_of_memory = false;
    OverflowResource resource;
    std::string diagnostic;
};

struct OverflowReleaseResult {
    bool success = false;
    std::string diagnostic;
};

class CpuOverflowHandler final {
public:
    const char* name() const noexcept;
    OverflowResult allocate(const OverflowRequest& request) const;
    OverflowReleaseResult release(OverflowResource& resource) const;
};

class CudaOverflowHandler final {
public:
    const char* name() const noexcept;
    OverflowResult allocate(const OverflowRequest& request) const;
    OverflowReleaseResult release(OverflowResource& resource) const;
};

const char* overflow_tier_name(OverflowTier tier) noexcept;
const char* overflow_storage_name(OverflowStorage storage) noexcept;

} // namespace easyapi
