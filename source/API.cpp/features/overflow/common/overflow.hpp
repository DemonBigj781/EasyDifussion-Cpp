#pragma once

#include "common/api.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace edcpp::api::overflow {

enum class Tier : std::uint8_t {
    none = 0,
    active_vram,
    secondary_vram,
    gpu_zram,
    system_ram,
    zram,
    swap,
};

enum class MemoryKind : std::uint8_t {
    none = 0,
    cpu_heap,
    cuda_device,
    cuda_managed,
    cuda_mapped_host,
};

struct Request {
    const void* data = nullptr;
    std::uint64_t size = 0;
    int device_index = 0;
    bool force_overflow = false;
    std::uint64_t host_reserve = 1024ull * 1024ull * 1024ull;
};

struct Resource {
    Backend backend = Backend::none;
    Tier tier = Tier::none;
    MemoryKind memory_kind = MemoryKind::none;
    void* host_handle = nullptr;
    void* device_handle = nullptr;
    std::uint64_t size = 0;
    int owner_device_index = -1;
    int access_device_index = -1;

    bool loaded() const noexcept;
    bool gpu_accessible() const noexcept;
};

struct Result {
    bool allocated = false;
    bool overflowed = false;
    bool primary_out_of_memory = false;
    Resource resource;
    std::string diagnostic;
};

struct ReleaseResult {
    bool released = false;
    std::string diagnostic;
};

struct SecondaryDevice {
    Backend backend = Backend::none;
    int device_index = -1;
    std::uint64_t free_memory = 0;
};

struct PlanInput {
    Backend active_backend = Backend::none;
    int active_device_index = -1;
    bool include_active_vram = true;
    std::vector<SecondaryDevice> secondary_devices;
    bool gpu_zram_available = false;
    bool managed_memory_available = false;
    bool mapped_host_available = false;
    bool zram_available = false;
    bool swap_available = false;
};

enum class CandidateKind : std::uint8_t {
    none = 0,
    device,
    managed,
    mapped_host,
    gpu_zram,
    zram,
    swap,
};

struct Candidate {
    Backend backend = Backend::none;
    Tier tier = Tier::none;
    CandidateKind kind = CandidateKind::none;
    int device_index = -1;
};

std::vector<Candidate> plan(const PlanInput& input, std::uint64_t size);
Result normalize(Result result);
ReleaseResult normalize(ReleaseResult result, const Resource& resource);
const char* tier_name(Tier tier) noexcept;
const char* memory_kind_name(MemoryKind kind) noexcept;

} // namespace edcpp::api::overflow
