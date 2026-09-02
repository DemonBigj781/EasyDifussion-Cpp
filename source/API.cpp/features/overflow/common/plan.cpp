#include "overflow.hpp"

#include <algorithm>

namespace edcpp::api::overflow {

std::vector<Candidate> plan(const PlanInput& input, std::uint64_t size) {
    std::vector<Candidate> result;

    if (input.include_active_vram && input.active_device_index >= 0) {
        result.push_back(
            {input.active_backend, Tier::active_vram, CandidateKind::device,
             input.active_device_index});
    }

    auto secondary = input.secondary_devices;
    std::stable_sort(
        secondary.begin(), secondary.end(),
        [](const SecondaryDevice& left, const SecondaryDevice& right) {
            return left.free_memory > right.free_memory;
        });
    for (const auto& device : secondary) {
        if (device.device_index >= 0 && device.free_memory >= size) {
            result.push_back(
                {device.backend, Tier::secondary_vram, CandidateKind::device,
                 device.device_index});
        }
    }

    if (input.gpu_zram_available) {
        result.push_back(
            {Backend::none, Tier::gpu_zram, CandidateKind::gpu_zram, -1});
    }
    if (input.managed_memory_available) {
        result.push_back(
            {input.active_backend, Tier::system_ram, CandidateKind::managed,
             input.active_device_index});
    }
    if (input.mapped_host_available) {
        result.push_back(
            {input.active_backend, Tier::system_ram, CandidateKind::mapped_host,
             input.active_device_index});
    }
    if (input.zram_available) {
        result.push_back({Backend::none, Tier::zram, CandidateKind::zram, -1});
    }
    if (input.swap_available) {
        result.push_back({Backend::none, Tier::swap, CandidateKind::swap, -1});
    }

    return result;
}

} // namespace edcpp::api::overflow
