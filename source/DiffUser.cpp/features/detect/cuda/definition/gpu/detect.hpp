#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace edcpp::api::detect::cuda::definition::gpu {

struct NativeDevice {
    int index = -1;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::uint64_t total_memory = 0;
    std::uint64_t free_memory = 0;
    bool available = false;
};

struct NativeResult {
    bool available = false;
    std::vector<NativeDevice> devices;
    std::string diagnostic;
};

// Native CUDA meaning: enumerate runtime-visible CUDA devices and snapshot
// their identity, compute capability, and current memory availability.
NativeResult detect();

} // namespace edcpp::api::detect::cuda::definition::gpu
