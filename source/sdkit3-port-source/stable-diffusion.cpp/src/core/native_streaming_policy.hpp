#pragma once

#include <cstddef>
#include <initializer_list>

namespace sd_streaming {
inline const char* validate_activation(bool cpu_parameters,
                                       size_t compute_devices,
                                       bool accelerator) {
    if (!cpu_parameters) return "requires CPU-resident diffusion weights (enable CPU offload)";
    if (compute_devices != 1 || !accelerator) return "requires exactly one accelerator compute device";
    return nullptr;
}

inline size_t reported_streaming_headroom(size_t free_bytes, size_t total_bytes,
                                           size_t library_cap = 0) {
    constexpr size_t reserve = 512ull * 1024 * 1024;
    // A shared-memory or unsupported provider can produce impossible values.
    if (total_bytes == 0 || free_bytes > total_bytes || free_bytes <= reserve) return 0;
    const size_t headroom = free_bytes - reserve;
    return library_cap != 0 && library_cap < headroom ? library_cap : headroom;
}

inline bool fits_budget(size_t budget_bytes, std::initializer_list<size_t> estimated_parts) {
    for (const auto part : estimated_parts) {
        if (part > budget_bytes) return false;
        budget_bytes -= part;
    }
    return true;
}
}
