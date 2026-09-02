#pragma once

#include <cstdint>
#include <string>

namespace edcpp::api::load::model::cuda::definition::gpu {

struct NativeResult {
    bool loaded = false;
    void* handle = nullptr;
    std::uint64_t size = 0;
    int device_index = -1;
    std::string diagnostic;
};

NativeResult load(const void* data, std::uint64_t size, int device_index);

} // namespace edcpp::api::load::model::cuda::definition::gpu
