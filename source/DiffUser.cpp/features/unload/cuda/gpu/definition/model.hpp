#pragma once

#include <string>

namespace edcpp::api::unload::cuda::gpu::definition {

struct NativeResult {
    bool unloaded = false;
    std::string diagnostic{};
};

NativeResult unload_model(void* handle, int device_index) noexcept;

} // namespace edcpp::api::unload::cuda::gpu::definition
