#pragma once

#include <string>

namespace edcpp::api::unload::model::cuda::definition::gpu {

struct NativeResult {
    bool unloaded = false;
    std::string diagnostic;
};

NativeResult unload(void* handle, int device_index) noexcept;

} // namespace edcpp::api::unload::model::cuda::definition::gpu
