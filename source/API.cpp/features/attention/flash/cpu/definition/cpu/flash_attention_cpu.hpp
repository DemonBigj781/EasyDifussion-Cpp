#pragma once

#include "features/attention/common/ggml-attention-common.h"

namespace edcpp::api::attention::flash::cpu::definition {

inline bool supported(const ggml_tensor* destination) noexcept {
    return ggml_cpu_flash_compat_supported(destination);
}

inline bool forward(const ggml_compute_params* params, ggml_tensor* destination) noexcept {
    if (params == nullptr || !supported(destination)) {
        return false;
    }
    ggml_cpu_flash_compat(params, destination);
    return true;
}

} // namespace edcpp::api::attention::flash::cpu::definition
