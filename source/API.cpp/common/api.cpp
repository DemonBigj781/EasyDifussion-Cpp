#include "api.hpp"

namespace edcpp::api {

const char* backend_name(Backend backend) noexcept {
    switch (backend) {
        case Backend::cpu: return "cpu";
        case Backend::cuda: return "cuda";
        case Backend::rocm: return "rocm";
        case Backend::oneapi: return "oneapi";
        case Backend::vulkan: return "vulkan";
        case Backend::none:
        default: return "none";
    }
}

const char* operation_name(Operation operation) noexcept {
    switch (operation) {
        case Operation::attention: return "attention";
        case Operation::flash_attention: return "flash_attention";
        case Operation::sage_attention: return "sage_attention";
        case Operation::xformers_attention: return "xformers_attention";
        case Operation::matmul: return "matmul";
        case Operation::convolution: return "convolution";
        case Operation::normalization: return "normalization";
        case Operation::sampling: return "sampling";
        case Operation::unknown:
        default: return "unknown";
    }
}

} // namespace edcpp::api
