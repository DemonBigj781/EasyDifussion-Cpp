#include "load.hpp"

#include "features/load/model/cpu/translation/cpu/load.hpp"
#include "features/load/model/cuda/translation/gpu/load.hpp"

#include <utility>

namespace edcpp::api::load {

Result model(Backend backend, const Request& request) {
    switch (backend) {
        case Backend::cpu:
            return normalize(model::cpu::translation::cpu::load(request));
        case Backend::cuda:
            return normalize(model::cuda::translation::gpu::load(request));
        default: {
            Result result;
            result.diagnostic = "Common model load has no wired translation for the requested backend";
            return normalize(std::move(result));
        }
    }
}

Result normalize(Result result) {
    const bool valid_resource =
        result.resource.backend != Backend::none &&
        result.resource.native_handle != nullptr &&
        result.resource.size != 0 &&
        result.resource.device_index >= 0;

    if (!result.loaded || !valid_resource) {
        result.loaded = false;
        result.resource = {};
        if (result.diagnostic.empty()) {
            result.diagnostic = "load did not produce a valid owned resource";
        }
    }

    return result;
}

} // namespace edcpp::api::load
