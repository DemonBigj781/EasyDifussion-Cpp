#include "unload.hpp"

#include "features/unload/model/cpu/translation/cpu/unload.hpp"
#include "features/unload/model/cuda/translation/gpu/unload.hpp"

namespace edcpp::api::unload {

Result model(load::Resource& resource) {
    switch (resource.backend) {
        case Backend::cpu:
            return normalize(model::cpu::translation::cpu::unload(resource), resource);
        case Backend::cuda:
            return normalize(model::cuda::translation::gpu::unload(resource), resource);
        default: {
            Result result;
            result.backend = resource.backend;
            result.diagnostic = "Common model unload has no wired translation for the resource backend";
            return normalize(std::move(result), resource);
        }
    }
}

Result normalize(Result result, load::Resource& resource) {
    if (!result.unloaded) {
        if (result.diagnostic.empty()) {
            result.diagnostic = "unload did not release the resource";
        }
        return result;
    }

    if (result.backend == Backend::none || result.backend != resource.backend) {
        result.unloaded = false;
        if (result.diagnostic.empty()) {
            result.diagnostic = "unload backend did not match the resource owner";
        }
        return result;
    }

    resource = {};
    return result;
}

} // namespace edcpp::api::unload
