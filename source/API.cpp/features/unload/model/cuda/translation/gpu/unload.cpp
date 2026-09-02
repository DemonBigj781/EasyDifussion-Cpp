#include "unload.hpp"

#include "features/unload/model/cuda/definition/gpu/unload.hpp"

#include <utility>

namespace edcpp::api::unload::model::cuda::translation::gpu {

Result unload(const load::Resource& resource) {
    Result result;
    result.backend = Backend::cuda;
    if (resource.backend != Backend::cuda) {
        result.diagnostic = "CUDA unload cannot release a resource owned by another backend";
        return result;
    }

    auto native = definition::gpu::unload(resource.native_handle, resource.device_index);
    result.unloaded = native.unloaded;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

namespace {
[[maybe_unused]] const bool registered = edcpp::api::unload::register_model_translation(Backend::cuda, &unload);
}

} // namespace edcpp::api::unload::model::cuda::translation::gpu
