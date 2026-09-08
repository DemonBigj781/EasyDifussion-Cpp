#include "features/unload/common/devices/unload.hpp"
#include "features/unload/cuda/gpu/definition/model.hpp"

#include <utility>

namespace edcpp::api::unload::cuda::gpu::translation {
namespace {

Result unload_model(const load::Resource& resource) {
    Result result;
    result.backend = Backend::cuda;
    if (resource.backend != Backend::cuda) {
        result.diagnostic = "CUDA unload cannot release a resource owned by another backend";
        return result;
    }
    auto native = definition::unload_model(resource.native_handle, resource.device_index);
    result.unloaded = native.unloaded;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

[[maybe_unused]] const bool registered =
    register_translation(Backend::cuda, load::ResourceType::model, &unload_model);

} // namespace
} // namespace edcpp::api::unload::cuda::gpu::translation
