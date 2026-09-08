#include "api/model_lifecycle.hpp"

#include "features/load/common/load.hpp"
#include "features/unload/common/unload.hpp"

#include <cstdint>
#include <utility>

namespace easyapi {
namespace {

ModelResource to_public(const edcpp::api::load::Resource& resource) {
    ModelResource result;
    result.storage = ModelStorage::cuda;
    result.native_handle = resource.native_handle;
    result.size = static_cast<std::size_t>(resource.size);
    result.device_index = resource.device_index;
    return result;
}

edcpp::api::load::Resource to_internal(const ModelResource& resource) {
    edcpp::api::load::Resource result;
    result.backend = edcpp::api::Backend::cuda;
    result.native_handle = resource.native_handle;
    result.size = static_cast<std::uint64_t>(resource.size);
    result.device_index = resource.device_index;
    return result;
}

} // namespace

const char* CudaModelLifecycleHandler::name() const noexcept { return "cuda"; }

LoadModelResult CudaModelLifecycleHandler::load(const void* data, std::size_t size, int device_index) const {
    edcpp::api::load::Request request;
    request.data = data;
    request.size = static_cast<std::uint64_t>(size);
    request.device_index = device_index;

    auto internal = edcpp::api::load::load_model(edcpp::api::Backend::cuda, request);
    LoadModelResult result;
    result.success = internal.loaded;
    result.diagnostic = std::move(internal.diagnostic);
    if (internal.loaded) result.resource = to_public(internal.resource);
    return result;
}

UnloadModelResult CudaModelLifecycleHandler::unload(ModelResource& resource) const {
    UnloadModelResult result;
    if (resource.storage != ModelStorage::cuda) {
        result.diagnostic = "CUDA lifecycle cannot unload a resource owned by another backend";
        return result;
    }
    auto internal_resource = to_internal(resource);
    auto internal = edcpp::api::unload::unload_model(internal_resource);
    result.success = internal.unloaded;
    result.diagnostic = std::move(internal.diagnostic);
    if (internal.unloaded) resource = {};
    return result;
}

} // namespace easyapi
