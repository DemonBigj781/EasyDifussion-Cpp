#include "api/model_lifecycle.hpp"

#include "features/load/common/load.hpp"
#include "features/unload/common/unload.hpp"

#include <cstdint>
#include <utility>

namespace easyapi {
namespace {

ModelResource to_public(const edcpp::api::load::Resource& resource) {
    ModelResource result;
    result.storage = ModelStorage::cpu;
    result.native_handle = resource.native_handle;
    result.size = static_cast<std::size_t>(resource.size);
    result.device_index = resource.device_index;
    return result;
}

edcpp::api::load::Resource to_internal(const ModelResource& resource) {
    edcpp::api::load::Resource result;
    result.backend = edcpp::api::Backend::cpu;
    result.native_handle = resource.native_handle;
    result.size = static_cast<std::uint64_t>(resource.size);
    result.device_index = resource.device_index;
    return result;
}

} // namespace

const char* CpuModelLifecycleHandler::name() const noexcept {
    return "cpu";
}

LoadModelResult CpuModelLifecycleHandler::load(
    const void* data, std::size_t size, int device_index) const {
    (void) device_index;

    edcpp::api::load::Request request;
    request.data = data;
    request.size = static_cast<std::uint64_t>(size);
    request.device_index = 0;

    auto internal = edcpp::api::load::model(edcpp::api::Backend::cpu, request);

    LoadModelResult result;
    result.success = internal.loaded;
    result.diagnostic = std::move(internal.diagnostic);
    if (internal.loaded) {
        result.resource = to_public(internal.resource);
    }
    return result;
}

UnloadModelResult CpuModelLifecycleHandler::unload(ModelResource& resource) const {
    UnloadModelResult result;
    if (resource.storage != ModelStorage::cpu) {
        result.diagnostic = "CPU lifecycle cannot unload a resource owned by another backend";
        return result;
    }

    auto internal_resource = to_internal(resource);
    auto internal = edcpp::api::unload::model(internal_resource);

    result.success = internal.unloaded;
    result.diagnostic = std::move(internal.diagnostic);
    if (internal.unloaded) {
        resource = {};
    }
    return result;
}

} // namespace easyapi
