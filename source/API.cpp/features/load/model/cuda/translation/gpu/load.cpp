#include "load.hpp"

#include "features/load/model/cuda/definition/gpu/load.hpp"

#include <utility>

namespace edcpp::api::load::model::cuda::translation::gpu {

Result load(const Request& request) {
    auto native = definition::gpu::load(
        request.data, request.size, request.device_index);

    Result result;
    result.loaded = native.loaded;
    result.resource.backend = Backend::cuda;
    result.resource.native_handle = native.handle;
    result.resource.size = native.size;
    result.resource.device_index = native.device_index;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

} // namespace edcpp::api::load::model::cuda::translation::gpu
