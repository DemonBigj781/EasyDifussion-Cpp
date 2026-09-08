#include "features/load/common/devices/load.hpp"
#include "features/load/cuda/definition/gpu/model.hpp"

#include <utility>

namespace edcpp::api::load::cuda::translation::gpu {
namespace {

Result load_model(const Request& request) {
    auto native = definition::gpu::load_model(request.data, request.size, request.device_index);
    Result result;
    result.loaded = native.loaded;
    result.resource = {Backend::cuda, native.handle, native.size,
                       native.device_index, ResourceType::model};
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

[[maybe_unused]] const bool registered =
    register_translation(Backend::cuda, ResourceType::model, &load_model);

} // namespace
} // namespace edcpp::api::load::cuda::translation::gpu
