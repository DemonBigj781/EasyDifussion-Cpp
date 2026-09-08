#include "features/load/common/devices/load.hpp"
#include "features/load/cpu/definition/cpu/model.hpp"

#include <utility>

namespace edcpp::api::load::cpu::translation::cpu {
namespace {

Result load_model(const Request& request) {
    auto native = definition::cpu::load_model(request.data, request.size);
    Result result;
    result.loaded = native.loaded;
    result.resource = {Backend::cpu, native.handle, native.size, 0, ResourceType::model};
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

[[maybe_unused]] const bool registered =
    register_translation(Backend::cpu, ResourceType::model, &load_model);

} // namespace
} // namespace edcpp::api::load::cpu::translation::cpu
