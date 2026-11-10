#include "features/load/common/devices/load.hpp"
#include "features/load/cpu/definition/cpu/model.hpp"

#include <utility>

namespace edcpp::api::load::cpu::translation::cpu {
namespace {

Result load_model(const Request& request) {
    auto native = definition::cpu::load_model(request.data, request.size, request.max_bytes);
    Result result;
    result.loaded = native.loaded;
    result.resource = {Backend::cpu, native.handle, native.size, 0, ResourceType::model};
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

void rollback(Resource& resource) noexcept {
    definition::cpu::discard_model(resource.native_handle);
    resource = {};
}

} // namespace

void ensure_registered() {
    // Function-local static initialization is thread-safe.
    static const bool registered = register_translation(Backend::cpu, ResourceType::model, &load_model, &rollback);
    (void) registered;
}

} // namespace edcpp::api::load::cpu::translation::cpu
