#include "features/load/common/devices/load.hpp"
#include "features/load/cpu/definition/cpu/model.hpp"

#include <utility>
#include <exception>

namespace edcpp::api::load::cpu::translation::cpu {
namespace {

void load_model(const Request& request, Result& result) {
    auto native = definition::cpu::load_model(request.data, request.size, request.max_bytes);
    result.loaded = native.loaded;
    result.resource = {Backend::cpu, native.handle, native.size, 0, ResourceType::model, native.allocation_id};
    result.diagnostic = std::move(native.diagnostic);
}

void rollback(Resource& resource) noexcept {
    if (!definition::cpu::discard_model(resource.native_handle, resource.allocation_id)) std::terminate();
    resource = {};
}

} // namespace

void ensure_registered() {
    // Function-local static initialization is thread-safe.
    static const bool registered = register_translation(Backend::cpu, ResourceType::model, &load_model, &rollback);
    (void) registered;
}

} // namespace edcpp::api::load::cpu::translation::cpu
