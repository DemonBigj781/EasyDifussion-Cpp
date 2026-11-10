#include "features/unload/common/devices/unload.hpp"
#include "features/unload/cpu/cpu/definition/model.hpp"

#include <utility>

namespace edcpp::api::unload::cpu::cpu::translation {
namespace {

Result unload_model(const load::Resource& resource) {
    Result result;
    result.backend = Backend::cpu;
    if (resource.backend != Backend::cpu) {
        result.diagnostic = "CPU unload cannot release a resource owned by another backend";
        return result;
    }
    auto native = definition::unload_model(resource.native_handle, resource.allocation_id);
    result.unloaded = native.unloaded;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

} // namespace

void ensure_registered() {
    // Function-local static initialization is thread-safe.
    static const bool registered = register_translation(Backend::cpu, load::ResourceType::model, &unload_model);
    (void) registered;
}

} // namespace edcpp::api::unload::cpu::cpu::translation
