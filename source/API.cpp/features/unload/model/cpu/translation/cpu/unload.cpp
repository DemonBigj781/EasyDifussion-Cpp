#include "unload.hpp"

#include "features/unload/model/cpu/definition/cpu/unload.hpp"

#include <utility>

namespace edcpp::api::unload::model::cpu::translation::cpu {

Result unload(const load::Resource& resource) {
    Result result;
    result.backend = Backend::cpu;
    if (resource.backend != Backend::cpu) {
        result.diagnostic = "CPU unload cannot release a resource owned by another backend";
        return result;
    }

    auto native = definition::cpu::unload(resource.native_handle);
    result.unloaded = native.unloaded;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

namespace {
[[maybe_unused]] const bool registered = edcpp::api::unload::register_model_translation(Backend::cpu, &unload);
}

} // namespace edcpp::api::unload::model::cpu::translation::cpu
