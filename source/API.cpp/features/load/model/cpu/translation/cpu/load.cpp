#include "load.hpp"

#include "features/load/model/cpu/definition/cpu/load.hpp"

#include <utility>

namespace edcpp::api::load::model::cpu::translation::cpu {

Result load(const Request& request) {
    auto native = definition::cpu::load(request.data, request.size);

    Result result;
    result.loaded = native.loaded;
    result.resource.backend = Backend::cpu;
    result.resource.native_handle = native.handle;
    result.resource.size = native.size;
    result.resource.device_index = 0;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

namespace {
const bool registered = edcpp::api::load::register_model_translation(Backend::cpu, &load);
}

} // namespace edcpp::api::load::model::cpu::translation::cpu
