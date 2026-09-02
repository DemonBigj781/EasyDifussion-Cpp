#include "load.hpp"

namespace edcpp::api::load {

Result normalize(Result result) {
    const bool valid_resource =
        result.resource.backend != Backend::none &&
        result.resource.native_handle != nullptr &&
        result.resource.size != 0 &&
        result.resource.device_index >= 0;

    if (!result.loaded || !valid_resource) {
        result.loaded = false;
        result.resource = {};
        if (result.diagnostic.empty()) {
            result.diagnostic = "load did not produce a valid owned resource";
        }
    }

    return result;
}

} // namespace edcpp::api::load
