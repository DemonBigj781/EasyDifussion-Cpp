#include "unload.hpp"

namespace edcpp::api::unload {

Result normalize(Result result, load::Resource& resource) {
    if (!result.unloaded) {
        if (result.diagnostic.empty()) {
            result.diagnostic = "unload did not release the resource";
        }
        return result;
    }

    if (result.backend == Backend::none || result.backend != resource.backend) {
        result.unloaded = false;
        if (result.diagnostic.empty()) {
            result.diagnostic = "unload backend did not match the resource owner";
        }
        return result;
    }

    resource = {};
    return result;
}

} // namespace edcpp::api::unload
