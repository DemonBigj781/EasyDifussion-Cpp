#pragma once

#include "features/load/common/load.hpp"

#include <string>

namespace edcpp::api::unload {

struct Result {
    Backend backend = Backend::none;
    bool unloaded = false;
    std::string diagnostic;
};

// Public Common model-unload entry point. Resource ownership selects the backend.
Result model(load::Resource& resource);

Result normalize(Result result, load::Resource& resource);

} // namespace edcpp::api::unload
