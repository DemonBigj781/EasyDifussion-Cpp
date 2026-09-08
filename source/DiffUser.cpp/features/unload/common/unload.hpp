#pragma once

#include "features/load/common/load.hpp"

#include <string>

namespace edcpp::api::unload {

struct Result {
    Backend backend = Backend::none;
    bool unloaded = false;
    std::string diagnostic;
};

using ModelTranslationFn = Result (*)(const load::Resource&);
bool register_model_translation(Backend backend, ModelTranslationFn translation) noexcept;
Result unload_model(load::Resource& resource);
Result normalize(Result result, load::Resource& resource);

} // namespace edcpp::api::unload
