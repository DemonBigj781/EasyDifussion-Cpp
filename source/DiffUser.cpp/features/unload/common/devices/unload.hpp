#pragma once

#include "features/load/common/devices/load.hpp"

#include <string>

namespace edcpp::api::unload {

struct Result {
    Backend backend = Backend::none;
    bool unloaded = false;
    std::string diagnostic;
};

using TranslationFn = Result (*)(const load::Resource&);
bool register_translation(Backend backend, load::ResourceType resource_type,
                          TranslationFn translation) noexcept;
Result unload_resource(load::Resource& resource);

using ModelTranslationFn = TranslationFn;
bool register_model_translation(Backend backend, ModelTranslationFn translation) noexcept;
Result unload_model(load::Resource& resource);
Result normalize(Result result, load::Resource& resource);

} // namespace edcpp::api::unload
