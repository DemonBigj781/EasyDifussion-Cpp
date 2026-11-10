#pragma once

#include "features/load/common/devices/load.hpp"

#include <string>
#include <memory>

namespace edcpp::api::unload {

struct Result {
    Backend backend = Backend::none;
    bool unloaded = false;
    std::string diagnostic;
};

using TranslationFn = Result (*)(const load::Resource&);
// Pass a module lifetime guard for callbacks not linked for process lifetime.
// Immutable registration pins that guard; plugin unloading is not supported.
bool register_translation(Backend backend, load::ResourceType resource_type,
                          TranslationFn translation, std::shared_ptr<const void> module_guard = {}) noexcept;
Result unload_resource(load::Resource& resource);

using ModelTranslationFn = TranslationFn;
bool register_model_translation(Backend backend, ModelTranslationFn translation, std::shared_ptr<const void> module_guard = {}) noexcept;
Result unload_model(load::Resource& resource);
Result normalize(Result result, load::Resource& resource);

} // namespace edcpp::api::unload
