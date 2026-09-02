#include "unload.hpp"

#include <array>
#include <cstddef>

namespace edcpp::api::unload {
namespace {
constexpr std::size_t backend_slots = static_cast<std::size_t>(Backend::directml) + 1;
std::array<ModelTranslationFn, backend_slots>& model_translations() {
    static std::array<ModelTranslationFn, backend_slots> table{};
    return table;
}
}

bool register_model_translation(Backend backend, ModelTranslationFn translation) noexcept {
    const auto index = static_cast<std::size_t>(backend);
    if (backend == Backend::none || index >= backend_slots || translation == nullptr) return false;
    model_translations()[index] = translation;
    return true;
}

Result unload_model(load::Resource& resource) {
    const auto index = static_cast<std::size_t>(resource.backend);
    if (resource.backend == Backend::none || index >= backend_slots || model_translations()[index] == nullptr) {
        Result result;
        result.backend = resource.backend;
        result.diagnostic = "Common model unload has no registered translation for the resource backend";
        return normalize(result, resource);
    }
    return normalize(model_translations()[index](resource), resource);
}

Result normalize(Result result, load::Resource& resource) {
    if (!result.unloaded) {
        if (result.diagnostic.empty()) result.diagnostic = "unload did not release the resource";
        return result;
    }
    if (result.backend == Backend::none || result.backend != resource.backend) {
        result.unloaded = false;
        if (result.diagnostic.empty()) result.diagnostic = "unload backend did not match the resource owner";
        return result;
    }
    resource = {};
    return result;
}

} // namespace edcpp::api::unload
