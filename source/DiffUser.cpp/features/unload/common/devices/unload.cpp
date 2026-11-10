#include "unload.hpp"

#include <array>
#include <cstddef>

namespace edcpp::api::unload {
namespace {
constexpr std::size_t backend_slots = static_cast<std::size_t>(Backend::directml) + 1;
constexpr std::size_t resource_slots = static_cast<std::size_t>(load::ResourceType::count);
using TranslationTable = std::array<std::array<TranslationFn, resource_slots>, backend_slots>;
TranslationTable& translations() {
    static TranslationTable table{};
    return table;
}
}

bool register_translation(Backend backend, load::ResourceType resource_type,
                          TranslationFn translation) noexcept {
    const auto backend_index = static_cast<std::size_t>(backend);
    const auto resource_index = static_cast<std::size_t>(resource_type);
    if (backend == Backend::none || backend_index >= backend_slots ||
        resource_index >= resource_slots || translation == nullptr) return false;
    translations()[backend_index][resource_index] = translation;
    return true;
}

Result unload_resource(load::Resource& resource) {
    const auto backend_index = static_cast<std::size_t>(resource.backend);
    const auto resource_index = static_cast<std::size_t>(resource.resource_type);
    if (resource.backend == Backend::none || backend_index >= backend_slots ||
        resource_index >= resource_slots ||
        translations()[backend_index][resource_index] == nullptr) {
        Result result;
        result.backend = resource.backend;
        result.diagnostic = "Common unload has no registered translation for the resource backend and type";
        return normalize(result, resource);
    }
    return normalize(translations()[backend_index][resource_index](resource), resource);
}

bool register_model_translation(Backend backend, ModelTranslationFn translation) noexcept {
    return register_translation(backend, load::ResourceType::model, translation);
}

Result unload_model(load::Resource& resource) {
    if (resource.resource_type != load::ResourceType::model) {
        Result result;
        result.backend = resource.backend;
        result.diagnostic = "Common model unload received a non-model resource";
        return normalize(result, resource);
    }
    return unload_resource(resource);
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
