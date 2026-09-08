#include "load.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace edcpp::api::load {
namespace {
constexpr std::size_t backend_slots = static_cast<std::size_t>(Backend::directml) + 1;
constexpr std::size_t resource_slots = static_cast<std::size_t>(ResourceType::count);
using TranslationTable = std::array<std::array<TranslationFn, resource_slots>, backend_slots>;
TranslationTable& translations() {
    static TranslationTable table{};
    return table;
}
}

bool register_translation(Backend backend, ResourceType resource_type,
                          TranslationFn translation) noexcept {
    const auto backend_index = static_cast<std::size_t>(backend);
    const auto resource_index = static_cast<std::size_t>(resource_type);
    if (backend == Backend::none || backend_index >= backend_slots ||
        resource_index >= resource_slots || translation == nullptr) return false;
    translations()[backend_index][resource_index] = translation;
    return true;
}

Result load_resource(Backend backend, const Request& request) {
    const auto backend_index = static_cast<std::size_t>(backend);
    const auto resource_index = static_cast<std::size_t>(request.resource_type);
    if (backend == Backend::none || backend_index >= backend_slots ||
        resource_index >= resource_slots ||
        translations()[backend_index][resource_index] == nullptr) {
        Result result;
        result.diagnostic = "Common load has no registered translation for the requested backend and resource type";
        return normalize(std::move(result));
    }
    return normalize(translations()[backend_index][resource_index](request));
}

bool register_model_translation(Backend backend, ModelTranslationFn translation) noexcept {
    return register_translation(backend, ResourceType::model, translation);
}

Result load_model(Backend backend, const Request& request) {
    Request model_request = request;
    model_request.resource_type = ResourceType::model;
    return load_resource(backend, model_request);
}

Result normalize(Result result) {
    const bool valid_resource = result.resource.backend != Backend::none &&
        result.resource.native_handle != nullptr && result.resource.size != 0 &&
        result.resource.device_index >= 0;
    if (!result.loaded || !valid_resource) {
        result.loaded = false;
        result.resource = {};
        if (result.diagnostic.empty()) result.diagnostic = "load did not produce a valid owned resource";
    }
    return result;
}

} // namespace edcpp::api::load
