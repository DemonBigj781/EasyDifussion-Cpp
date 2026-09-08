#include "load.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace edcpp::api::load {
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

Result load_model(Backend backend, const Request& request) {
    const auto index = static_cast<std::size_t>(backend);
    if (backend == Backend::none || index >= backend_slots || model_translations()[index] == nullptr) {
        Result result;
        result.diagnostic = "Common model load has no registered translation for the requested backend";
        return normalize(std::move(result));
    }
    return normalize(model_translations()[index](request));
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
