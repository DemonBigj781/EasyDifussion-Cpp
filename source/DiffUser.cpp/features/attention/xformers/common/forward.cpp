#include "xformers.hpp"

#include <array>
#include <cstddef>

namespace edcpp::api::attention::xformers {
namespace {
constexpr std::size_t backend_slots = static_cast<std::size_t>(Backend::directml) + 1;
std::array<const Translation*, backend_slots>& translations() {
    static std::array<const Translation*, backend_slots> table{};
    return table;
}
}

bool register_translation(const Translation* translation) noexcept {
    if (translation == nullptr || translation->backend == Backend::none) {
        return false;
    }
    const auto index = static_cast<std::size_t>(translation->backend);
    if (index >= backend_slots) {
        return false;
    }
    translations()[index] = translation;
    return true;
}

const Translation* translation_for(Backend backend) noexcept {
    if (backend == Backend::none) {
        return nullptr;
    }
    const auto index = static_cast<std::size_t>(backend);
    return index < backend_slots ? translations()[index] : nullptr;
}

Capabilities capabilities(Backend backend) noexcept {
    const Translation* translation = translation_for(backend);
    return translation == nullptr ? Capabilities{} : translation->capabilities;
}

ValidationResult validate(Backend backend, const AttentionRequest& request) {
    const Translation* translation = translation_for(backend);
    if (translation == nullptr || translation->validate == nullptr) {
        return {false, "No xFormers translation registered for requested backend"};
    }
    return translation->validate(request);
}

bool forward(Backend backend, const AttentionRequest& request) {
    const Translation* translation = translation_for(backend);
    if (translation == nullptr || translation->forward == nullptr) {
        return false;
    }
    const ValidationResult result = validate(backend, request);
    return result.ok && translation->forward(request);
}

} // namespace edcpp::api::attention::xformers
