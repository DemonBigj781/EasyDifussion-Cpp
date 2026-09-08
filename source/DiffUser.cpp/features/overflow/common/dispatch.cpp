#include "overflow.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace edcpp::api::overflow {
namespace {

constexpr std::size_t backend_slots =
    static_cast<std::size_t>(Backend::directml) + 1;

std::array<const Translation*, backend_slots>& translations() {
    static std::array<const Translation*, backend_slots> table{};
    return table;
}

} // namespace

bool register_translation(const Translation* translation) noexcept {
    if (translation == nullptr || translation->backend == Backend::none ||
        translation->allocate == nullptr || translation->release == nullptr) {
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
    const auto index = static_cast<std::size_t>(backend);
    return backend != Backend::none && index < backend_slots
        ? translations()[index]
        : nullptr;
}

Result allocate(Backend backend, const Request& request) {
    const Translation* translation = translation_for(backend);
    if (translation == nullptr) {
        Result result;
        result.diagnostic =
            "Common Overflow has no registered translation for the requested backend";
        return result;
    }
    return normalize(translation->allocate(request));
}

ReleaseResult release(Resource& resource) {
    const Translation* translation = translation_for(resource.backend);
    if (translation == nullptr) {
        return {
            false,
            "Common Overflow has no registered translation for the resource owner"};
    }
    ReleaseResult result =
        normalize(translation->release(resource), resource);
    if (result.released) {
        resource = {};
    }
    return result;
}

} // namespace edcpp::api::overflow
