#pragma once

#include "common/api.hpp"

#include <array>
#include <cstddef>

namespace edcpp::api::attention::flash {

struct Request {
    // Opaque normalized hand-off. Translation owns native casts.
    const void* compute_params = nullptr;
    void* destination = nullptr;
};

struct Capabilities {
    bool forward = false;
    bool ggml_flash_attn_ext = false;
};

struct ValidationResult {
    bool ok = false;
    const char* message = nullptr;
};

using ValidateFn = ValidationResult (*)(const Request&) noexcept;
using ForwardFn = bool (*)(const Request&) noexcept;

struct Translation {
    Backend backend = Backend::none;
    const char* name = "unknown";
    Capabilities capabilities{};
    ValidateFn validate = nullptr;
    ForwardFn forward = nullptr;
};

inline std::array<const Translation*, 16>& registry() noexcept {
    static std::array<const Translation*, 16> translations{};
    return translations;
}

inline std::size_t backend_slot(Backend backend) noexcept {
    return static_cast<std::size_t>(backend);
}

inline bool register_translation(const Translation* translation) noexcept {
    if (translation == nullptr || translation->backend == Backend::none) {
        return false;
    }
    const auto slot = backend_slot(translation->backend);
    if (slot >= registry().size()) {
        return false;
    }
    registry()[slot] = translation;
    return true;
}

inline const Translation* translation_for(Backend backend) noexcept {
    const auto slot = backend_slot(backend);
    return slot < registry().size() ? registry()[slot] : nullptr;
}

inline Capabilities capabilities(Backend backend) noexcept {
    const auto* translation = translation_for(backend);
    return translation != nullptr ? translation->capabilities : Capabilities{};
}

inline ValidationResult validate(Backend backend, const Request& request) noexcept {
    const auto* translation = translation_for(backend);
    if (translation == nullptr) {
        return {false, "FlashAttention backend is not registered"};
    }
    if (translation->validate == nullptr) {
        return {false, "FlashAttention backend has no validator"};
    }
    return translation->validate(request);
}

inline bool forward(Backend backend, const Request& request) noexcept {
    const auto* translation = translation_for(backend);
    if (translation == nullptr || translation->forward == nullptr) {
        return false;
    }
    const auto validation = translation->validate != nullptr
        ? translation->validate(request)
        : ValidationResult{true, nullptr};
    return validation.ok && translation->forward(request);
}

} // namespace edcpp::api::attention::flash
