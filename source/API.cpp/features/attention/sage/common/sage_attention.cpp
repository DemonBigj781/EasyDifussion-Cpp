#include "features/attention/sage/common/sage_attention.hpp"

#include <array>
#include <cmath>

namespace edcpp::api::attention::sage {
namespace {

std::array<const Translation*, 16>& registry() noexcept {
    static std::array<const Translation*, 16> translations{};
    return translations;
}

std::size_t backend_slot(Backend backend) noexcept {
    return static_cast<std::size_t>(backend);
}

template <typename Tensor>
bool positive(const Tensor& tensor) noexcept {
    return tensor.head_dim > 0 && tensor.tokens > 0 && tensor.heads > 0 &&
           tensor.batch > 0;
}

Result validate_common(const SupportRequest& request) noexcept {
    if (!positive(request.query) || !positive(request.key) ||
        !positive(request.value) || !positive(request.output)) {
        return {false, "SageAttention tensor dimensions must be positive"};
    }
    if (request.query.head_dim != request.key.head_dim ||
        request.key.tokens != request.value.tokens ||
        request.key.heads != request.value.heads) {
        return {false, "SageAttention requires matching Q/K dimensions and K/V token/head counts"};
    }
    if (request.query.batch != request.key.batch ||
        request.query.batch != request.value.batch ||
        request.query.heads % request.key.heads != 0) {
        return {false, "SageAttention requires matching batches and grouped-query-compatible heads"};
    }
    if (request.output.head_dim != request.value.head_dim ||
        request.output.tokens != request.query.tokens ||
        request.output.heads != request.query.heads ||
        request.output.batch != request.query.batch) {
        return {false, "SageAttention output shape must match [V head_dim, Q tokens, Q heads, Q batch]"};
    }
    if (request.device.index < 0 || request.device.compute_major <= 0 ||
        request.device.compute_minor < 0) {
        return {false, "SageAttention device identity and compute capability are invalid"};
    }
    if (!std::isfinite(request.scale) || request.scale < 0.0f ||
        !std::isfinite(request.max_bias) || request.max_bias < 0.0f ||
        !std::isfinite(request.logit_softcap) ||
        request.logit_softcap < 0.0f) {
        return {false, "SageAttention parameters must be finite and non-negative"};
    }
    return {true, nullptr};
}

} // namespace

bool register_translation(const Translation* translation) noexcept {
    if (translation == nullptr || translation->backend == Backend::none ||
        translation->support == nullptr) {
        return false;
    }
    const auto slot = backend_slot(translation->backend);
    if (slot >= registry().size()) return false;
    registry()[slot] = translation;
    return true;
}

const Translation* translation_for(Backend backend) noexcept {
    const auto slot = backend_slot(backend);
    return slot < registry().size() ? registry()[slot] : nullptr;
}

Capabilities capabilities(Backend backend) noexcept {
    const auto* translation = translation_for(backend);
    return translation != nullptr ? translation->capabilities : Capabilities{};
}

Result support(Backend backend, const SupportRequest& request) noexcept {
    const Result common = validate_common(request);
    if (!common.ok) return common;
    const auto* translation = translation_for(backend);
    if (translation == nullptr) {
        return {false, "SageAttention backend is not registered"};
    }
    return translation->support(request);
}

} // namespace edcpp::api::attention::sage
