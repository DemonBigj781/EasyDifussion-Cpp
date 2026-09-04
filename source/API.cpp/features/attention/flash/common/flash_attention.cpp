#include "features/attention/flash/common/flash_attention.hpp"

#include <array>
#include <cmath>

namespace edcpp::api::attention::flash {
namespace {

std::array<const Translation*, 16>& registry() noexcept {
    static std::array<const Translation*, 16> translations{};
    return translations;
}

std::size_t backend_slot(Backend backend) noexcept { return static_cast<std::size_t>(backend); }

bool positive(const Tensor4D& tensor) noexcept {
    return tensor.head_dim > 0 && tensor.tokens > 0 && tensor.heads > 0 && tensor.batch > 0;
}

Result validate_common(const Request& request) noexcept {
    if (request.query.data == nullptr || request.key.data == nullptr || request.value.data == nullptr || request.output.data == nullptr) {
        return {false, "FlashAttention requires query, key, value, and output buffers"};
    }
    if (!positive(request.query) || !positive(request.key) || !positive(request.value) ||
        request.output.head_dim <= 0 || request.output.tokens <= 0 || request.output.heads <= 0 || request.output.batch <= 0) {
        return {false, "FlashAttention tensor dimensions must be positive"};
    }
    if (request.query.head_dim != request.key.head_dim || request.key.tokens != request.value.tokens) {
        return {false, "FlashAttention requires matching Q/K head dimensions and K/V token counts"};
    }
    if (request.query.batch != request.key.batch || request.query.batch != request.value.batch ||
        request.query.heads % request.key.heads != 0 || request.query.heads % request.value.heads != 0) {
        return {false, "FlashAttention requires compatible batch and grouped-query head counts"};
    }
    if (request.output.head_dim != request.value.head_dim || request.output.tokens != request.query.tokens ||
        request.output.heads != request.query.heads || request.output.batch != request.query.batch) {
        return {false, "FlashAttention output shape must match [V head_dim, Q tokens, Q heads, Q batch]"};
    }
    if (request.mask.data != nullptr && !positive(request.mask)) {
        return {false, "FlashAttention mask dimensions must be positive when a mask is supplied"};
    }
    if (!std::isfinite(request.scale) || request.scale < 0.0f || !std::isfinite(request.max_bias) ||
        request.max_bias < 0.0f || !std::isfinite(request.logit_softcap) || request.logit_softcap < 0.0f) {
        return {false, "FlashAttention scale, max bias, and logit softcap must be finite and non-negative"};
    }
    if (request.max_bias > 0.0f && request.mask.data == nullptr) {
        return {false, "FlashAttention max bias requires a mask"};
    }
    if (request.execution.thread_count <= 0 || request.execution.thread_index < 0 ||
        request.execution.thread_index >= request.execution.thread_count) {
        return {false, "FlashAttention execution thread index/count is invalid"};
    }
    return {true, nullptr};
}

} // namespace

bool register_translation(const Translation* translation) noexcept {
    if (translation == nullptr || translation->backend == Backend::none) return false;
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

Result validate(Backend backend, const Request& request) noexcept {
    const auto common = validate_common(request);
    if (!common.ok) return common;
    const auto* translation = translation_for(backend);
    if (translation == nullptr) return {false, "FlashAttention backend is not registered"};
    if (translation->validate == nullptr) return {false, "FlashAttention backend has no validator"};
    return translation->validate(request);
}

Result forward(Backend backend, const Request& request) noexcept {
    const auto validation = validate(backend, request);
    if (!validation.ok) return validation;
    const auto* translation = translation_for(backend);
    if (translation->forward == nullptr) return {false, "FlashAttention backend has no forward implementation"};
    return translation->forward(request);
}

} // namespace edcpp::api::attention::flash
