#include "xformers.hpp"

#include <cmath>
#include <cstddef>

namespace edcpp::api::attention::xformers::cuda::definition::gpu {
namespace {

NativeValidationResult invalid(const char* message) noexcept {
    return {false, message};
}

std::size_t element_size(NativeDType dtype) noexcept {
    return dtype == NativeDType::f32 ? sizeof(float) : sizeof(std::uint16_t);
}

} // namespace

NativeValidationResult validate_mask(const NativeRequest& request) noexcept {
    if (!std::isfinite(request.scale) || request.scale < 0.0f) {
        return invalid("CUDA xFormers scale must be finite and non-negative");
    }
    if (!std::isfinite(request.softcap) || request.softcap < 0.0f) {
        return invalid("CUDA xFormers softcap must be finite and non-negative");
    }
    if (!std::isfinite(request.alibi.max_bias) || request.alibi.max_bias < 0.0f) {
        return invalid("CUDA xFormers max bias must be finite and non-negative");
    }
    if (request.mask.data != nullptr) {
        if (request.mask.batch <= 0 || request.mask.heads <= 0 ||
            (request.mask.query_tokens != 1 &&
             request.mask.query_tokens < request.q.tokens) ||
            (request.mask.key_tokens != 1 &&
             request.mask.key_tokens < request.k.tokens) ||
            request.q.batch % request.mask.batch != 0 ||
            request.q.heads % request.mask.heads != 0 ||
            (request.mask.byte_strides[0] != 0 &&
             request.mask.byte_strides[0] != element_size(request.mask.dtype))) {
            return invalid("CUDA xFormers mask shape or row layout is not broadcast-compatible");
        }
    }
    if (request.alibi.enabled && request.alibi.slopes == nullptr &&
        (request.alibi.max_bias <= 0.0f || request.mask.data == nullptr)) {
        return invalid("CUDA xFormers ALiBi requires explicit slopes or a masked max-bias mode");
    }
    if (request.alibi.enabled && request.alibi.slopes != nullptr &&
        request.alibi.slope_count < request.q.heads) {
        return invalid("CUDA xFormers ALiBi requires one device-resident slope per Q head");
    }
    if (request.sinks.enabled &&
        (request.sinks.values == nullptr || request.sinks.value_count < request.q.heads)) {
        return invalid("CUDA xFormers attention sinks require one value per Q head");
    }
    return {true, nullptr};
}

} // namespace edcpp::api::attention::xformers::cuda::definition::gpu
