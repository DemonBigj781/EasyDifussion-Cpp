#include "xformers.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace edcpp::api::attention::xformers::cpu::definition::cpu {
namespace {

NativeResult invalid(const char* message) noexcept {
    return {false, message};
}

bool broadcast_dimension(
    std::int64_t value, std::int64_t expected) noexcept {
    return value == 1 || value == expected;
}

template <typename Tensor>
bool contiguous(const Tensor& tensor) noexcept {
    return tensor.byte_strides == std::array<std::size_t, 4>{};
}

} // namespace

NativeCapabilities capabilities() noexcept {
    NativeCapabilities result;
    result.forward = true;
    result.separate_stages = true;
    result.additive_mask = true;
    result.causal_mask = true;
    result.alibi = true;
    result.softcap = true;
    result.attention_sinks = false;
    result.gqa = true;
    result.mqa = true;
    result.f32 = true;
    result.f16 = false;
    result.bf16 = false;
    return result;
}

NativeResult validate(const NativeRequest& request) noexcept {
    if (request.dtype != NativeDType::f32 ||
        request.q.dtype != NativeDType::f32 ||
        request.k.dtype != NativeDType::f32 ||
        request.v.dtype != NativeDType::f32 ||
        request.out.dtype != NativeDType::f32) {
        return invalid("CPU xFormers currently supports F32 only");
    }
    if (!contiguous(request.q) || !contiguous(request.k) ||
        !contiguous(request.v) || !contiguous(request.out)) {
        return invalid("CPU xFormers currently requires contiguous tensors");
    }
    if (request.q.data == nullptr || request.k.data == nullptr ||
        request.v.data == nullptr || request.out.data == nullptr) {
        return invalid(
            "CPU xFormers requires non-null Q, K, V, and output buffers");
    }
    if (request.q.batch <= 0 || request.q.heads <= 0 ||
        request.q.tokens <= 0 || request.q.head_dim <= 0 ||
        request.k.batch <= 0 || request.k.heads <= 0 ||
        request.k.tokens <= 0 || request.k.head_dim <= 0 ||
        request.v.batch <= 0 || request.v.heads <= 0 ||
        request.v.tokens <= 0 || request.v.head_dim <= 0) {
        return invalid("CPU xFormers tensor dimensions must be positive");
    }
    if (request.q.batch != request.k.batch ||
        request.q.batch != request.v.batch) {
        return invalid(
            "CPU xFormers currently requires matching Q/K/V batch counts");
    }
    if (request.q.head_dim != request.k.head_dim) {
        return invalid(
            "CPU xFormers requires matching Q and K head dimensions");
    }
    if (request.k.tokens != request.v.tokens) {
        return invalid(
            "CPU xFormers requires matching K and V token counts");
    }
    if (request.q.heads % request.k.heads != 0 ||
        request.q.heads % request.v.heads != 0) {
        return invalid(
            "CPU xFormers requires K/V head counts to divide Q head count");
    }
    if (request.out.batch != request.q.batch ||
        request.out.heads != request.q.heads ||
        request.out.tokens != request.q.tokens ||
        request.out.head_dim != request.v.head_dim) {
        return invalid(
            "CPU xFormers output shape must be [Q batch, Q heads, Q tokens, V head_dim]");
    }
    if (!std::isfinite(request.scale) || request.scale < 0.0f) {
        return invalid(
            "CPU xFormers scale must be finite and non-negative");
    }
    if (!std::isfinite(request.softcap) || request.softcap < 0.0f) {
        return invalid(
            "CPU xFormers softcap must be finite and non-negative");
    }
    if (request.mask.data != nullptr &&
        (request.mask.dtype != NativeDType::f32 ||
         !contiguous(request.mask) ||
         !broadcast_dimension(request.mask.batch, request.q.batch) ||
         !broadcast_dimension(request.mask.heads, request.q.heads) ||
         !broadcast_dimension(
             request.mask.query_tokens, request.q.tokens) ||
         !broadcast_dimension(
             request.mask.key_tokens, request.k.tokens))) {
        return invalid(
            "CPU xFormers mask dimensions must be 1 or match the corresponding attention dimension");
    }
    if (request.alibi.enabled &&
        (request.alibi.slopes == nullptr ||
         request.alibi.slope_count < request.q.heads)) {
        return invalid(
            "CPU xFormers ALiBi requires one slope per Q head");
    }
    if (request.alibi.max_bias != 0.0f) {
        return invalid(
            "CPU xFormers does not support max-bias mask scaling");
    }
    if (request.sinks.enabled) {
        return invalid(
            "CPU xFormers attention sinks are not implemented yet");
    }
    return {true, nullptr};
}

bool forward(const NativeRequest& request) {
    NativeScoreBuffer scores;
    return qkt(request, scores) &&
           apply_mask_and_bias(request, scores) &&
           softmax(request, scores) &&
           av(request, scores);
}

} // namespace edcpp::api::attention::xformers::cpu::definition::cpu
