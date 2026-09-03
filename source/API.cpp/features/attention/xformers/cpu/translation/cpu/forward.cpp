#include "features/attention/xformers/common/xformers.hpp"

#include <cmath>

namespace edcpp::api::attention::xformers::cpu::translation {

bool qkt(const AttentionRequest&, ScoreBuffer&);
bool apply_mask_and_bias(const AttentionRequest&, ScoreBuffer&);
bool softmax(const AttentionRequest&, ScoreBuffer&);
bool av(const AttentionRequest&, const ScoreBuffer&);

namespace {
ValidationResult invalid(const char* message) {
    return {false, message};
}

bool broadcast_dim(std::int64_t value, std::int64_t expected) {
    return value == 1 || value == expected;
}

ValidationResult validate(const AttentionRequest& request) {
    if (request.dtype != DType::f32) {
        return invalid("CPU xFormers currently supports F32 only");
    }
    if (request.q.data == nullptr || request.k.data == nullptr || request.v.data == nullptr ||
        request.out.data == nullptr) {
        return invalid("CPU xFormers requires non-null Q, K, V, and output buffers");
    }
    if (request.q.batch <= 0 || request.q.heads <= 0 || request.q.tokens <= 0 || request.q.head_dim <= 0 ||
        request.k.batch <= 0 || request.k.heads <= 0 || request.k.tokens <= 0 || request.k.head_dim <= 0 ||
        request.v.batch <= 0 || request.v.heads <= 0 || request.v.tokens <= 0 || request.v.head_dim <= 0) {
        return invalid("CPU xFormers tensor dimensions must be positive");
    }
    if (request.q.batch != request.k.batch || request.q.batch != request.v.batch) {
        return invalid("CPU xFormers currently requires matching Q/K/V batch counts");
    }
    if (request.q.head_dim != request.k.head_dim) {
        return invalid("CPU xFormers requires matching Q and K head dimensions");
    }
    if (request.k.tokens != request.v.tokens) {
        return invalid("CPU xFormers requires matching K and V token counts");
    }
    if (request.q.heads % request.k.heads != 0 || request.q.heads % request.v.heads != 0) {
        return invalid("CPU xFormers requires K/V head counts to divide Q head count");
    }
    if (request.out.batch != request.q.batch || request.out.heads != request.q.heads ||
        request.out.tokens != request.q.tokens || request.out.head_dim != request.v.head_dim) {
        return invalid("CPU xFormers output shape must be [Q batch, Q heads, Q tokens, V head_dim]");
    }
    if (!std::isfinite(request.scale) || request.scale < 0.0f) {
        return invalid("CPU xFormers scale must be finite and non-negative");
    }
    if (!std::isfinite(request.softcap) || request.softcap < 0.0f) {
        return invalid("CPU xFormers softcap must be finite and non-negative");
    }
    if (request.mask.data != nullptr &&
        (!broadcast_dim(request.mask.batch, request.q.batch) ||
         !broadcast_dim(request.mask.heads, request.q.heads) ||
         !broadcast_dim(request.mask.query_tokens, request.q.tokens) ||
         !broadcast_dim(request.mask.key_tokens, request.k.tokens))) {
        return invalid("CPU xFormers mask dimensions must be 1 or match the corresponding attention dimension");
    }
    if (request.alibi.enabled &&
        (request.alibi.slopes == nullptr || request.alibi.slope_count < request.q.heads)) {
        return invalid("CPU xFormers ALiBi requires one slope per Q head");
    }
    if (request.sinks.enabled) {
        return invalid("CPU xFormers attention sinks are not implemented yet");
    }
    return {true, {}};
}

bool forward_impl(const AttentionRequest& request) {
    ScoreBuffer scores;
    return qkt(request, scores) &&
           apply_mask_and_bias(request, scores) &&
           softmax(request, scores) &&
           av(request, scores);
}

const Translation cpu_translation = [] {
    Translation translation;
    translation.backend = Backend::cpu;
    translation.name = "cpu";
    translation.capabilities.forward = true;
    translation.capabilities.qkt = true;
    translation.capabilities.additive_mask = true;
    translation.capabilities.causal_mask = true;
    translation.capabilities.alibi = true;
    translation.capabilities.softcap = true;
    translation.capabilities.attention_sinks = false;
    translation.capabilities.gqa = true;
    translation.capabilities.mqa = true;
    translation.capabilities.f32 = true;
    translation.capabilities.f16 = false;
    translation.capabilities.bf16 = false;
    translation.validate = &validate;
    translation.qkt = &qkt;
    translation.mask = &apply_mask_and_bias;
    translation.softmax = &softmax;
    translation.av = &av;
    translation.forward = &forward_impl;
    return translation;
}();

const bool registered = register_translation(&cpu_translation);
}

} // namespace edcpp::api::attention::xformers::cpu::translation
