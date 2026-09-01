#include "xformers.hpp"

#include <cmath>

namespace xformers::prototype {

BackendIdentity backend_identity() {
    return BackendIdentity{};
}

Capabilities capabilities() {
    return Capabilities{};
}

ValidationResult validate(const AttentionRequest & request) {
    ValidationResult result{};

    if (!request.q.data || !request.k.data || !request.v.data || !request.out.data) {
        result.message = "Q/K/V/output data pointers must be non-null";
        return result;
    }

    if (request.q.batch <= 0 || request.q.heads <= 0 || request.q.tokens <= 0 || request.q.head_dim <= 0) {
        result.message = "Q shape is invalid";
        return result;
    }

    if (request.k.batch != request.q.batch || request.v.batch != request.q.batch) {
        result.message = "Q/K/V batch dimensions must match";
        return result;
    }

    if (request.k.tokens != request.v.tokens) {
        result.message = "K and V token counts must match";
        return result;
    }

    if (request.q.head_dim != request.k.head_dim || request.k.head_dim != request.v.head_dim) {
        result.message = "Q/K/V head dimensions must match";
        return result;
    }

    if (request.k.heads <= 0 || request.v.heads <= 0 || request.k.heads != request.v.heads) {
        result.message = "K/V head counts must be positive and equal";
        return result;
    }

    if (request.q.heads % request.k.heads != 0) {
        result.message = "Q head count must be divisible by K/V head count for MHA/GQA/MQA mapping";
        return result;
    }

    if (request.out.batch != request.q.batch || request.out.heads != request.q.heads ||
        request.out.tokens != request.q.tokens || request.out.head_dim != request.v.head_dim) {
        result.message = "Output shape must be [Q batch, Q heads, Q tokens, V head_dim]";
        return result;
    }

    if (request.dtype == DType::BF16) {
        result.message = "Prototype reference path does not execute BF16 yet";
        return result;
    }

    if (request.mask.data) {
        const bool batch_ok = request.mask.batch == 1 || request.mask.batch == request.q.batch;
        const bool heads_ok = request.mask.heads == 1 || request.mask.heads == request.q.heads;
        const bool q_ok = request.mask.query_tokens == 1 || request.mask.query_tokens == request.q.tokens;
        const bool k_ok = request.mask.key_tokens == 1 || request.mask.key_tokens == request.k.tokens;
        if (!batch_ok || !heads_ok || !q_ok || !k_ok) {
            result.message = "Mask dimensions are not broadcast-compatible with attention scores";
            return result;
        }
    }

    if (request.alibi.enabled && (!request.alibi.slopes || request.alibi.slope_count < request.q.heads)) {
        result.message = "ALiBi requires at least one slope per query head";
        return result;
    }

    if (request.sinks.enabled && (!request.sinks.values || request.sinks.value_count < request.q.heads)) {
        result.message = "Attention sinks require at least one value per query head";
        return result;
    }

    if (request.softcap < 0.0f || !std::isfinite(request.softcap)) {
        result.message = "Softcap must be finite and non-negative";
        return result;
    }

    result.ok = true;
    result.message = request.dtype == DType::F16 ? "ok (F16 storage, F32 accumulation)" : "ok";
    return result;
}

bool forward(const AttentionRequest & request) {
    const ValidationResult validation = validate(request);
    if (!validation.ok) {
        return false;
    }

    ScoreBuffer scores{};
    if (!qkt(request, scores)) {
        return false;
    }
    if (!apply_mask_and_bias(request, scores)) {
        return false;
    }
    if (!softmax(scores)) {
        return false;
    }
    if (!av(request, scores)) {
        return false;
    }

    return true;
}

} // namespace xformers::prototype
