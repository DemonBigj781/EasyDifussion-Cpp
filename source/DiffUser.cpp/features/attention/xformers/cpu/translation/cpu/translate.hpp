#pragma once

#include "features/attention/xformers/common/xformers.hpp"
#include "features/attention/xformers/cpu/definition/cpu/xformers.hpp"

#include <utility>

namespace edcpp::api::attention::xformers::cpu::translation {

namespace native =
    edcpp::api::attention::xformers::cpu::definition::cpu;

inline native::NativeDType translate_dtype(DType dtype) noexcept {
    switch (dtype) {
        case DType::f32: return native::NativeDType::f32;
        case DType::f16: return native::NativeDType::f16;
        case DType::bf16: return native::NativeDType::bf16;
    }
    return native::NativeDType::f32;
}

inline native::NativeRequest translate_request(
    const AttentionRequest& request) noexcept {
    native::NativeRequest result;
    result.q = {
        request.q.data, request.q.batch, request.q.heads, request.q.tokens,
        request.q.head_dim, translate_dtype(request.q.dtype),
        request.q.byte_strides};
    result.k = {
        request.k.data, request.k.batch, request.k.heads, request.k.tokens,
        request.k.head_dim, translate_dtype(request.k.dtype),
        request.k.byte_strides};
    result.v = {
        request.v.data, request.v.batch, request.v.heads, request.v.tokens,
        request.v.head_dim, translate_dtype(request.v.dtype),
        request.v.byte_strides};
    result.out = {
        request.out.data, request.out.batch, request.out.heads,
        request.out.tokens, request.out.head_dim,
        translate_dtype(request.out.dtype), request.out.byte_strides};
    result.dtype = translate_dtype(request.dtype);
    result.scale = request.scale;
    result.softcap = request.softcap;
    result.causal = request.causal;
    result.mask = {
        request.mask.data, request.mask.batch, request.mask.heads,
        request.mask.query_tokens, request.mask.key_tokens,
        translate_dtype(request.mask.dtype), request.mask.byte_strides};
    result.alibi = {
        request.alibi.enabled, request.alibi.slopes,
        request.alibi.slope_count, request.alibi.max_bias};
    result.sinks = {
        request.sinks.enabled, request.sinks.values,
        request.sinks.value_count};
    return result;
}

inline native::NativeScoreBuffer translate_scores(
    const ScoreBuffer& scores) {
    native::NativeScoreBuffer result;
    result.values = scores.values;
    result.batch = scores.batch;
    result.heads = scores.heads;
    result.query_tokens = scores.query_tokens;
    result.key_tokens = scores.key_tokens;
    result.extra_denominator_logits = scores.extra_denominator_logits;
    return result;
}

inline void translate_scores(
    native::NativeScoreBuffer&& source, ScoreBuffer& destination) {
    destination.values = std::move(source.values);
    destination.batch = source.batch;
    destination.heads = source.heads;
    destination.query_tokens = source.query_tokens;
    destination.key_tokens = source.key_tokens;
    destination.extra_denominator_logits =
        std::move(source.extra_denominator_logits);
}

} // namespace edcpp::api::attention::xformers::cpu::translation
