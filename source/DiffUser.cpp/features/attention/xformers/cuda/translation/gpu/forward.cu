#include "features/attention/xformers/common/xformers.hpp"
#include "features/attention/xformers/cuda/definition/gpu/xformers.hpp"

#include <string>

// Static-library builds retain this adapter object through this anchor so its
// initializer can publish the CUDA translation to Common.
extern "C" void edcpp_xformers_cuda_translation_link_anchor() {}

namespace edcpp::api::attention::xformers::cuda::translation::gpu {
namespace {

namespace native =
    edcpp::api::attention::xformers::cuda::definition::gpu;

native::NativeDType translate_dtype(DType dtype) noexcept {
    switch (dtype) {
        case DType::f32: return native::NativeDType::f32;
        case DType::f16: return native::NativeDType::f16;
        case DType::bf16: return native::NativeDType::bf16;
    }
    return native::NativeDType::f32;
}

native::NativeRequest translate_request(
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
    result.execution = {
        request.execution.stream, request.execution.synchronize,
        request.execution.device_index};
    return result;
}

ValidationResult validate(const AttentionRequest& request) {
    const native::NativeValidationResult result =
        native::validate_execution(translate_request(request));
    return {
        result.ok,
        result.message == nullptr ? std::string{} : std::string{result.message}};
}

bool forward(const AttentionRequest& request) {
    return native::forward(translate_request(request));
}

Capabilities translated_capabilities() noexcept {
    const native::NativeCapabilities source = native::capabilities();
    Capabilities result;
    result.forward = source.fused_forward;
    result.qkt = source.separate_stages;
    result.additive_mask = source.additive_mask;
    result.causal_mask = source.causal_mask;
    result.alibi = source.alibi;
    result.softcap = source.softcap;
    result.attention_sinks = source.attention_sinks;
    result.gqa = source.gqa;
    result.mqa = source.mqa;
    result.f32 = source.f32;
    result.f16 = source.f16;
    result.bf16 = source.bf16;
    return result;
}

const Translation cuda_translation = {
    Backend::cuda,
    "cuda",
    translated_capabilities(),
    &validate,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    &forward,
};

[[maybe_unused]] const bool registered =
    register_translation(&cuda_translation);

} // namespace
} // namespace edcpp::api::attention::xformers::cuda::translation::gpu
