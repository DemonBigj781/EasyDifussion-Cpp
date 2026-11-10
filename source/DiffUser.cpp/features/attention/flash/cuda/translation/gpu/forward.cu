#include "features/attention/flash/common/flash_attention.hpp"
#include "features/attention/flash/cuda/definition/gpu/flash_attention_cuda.hpp"

// Static-library builds retain this adapter object through this anchor so its
// initializer can publish the CUDA translation to Common.
extern "C" void edcpp_flash_attention_cuda_translation_link_anchor() {}

namespace edcpp::api::attention::flash::cuda::translation::gpu {
namespace {

namespace native =
    edcpp::api::attention::flash::cuda::definition::gpu;

native::NativeDType translate_dtype(DType dtype) noexcept {
    switch (dtype) {
        case DType::f32: return native::NativeDType::f32;
        case DType::f16: return native::NativeDType::f16;
        case DType::bf16: return native::NativeDType::bf16;
    }
    return static_cast<native::NativeDType>(255);
}

template <typename Destination, typename Source>
Destination translate_tensor(const Source& source) noexcept {
    Destination result;
    result.data = source.data;
    result.dtype = translate_dtype(source.dtype);
    result.head_dim = source.head_dim;
    result.tokens = source.tokens;
    result.heads = source.heads;
    result.batch = source.batch;
    result.byte_strides = source.byte_strides;
    return result;
}

native::NativeRequest translate_request(const Request& request) noexcept {
    native::NativeRequest result;
    result.query =
        translate_tensor<native::NativeTensor4D>(request.query);
    result.key =
        translate_tensor<native::NativeTensor4D>(request.key);
    result.value =
        translate_tensor<native::NativeTensor4D>(request.value);
    result.mask =
        translate_tensor<native::NativeTensor4D>(request.mask);
    result.output =
        translate_tensor<native::NativeMutableTensor4D>(request.output);
    result.scale = request.scale;
    result.max_bias = request.max_bias;
    result.logit_softcap = request.logit_softcap;
    result.execution = {
        request.execution.thread_index,
        request.execution.thread_count,
        request.execution.stream,
        request.execution.synchronize,
        request.execution.device_index};
    return result;
}

Result translate_result(native::NativeResult result) noexcept {
    return {result.ok, result.message};
}

Result validate_cuda(const Request& request) noexcept {
    return translate_result(
        native::validate_execution(translate_request(request)));
}

Result forward_cuda(const Request& request) noexcept {
    return translate_result(native::forward(translate_request(request)));
}

Capabilities translated_capabilities() noexcept {
    const native::NativeCapabilities source = native::capabilities();
    Capabilities result;
    result.forward = source.fused_forward;
    result.additive_mask = source.additive_mask;
    result.alibi_bias = source.alibi_bias;
    result.logit_softcap = source.logit_softcap;
    result.grouped_query = source.grouped_query;
    result.f32_accumulation = source.f32_accumulation;
    return result;
}

const Translation cuda_translation = {
    Backend::cuda,
    "cuda",
    translated_capabilities(),
    &validate_cuda,
    &forward_cuda,
};

[[maybe_unused]] const bool registered =
    register_translation(&cuda_translation);

} // namespace
} // namespace edcpp::api::attention::flash::cuda::translation::gpu
