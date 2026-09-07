#include "features/attention/sage/common/sage_attention.hpp"
#include "features/attention/sage/cuda/definition/gpu/sage_attention_cuda.hpp"
#include "features/attention/sage/cuda/translation/gpu/support.hpp"

namespace edcpp::api::attention::sage::cuda::translation::gpu {
namespace {

namespace native = edcpp::api::attention::sage::cuda::definition::gpu;

native::NativeDType translate_dtype(DType dtype) noexcept {
    switch (dtype) {
        case DType::f16: return native::NativeDType::f16;
        case DType::bf16: return native::NativeDType::bf16;
        case DType::f32:
        default: return native::NativeDType::f32;
    }
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

native::NativeSupportRequest translate_request(
        const SupportRequest& request) noexcept {
    native::NativeSupportRequest result;
    result.query = translate_tensor<native::NativeTensor4D>(request.query);
    result.key = translate_tensor<native::NativeTensor4D>(request.key);
    result.value = translate_tensor<native::NativeTensor4D>(request.value);
    result.output =
        translate_tensor<native::NativeMutableTensor4D>(request.output);
    result.device_index = request.device.index;
    result.compute_major = request.device.compute_major;
    result.compute_minor = request.device.compute_minor;
    result.scale = request.scale;
    result.max_bias = request.max_bias;
    result.logit_softcap = request.logit_softcap;
    result.additive_mask = request.additive_mask;
    result.attention_sinks = request.attention_sinks;
    return result;
}

Result support_cuda(const SupportRequest& request) noexcept {
    const native::NativeResult result = native::support(
        translate_request(request));
    return {result.ok, result.message};
}

Capabilities translated_capabilities() noexcept {
    const native::NativeCapabilities source = native::capabilities();
    Capabilities result;
    result.support = source.support;
    result.forward = source.forward;
    result.int8_qk = source.int8_qk;
    result.fp16_pv = source.fp16_pv;
    result.grouped_query = source.grouped_query;
    result.minimum_compute_major = source.minimum_compute_major;
    result.maximum_compute_major_exclusive =
        source.maximum_compute_major_exclusive;
    return result;
}

const Translation cuda_translation{
    Backend::cuda,
    "cuda-sage-sm80-support",
    translated_capabilities(),
    &support_cuda,
};

} // namespace

bool ensure_registered() noexcept {
    static const bool registered = register_translation(&cuda_translation);
    return registered;
}

[[maybe_unused]] const bool registered = ensure_registered();

} // namespace edcpp::api::attention::sage::cuda::translation::gpu
