#include "xformers.hpp"

namespace edcpp::api::attention::xformers::cuda::definition::gpu {

NativeCapabilities capabilities() noexcept {
    NativeCapabilities result;
    result.fused_forward = true;
    result.separate_stages = false;
    result.additive_mask = true;
    result.causal_mask = true;
    result.alibi = true;
    result.softcap = true;
    result.attention_sinks = true;
    result.gqa = true;
    result.mqa = true;
    result.f32 = true;
    result.f16 = true;
    result.bf16 = true;
    result.maximum_value_dimension = 512;
    result.minimum_compute_major = 6;
    return result;
}

NativeValidationResult validate_forward(const NativeRequest& request) noexcept {
    const NativeValidationResult qkt = validate_qkt(request);
    if (!qkt.ok) return qkt;

    const NativeValidationResult mask = validate_mask(request);
    if (!mask.ok) return mask;

    const NativeValidationResult softmax = validate_softmax(request);
    if (!softmax.ok) return softmax;

    return validate_av(request);
}

} // namespace edcpp::api::attention::xformers::cuda::definition::gpu
