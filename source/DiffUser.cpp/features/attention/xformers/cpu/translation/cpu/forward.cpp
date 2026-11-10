#include "translate.hpp"

#include <string>

namespace edcpp::api::attention::xformers::cpu::translation {

bool qkt(const AttentionRequest&, ScoreBuffer&);
bool apply_mask_and_bias(const AttentionRequest&, ScoreBuffer&);
bool softmax(const AttentionRequest&, ScoreBuffer&);
bool av(const AttentionRequest&, const ScoreBuffer&);

namespace {

ValidationResult validate(const AttentionRequest& request) {
    const native::NativeResult result =
        native::validate(translate_request(request));
    return {
        result.ok,
        result.message == nullptr ? std::string{} : std::string{result.message}};
}

bool forward_impl(const AttentionRequest& request) {
    return native::forward(translate_request(request));
}

Capabilities translated_capabilities() noexcept {
    const native::NativeCapabilities source = native::capabilities();
    Capabilities result;
    result.forward = source.forward;
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

const Translation cpu_translation = {
    Backend::cpu,
    "cpu",
    translated_capabilities(),
    &validate,
    &qkt,
    &apply_mask_and_bias,
    &softmax,
    &av,
    &forward_impl,
};

[[maybe_unused]] const bool registered =
    register_translation(&cpu_translation);

} // namespace
} // namespace edcpp::api::attention::xformers::cpu::translation
