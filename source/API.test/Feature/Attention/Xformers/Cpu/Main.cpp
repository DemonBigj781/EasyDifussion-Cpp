#include "features/attention/xformers/common/xformers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

using edcpp::api::Backend;
using edcpp::api::attention::xformers::AttentionRequest;
using edcpp::api::attention::xformers::DType;
using edcpp::api::attention::xformers::MaskView;
using edcpp::api::attention::xformers::MutableTensor4D;
using edcpp::api::attention::xformers::Tensor4D;
using edcpp::api::attention::xformers::capabilities;
using edcpp::api::attention::xformers::forward;
using edcpp::api::attention::xformers::translation_for;
using edcpp::api::attention::xformers::validate;

constexpr float kTolerance = 1.0e-5f;

bool nearly_equal(float actual, float expected, float tolerance = kTolerance) {
    return std::fabs(actual - expected) <= tolerance;
}

bool check(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
        return false;
    }
    std::cout << "[PASS] " << message << '\n';
    return true;
}

bool check_array(const float* actual, const float* expected, int count, std::string_view message,
                 float tolerance = kTolerance) {
    for (int i = 0; i < count; ++i) {
        if (!nearly_equal(actual[i], expected[i], tolerance)) {
            std::cerr << "[FAIL] " << message << " at index " << i << ": expected "
                      << expected[i] << " but received " << actual[i] << '\n';
            return false;
        }
    }
    std::cout << "[PASS] " << message << '\n';
    return true;
}

AttentionRequest make_basic_request(const float* q, const float* k, const float* v, float* out) {
    AttentionRequest request{};
    request.q = Tensor4D{q, 1, 1, 2, 2};
    request.k = Tensor4D{k, 1, 1, 2, 2};
    request.v = Tensor4D{v, 1, 1, 2, 2};
    request.out = MutableTensor4D{out, 1, 1, 2, 2};
    request.dtype = DType::f32;
    request.scale = 1.0f;
    return request;
}

} // namespace

int main() {
    bool ok = true;

    const auto* translation = translation_for(Backend::cpu);
    ok &= check(translation != nullptr, "CPU xFormers translation is registered with Common");
    if (translation == nullptr) return EXIT_FAILURE;

    const auto caps = capabilities(Backend::cpu);
    ok &= check(caps.forward, "CPU xFormers advertises forward support");
    ok &= check(caps.qkt, "CPU xFormers advertises QK^T support");
    ok &= check(caps.additive_mask, "CPU xFormers advertises additive-mask support");
    ok &= check(caps.causal_mask, "CPU xFormers advertises causal-mask support");
    ok &= check(caps.alibi, "CPU xFormers advertises ALiBi support");
    ok &= check(caps.softcap, "CPU xFormers advertises soft-cap support");
    ok &= check(caps.gqa, "CPU xFormers advertises GQA support");
    ok &= check(caps.mqa, "CPU xFormers advertises MQA support");
    ok &= check(!caps.attention_sinks, "CPU xFormers does not falsely advertise attention-sink support");
    ok &= check(caps.f32, "CPU xFormers advertises F32 support");
    ok &= check(!caps.f16, "CPU xFormers does not falsely advertise F16 support");
    ok &= check(!caps.bf16, "CPU xFormers does not falsely advertise BF16 support");

    const float q[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float k[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float v[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float out[4] = {};
    AttentionRequest request = make_basic_request(q, k, v, out);

    const auto validation = validate(Backend::cpu, request);
    ok &= check(validation.ok, "Common validation accepts deterministic CPU request");
    ok &= check(forward(Backend::cpu, request), "Common dispatch executes CPU xFormers forward");

    const float e = std::exp(1.0f);
    const float strong = e / (e + 1.0f);
    const float weak = 1.0f / (e + 1.0f);
    const float expected[] = {
        strong * 10.0f + weak * 30.0f,
        strong * 20.0f + weak * 40.0f,
        weak * 10.0f + strong * 30.0f,
        weak * 20.0f + strong * 40.0f,
    };
    ok &= check_array(out, expected, 4, "Deterministic attention output matches analytical result");

    const float zero_qk[] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float scalar_v[] = {10.0f, 30.0f};
    float causal_out[2] = {};
    AttentionRequest causal{};
    causal.q = Tensor4D{zero_qk, 1, 1, 2, 2};
    causal.k = Tensor4D{zero_qk, 1, 1, 2, 2};
    causal.v = Tensor4D{scalar_v, 1, 1, 2, 1};
    causal.out = MutableTensor4D{causal_out, 1, 1, 2, 1};
    causal.dtype = DType::f32;
    causal.scale = 1.0f;
    causal.causal = true;
    const float causal_expected[] = {10.0f, 20.0f};
    ok &= check(validate(Backend::cpu, causal).ok, "CPU validation accepts causal attention request");
    ok &= check(forward(Backend::cpu, causal), "CPU Common dispatch executes causal attention");
    ok &= check_array(causal_out, causal_expected, 2, "Causal mask produces expected output");

    const float additive_mask[] = {0.0f, std::log(3.0f)};
    float mask_out[2] = {};
    AttentionRequest masked = causal;
    masked.causal = false;
    masked.out.data = mask_out;
    masked.mask = MaskView{additive_mask, 1, 1, 1, 2};
    const float mask_expected[] = {25.0f, 25.0f};
    ok &= check(validate(Backend::cpu, masked).ok, "CPU validation accepts broadcast additive mask");
    ok &= check(forward(Backend::cpu, masked), "CPU Common dispatch executes additive mask");
    ok &= check_array(mask_out, mask_expected, 2, "Additive mask broadcasting produces expected output");

    const float alibi_slope[] = {std::log(3.0f)};
    float alibi_out[2] = {};
    AttentionRequest alibi = causal;
    alibi.causal = false;
    alibi.out.data = alibi_out;
    alibi.alibi.enabled = true;
    alibi.alibi.slopes = alibi_slope;
    alibi.alibi.slope_count = 1;
    const float alibi_expected[] = {25.0f, 25.0f};
    ok &= check(validate(Backend::cpu, alibi).ok, "CPU validation accepts ALiBi request");
    ok &= check(forward(Backend::cpu, alibi), "CPU Common dispatch executes ALiBi");
    ok &= check_array(alibi_out, alibi_expected, 2, "ALiBi bias produces expected output");

    const float soft_q[] = {2.0f, 0.0f};
    const float soft_k[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float soft_v[] = {4.0f, 8.0f};
    float soft_out[1] = {};
    AttentionRequest softcap{};
    softcap.q = Tensor4D{soft_q, 1, 1, 1, 2};
    softcap.k = Tensor4D{soft_k, 1, 1, 2, 2};
    softcap.v = Tensor4D{soft_v, 1, 1, 2, 1};
    softcap.out = MutableTensor4D{soft_out, 1, 1, 1, 1};
    softcap.dtype = DType::f32;
    softcap.scale = 1.0f;
    softcap.softcap = 1.0f;
    const float capped = std::tanh(2.0f);
    const float capped_e = std::exp(capped);
    const float soft_expected[] = {(capped_e * 4.0f + 8.0f) / (capped_e + 1.0f)};
    ok &= check(validate(Backend::cpu, softcap).ok, "CPU validation accepts soft-cap request");
    ok &= check(forward(Backend::cpu, softcap), "CPU Common dispatch executes soft-cap attention");
    ok &= check_array(soft_out, soft_expected, 1, "Soft-cap produces expected analytical output");

    const float gqa_q[] = {
        1,0, 1,0,
        1,0, 1,0,
        1,0, 1,0,
        1,0, 1,0,
    };
    const float gqa_k[] = {
        1,0, 0,1,
        0,1, 1,0,
    };
    const float gqa_v[] = {10,30, 100,300};
    float gqa_out[8] = {};
    AttentionRequest gqa{};
    gqa.q = Tensor4D{gqa_q, 1, 4, 2, 2};
    gqa.k = Tensor4D{gqa_k, 1, 2, 2, 2};
    gqa.v = Tensor4D{gqa_v, 1, 2, 2, 1};
    gqa.out = MutableTensor4D{gqa_out, 1, 4, 2, 1};
    gqa.dtype = DType::f32;
    gqa.scale = 1.0f;
    const float gqa_expected[] = {
        strong*10 + weak*30, strong*10 + weak*30,
        strong*10 + weak*30, strong*10 + weak*30,
        weak*100 + strong*300, weak*100 + strong*300,
        weak*100 + strong*300, weak*100 + strong*300,
    };
    ok &= check(validate(Backend::cpu, gqa).ok, "CPU validation accepts GQA request");
    ok &= check(forward(Backend::cpu, gqa), "CPU Common dispatch executes GQA");
    ok &= check_array(gqa_out, gqa_expected, 8, "GQA maps Q-head groups to expected K/V heads");

    const float mqa_q[] = {1,0, 0,1, 1,0, 0,1};
    const float mqa_k[] = {1,0, 0,1};
    const float mqa_v[] = {5,9};
    float mqa_out[4] = {};
    AttentionRequest mqa{};
    mqa.q = Tensor4D{mqa_q, 1, 4, 1, 2};
    mqa.k = Tensor4D{mqa_k, 1, 1, 2, 2};
    mqa.v = Tensor4D{mqa_v, 1, 1, 2, 1};
    mqa.out = MutableTensor4D{mqa_out, 1, 4, 1, 1};
    mqa.dtype = DType::f32;
    mqa.scale = 1.0f;
    const float mqa_expected[] = {
        strong*5 + weak*9,
        weak*5 + strong*9,
        strong*5 + weak*9,
        weak*5 + strong*9,
    };
    ok &= check(validate(Backend::cpu, mqa).ok, "CPU validation accepts MQA request");
    ok &= check(forward(Backend::cpu, mqa), "CPU Common dispatch executes MQA");
    ok &= check_array(mqa_out, mqa_expected, 4, "MQA shares one K/V head across all Q heads");

    AttentionRequest unsupported = request;
    unsupported.dtype = DType::f16;
    ok &= check(!validate(Backend::cpu, unsupported).ok, "CPU validation rejects unsupported F16 request");

    AttentionRequest bad_heads = request;
    bad_heads.q.heads = 3;
    bad_heads.k.heads = 2;
    bad_heads.v.heads = 2;
    bad_heads.out.heads = 3;
    ok &= check(!validate(Backend::cpu, bad_heads).ok, "CPU validation rejects non-divisible K/V head counts");

    AttentionRequest bad_mask = request;
    const float invalid_mask[] = {0.0f};
    bad_mask.mask = MaskView{invalid_mask, 1, 1, 3, 1};
    ok &= check(!validate(Backend::cpu, bad_mask).ok, "CPU validation rejects incompatible mask dimensions");

    AttentionRequest bad_output = request;
    bad_output.out.tokens = 1;
    ok &= check(!validate(Backend::cpu, bad_output).ok, "CPU validation rejects mismatched output shape");

    AttentionRequest bad_scale = request;
    bad_scale.scale = -1.0f;
    ok &= check(!validate(Backend::cpu, bad_scale).ok, "CPU validation rejects negative scale");

    AttentionRequest bad_softcap = request;
    bad_softcap.softcap = std::numeric_limits<float>::infinity();
    ok &= check(!validate(Backend::cpu, bad_softcap).ok, "CPU validation rejects non-finite soft-cap");

    AttentionRequest bad_alibi = request;
    bad_alibi.alibi.enabled = true;
    bad_alibi.alibi.slopes = nullptr;
    bad_alibi.alibi.slope_count = 0;
    ok &= check(!validate(Backend::cpu, bad_alibi).ok, "CPU validation rejects missing ALiBi slopes");

    AttentionRequest null_q = request;
    null_q.q.data = nullptr;
    ok &= check(!validate(Backend::cpu, null_q).ok, "CPU validation rejects null Q buffer");

    AttentionRequest sinks = request;
    const float sink_value[] = {0.0f};
    sinks.sinks.enabled = true;
    sinks.sinks.values = sink_value;
    sinks.sinks.value_count = 1;
    ok &= check(!validate(Backend::cpu, sinks).ok, "CPU validation rejects unimplemented attention sinks");

    if (!ok) {
        std::cerr << "CPU xFormers Common API test failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "CPU xFormers Common API test passed\n";
    return EXIT_SUCCESS;
}
