#include "features/attention/xformers/common/xformers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using edcpp::api::Backend;
using edcpp::api::attention::xformers::AttentionRequest;
using edcpp::api::attention::xformers::DType;
using edcpp::api::attention::xformers::Tensor4D;
using edcpp::api::attention::xformers::MutableTensor4D;
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

} // namespace

int main() {
    bool ok = true;

    const auto* translation = translation_for(Backend::cpu);
    ok &= check(translation != nullptr, "CPU xFormers translation is registered with Common");
    if (translation == nullptr) {
        return EXIT_FAILURE;
    }

    const auto caps = capabilities(Backend::cpu);
    ok &= check(caps.forward, "CPU xFormers advertises forward support");
    ok &= check(caps.qkt, "CPU xFormers advertises QK^T support");
    ok &= check(caps.additive_mask, "CPU xFormers advertises additive-mask support");
    ok &= check(caps.causal_mask, "CPU xFormers advertises causal-mask support");
    ok &= check(caps.f32, "CPU xFormers advertises F32 support");
    ok &= check(!caps.f16, "CPU xFormers does not falsely advertise F16 support");
    ok &= check(!caps.bf16, "CPU xFormers does not falsely advertise BF16 support");

    // Q and K are orthogonal unit vectors. With scale=1 the score rows are
    // [1, 0] and [0, 1]. This gives exact expected softmax weights based on e.
    const float q[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const float k[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const float v[] = {
        10.0f, 20.0f,
        30.0f, 40.0f,
    };
    float out[4] = {};

    AttentionRequest request{};
    request.q = Tensor4D{q, 1, 1, 2, 2};
    request.k = Tensor4D{k, 1, 1, 2, 2};
    request.v = Tensor4D{v, 1, 1, 2, 2};
    request.out = MutableTensor4D{out, 1, 1, 2, 2};
    request.dtype = DType::f32;
    request.scale = 1.0f;

    const auto validation = validate(Backend::cpu, request);
    if (!validation.ok) {
        std::cerr << "[FAIL] Valid CPU request rejected: " << validation.message << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "[PASS] Common validation accepts deterministic CPU request\n";

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

    for (int i = 0; i < 4; ++i) {
        if (!nearly_equal(out[i], expected[i])) {
            std::cerr << "[FAIL] output[" << i << "] expected " << expected[i]
                      << " but received " << out[i] << '\n';
            ok = false;
        }
    }
    if (ok) {
        std::cout << "[PASS] Deterministic attention output matches analytical result\n";
    }

    // Confirm validation rejects a dtype the CPU translation does not advertise.
    AttentionRequest unsupported = request;
    unsupported.dtype = DType::f16;
    const auto unsupported_validation = validate(Backend::cpu, unsupported);
    ok &= check(!unsupported_validation.ok, "CPU validation rejects unsupported F16 request");

    if (!ok) {
        std::cerr << "CPU xFormers Common API test failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "CPU xFormers Common API test passed\n";
    return EXIT_SUCCESS;
}
