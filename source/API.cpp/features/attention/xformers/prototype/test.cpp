#include "xformers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace xformers::prototype;

static bool nearly_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // F32 reference: one batch, two query heads, one KV head (MQA-style),
    // two tokens, head dimension two.
    std::vector<float> q = {
        1.f, 0.f,  0.f, 1.f,
        1.f, 1.f,  1.f,-1.f,
    };
    std::vector<float> k = {
        1.f, 0.f,
        0.f, 1.f,
    };
    std::vector<float> v = {
        10.f, 0.f,
        0.f, 20.f,
    };
    std::vector<float> out(8, 0.f);

    AttentionRequest request{};
    request.q = Tensor4D{q.data(), 1, 2, 2, 2};
    request.k = Tensor4D{k.data(), 1, 1, 2, 2};
    request.v = Tensor4D{v.data(), 1, 1, 2, 2};
    request.out = MutableTensor4D{out.data(), 1, 2, 2, 2};
    request.dtype = DType::F32;
    request.causal = false;

    const auto validation = validate(request);
    if (!validation.ok) {
        std::cerr << "F32 validation failed: " << validation.message << '\n';
        return 1;
    }

    if (!forward(request)) {
        std::cerr << "F32 forward failed\n";
        return 2;
    }

    for (std::size_t i = 0; i < out.size(); i += 2) {
        const float x = out[i] / 10.f;
        const float y = out[i + 1] / 20.f;
        if (!nearly_equal(x + y, 1.f, 2e-4f)) {
            std::cerr << "F32 row " << (i / 2) << " is not normalized\n";
            return 3;
        }
    }

    request.causal = true;
    std::fill(out.begin(), out.end(), 0.f);
    if (!forward(request)) {
        std::cerr << "F32 causal forward failed\n";
        return 4;
    }

    if (!nearly_equal(out[0], 10.f) || !nearly_equal(out[1], 0.f)) {
        std::cerr << "F32 causal token-0 result mismatch\n";
        return 5;
    }

    // F16 storage path. Convert the same inputs to binary16, execute with F32
    // score/softmax/AV accumulation, then compare decoded output to F32.
    std::vector<std::uint16_t> q16(q.size());
    std::vector<std::uint16_t> k16(k.size());
    std::vector<std::uint16_t> v16(v.size());
    std::vector<std::uint16_t> out16(out.size(), 0);
    for (std::size_t i = 0; i < q.size(); ++i) q16[i] = float_to_f16(q[i]);
    for (std::size_t i = 0; i < k.size(); ++i) k16[i] = float_to_f16(k[i]);
    for (std::size_t i = 0; i < v.size(); ++i) v16[i] = float_to_f16(v[i]);

    request.q = Tensor4D{q16.data(), 1, 2, 2, 2};
    request.k = Tensor4D{k16.data(), 1, 1, 2, 2};
    request.v = Tensor4D{v16.data(), 1, 1, 2, 2};
    request.out = MutableTensor4D{out16.data(), 1, 2, 2, 2};
    request.dtype = DType::F16;
    request.causal = true;

    const auto validation16 = validate(request);
    if (!validation16.ok) {
        std::cerr << "F16 validation failed: " << validation16.message << '\n';
        return 6;
    }

    if (!forward(request)) {
        std::cerr << "F16 forward failed\n";
        return 7;
    }

    if (!nearly_equal(f16_to_float(out16[0]), 10.f, 1e-2f) ||
        !nearly_equal(f16_to_float(out16[1]), 0.f, 1e-2f)) {
        std::cerr << "F16 causal token-0 result mismatch\n";
        return 8;
    }

    const auto caps = capabilities();
    if (!caps.f16) {
        std::cerr << "F16 capability was not advertised\n";
        return 9;
    }

    std::cout << "xFormers prototype F32/F16 self-test passed\n";
    return 0;
}
