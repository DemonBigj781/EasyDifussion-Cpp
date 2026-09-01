#include "xformers.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace xformers::prototype;

static bool nearly_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // One batch, two query heads, one KV head (MQA-style mapping),
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
        std::cerr << "validation failed: " << validation.message << '\n';
        return 1;
    }

    if (!forward(request)) {
        std::cerr << "forward failed\n";
        return 2;
    }

    // Each output row must be a convex combination of the two V rows.
    for (std::size_t i = 0; i < out.size(); i += 2) {
        const float x = out[i] / 10.f;
        const float y = out[i + 1] / 20.f;
        if (!nearly_equal(x + y, 1.f, 2e-4f)) {
            std::cerr << "row " << (i / 2) << " is not normalized\n";
            return 3;
        }
    }

    // Causal mode should force token 0 to attend only to key 0.
    request.causal = true;
    std::fill(out.begin(), out.end(), 0.f);
    if (!forward(request)) {
        std::cerr << "causal forward failed\n";
        return 4;
    }

    if (!nearly_equal(out[0], 10.f) || !nearly_equal(out[1], 0.f)) {
        std::cerr << "causal token-0 result mismatch\n";
        return 5;
    }

    std::cout << "xFormers prototype self-test passed\n";
    return 0;
}
