#include "features/attention/flash/common/flash_attention.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace flash = edcpp::api::attention::flash;

int main() {
    using edcpp::api::Backend;
    const std::vector<float> query = {1.0f, 0.0f};
    const std::vector<float> key = {1.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<float> value = {10.0f, 20.0f};
    std::vector<float> output(1);

    flash::Request request;
    request.query = {query.data(), flash::DType::f32, 2, 1, 1, 1, {}};
    request.key = {key.data(), flash::DType::f32, 2, 2, 1, 1, {}};
    request.value = {value.data(), flash::DType::f32, 1, 2, 1, 1, {}};
    request.output = {output.data(), flash::DType::f32, 1, 1, 1, 1, {}};

    const auto caps = flash::capabilities(Backend::cpu);
    assert(caps.forward && caps.additive_mask && caps.alibi_bias);
    assert(caps.logit_softcap && caps.grouped_query && caps.f32_accumulation);

    auto invalid = request;
    invalid.query.data = nullptr;
    assert(!flash::validate(Backend::cpu, invalid).ok);
    invalid = request;
    invalid.execution.thread_count = 2;
    assert(!flash::validate(Backend::cpu, invalid).ok);
    assert(!flash::validate(Backend::none, request).ok);

    assert(flash::validate(Backend::cpu, request).ok);
    assert(flash::forward(Backend::cpu, request).ok);
    const float expected = (std::exp(1.0f) * 10.0f + 20.0f) / (std::exp(1.0f) + 1.0f);
    assert(std::fabs(output[0] - expected) < 1e-5f);
    std::cout << "Self-contained CPU FlashAttention route passed\n";
}
