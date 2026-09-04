#include "features/attention/flash/common/flash_attention.hpp"

#include "ggml-cpu-impl.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace flash = edcpp::api::attention::flash;

static int native_calls = 0;

extern "C" void ggml_abort(const char*, int, const char*, ...) {
    std::abort();
}

extern "C" void ggml_compute_forward_flash_attn_ext(
        const ggml_compute_params* params,
        ggml_tensor* destination) {
    ++native_calls;
    assert(params != nullptr);
    assert(params->ith == 0 && params->nth == 1);
    assert(destination != nullptr && destination->op == GGML_OP_FLASH_ATTN_EXT);
    assert(destination->src[0]->type == GGML_TYPE_F32);
    assert(destination->src[1]->type == GGML_TYPE_F16);
    assert(destination->src[2]->type == GGML_TYPE_F16);
    assert(destination->src[3] != nullptr && destination->src[3]->type == GGML_TYPE_F16);
    assert(destination->ne[0] == 4 && destination->ne[1] == 2);
    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    std::memcpy(&scale, destination->op_params, sizeof(scale));
    std::memcpy(&max_bias, destination->op_params + 1, sizeof(max_bias));
    std::memcpy(&logit_softcap, destination->op_params + 2, sizeof(logit_softcap));
    assert(std::fabs(scale - 0.5f) < 1e-6f);
    assert(std::fabs(max_bias - 1.0f) < 1e-6f);
    assert(std::fabs(logit_softcap - 2.0f) < 1e-6f);
}

int main() {
    using edcpp::api::Backend;

    std::vector<float> query(8);
    std::vector<std::uint16_t> key(12);
    std::vector<std::uint16_t> value(12);
    std::vector<std::uint16_t> mask(6);
    std::vector<float> output(8);

    flash::Request request;
    request.query = {query.data(), flash::DType::f32, 4, 2, 1, 1, {}};
    request.key = {key.data(), flash::DType::f16, 4, 3, 1, 1, {}};
    request.value = {value.data(), flash::DType::f16, 4, 3, 1, 1, {}};
    request.mask = {mask.data(), flash::DType::f16, 3, 2, 1, 1, {}};
    request.output = {output.data(), flash::DType::f32, 4, 2, 1, 1, {}};
    request.scale = 0.5f;
    request.max_bias = 1.0f;
    request.logit_softcap = 2.0f;

    const auto caps = flash::capabilities(Backend::cpu);
    assert(caps.forward && caps.additive_mask && caps.alibi_bias);
    assert(caps.logit_softcap && caps.grouped_query && caps.f32_accumulation);

    auto invalid = request;
    invalid.query.data = nullptr;
    assert(!flash::validate(Backend::cpu, invalid).ok);

    invalid = request;
    invalid.output.tokens = 3;
    assert(!flash::validate(Backend::cpu, invalid).ok);

    invalid = request;
    invalid.execution.thread_count = 0;
    assert(!flash::validate(Backend::cpu, invalid).ok);

    invalid = request;
    invalid.execution.thread_count = 2;
    assert(!flash::validate(Backend::cpu, invalid).ok);

    invalid = request;
    invalid.mask.data = nullptr;
    assert(!flash::validate(Backend::cpu, invalid).ok);

    assert(!flash::validate(Backend::none, request).ok);
    assert(native_calls == 0);
    assert(flash::validate(Backend::cpu, request).ok);
    assert(flash::forward(Backend::cpu, request).ok);
    assert(native_calls == 1);

    std::cout << "CPU FlashAttention normalized Common route passed\n";
}
