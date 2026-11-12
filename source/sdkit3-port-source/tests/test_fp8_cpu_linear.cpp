#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "ggml-cpu.h"
#include "core/ggml_extend.hpp"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void check(ggml_backend_t backend, ggml_type type, bool force_f32) {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
    require(ctx != nullptr, "GGML context unavailable");
    String2TensorStorage storage;
    storage["linear.weight"].type = type;
    storage["linear.weight"].expected_type = GGML_TYPE_COUNT;
    Linear layer(4, 2, true);
    layer.set_force_prec_f32(force_f32);
    layer.init(ctx.get(), storage, "linear");
    std::map<std::string, ggml_tensor*> params;
    layer.get_param_tensors(params, "linear");
    auto weight = params.at("linear.weight");
    require(weight->type == type, "test did not preserve source weight dtype");
    if (type == GGML_TYPE_F8_E4M3 || type == GGML_TYPE_F8_E5M2) {
        auto bytes = static_cast<uint8_t*>(weight->data);
        const uint8_t one = type == GGML_TYPE_F8_E4M3 ? 0x38 : 0x3c;
        for (int i = 0; i < 8; ++i) bytes[i] = one | (i % 2 ? 0x80 : 0);
    } else {
        for (int i = 0; i < 8; ++i) ggml_set_f32_1d(weight, i, i % 2 ? -1.f : 1.f);
    }
    auto bias = params.at("linear.bias");
    ggml_set_f32_1d(bias, 0, 0.5f);
    ggml_set_f32_1d(bias, 1, -0.5f);
    auto input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 3);
    for (int i = 0; i < 12; ++i) ggml_set_f32_1d(input, i, static_cast<float>(i + 1));
    GGMLRunnerContext runner;
    runner.ggml_ctx = ctx.get();
    runner.backend = backend;
    auto output = layer.forward(&runner, input);
    auto graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    bool bf16_matmul = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT) {
            require(node->src[0]->type != GGML_TYPE_F8_E4M3 && node->src[0]->type != GGML_TYPE_F8_E5M2,
                    "CPU graph still contains unsupported direct FP8 matmul");
            bf16_matmul |= node->src[0]->type == GGML_TYPE_BF16;
        }
    }
    if (type == GGML_TYPE_F8_E4M3 || type == GGML_TYPE_F8_E5M2) require(bf16_matmul, "FP8 weights did not use BF16 fallback");
    require(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS, "CPU linear compute failed");
    auto values = static_cast<float*>(output->data);
    require(ggml_nelements(output) == 6, "linear output shape changed");
    for (int i = 0; i < 6; ++i) require(std::abs(values[i] - (i % 2 ? -2.5f : -1.5f)) < 1e-5f, "linear result differs from scalar reference");
}

int main() {
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(backend != nullptr, "CPU backend unavailable");
        for (auto type : {GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E5M2, GGML_TYPE_BF16, GGML_TYPE_F32}) {
            for (bool force_f32 : {false, true}) check(backend.get(), type, force_f32);
        }
        std::cout << "PASS E4M3/E5M2 CPU BF16 fallback, bias and multirow numerical results, forced precision, BF16/F32 controls\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
