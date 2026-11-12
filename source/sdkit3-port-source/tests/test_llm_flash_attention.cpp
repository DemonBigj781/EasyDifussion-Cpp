#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ggml-cpu.h"
#include "model/te/llm.hpp"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<float> evaluate(ggml_backend_t backend, uint8_t flags, bool training, bool masked) {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free);
    require(ctx != nullptr, "GGML context unavailable");
    LLM::LLMConfig config;
    config.arch = LLM::LLMArch::QWEN3;
    config.hidden_size = 128;
    config.head_dim = 128;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.qk_norm = false;
    config.qkv_bias = false;
    config.attention_out_bias = false;
    LLM::Attention attention(config);
    attention.init(ctx.get());
    std::map<std::string, ggml_tensor*> weights;
    attention.get_param_tensors(weights, "attention");
    for (auto& [name, tensor] : weights) {
        ggml_set_zero(tensor);
        for (int i = 0; i < 128; ++i) ggml_set_f32_1d(tensor, i * 128 + i, 0.5f);
    }
    auto input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 8, 1);
    for (int i = 0; i < 1024; ++i) ggml_set_f32_1d(input, i, std::sin(static_cast<float>(i) * 0.1f));
    auto positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 8);
    for (int i = 0; i < 8; ++i) ggml_set_i32_1d(positions, i, i);
    ggml_tensor* mask = nullptr;
    if (masked) {
        mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 8);
        for (int q = 0; q < 8; ++q) {
            for (int k = 0; k < 8; ++k) ggml_set_f32_1d(mask, q * 8 + k, k > q ? -INFINITY : 0.f);
        }
    }
    GGMLRunnerContext runner;
    runner.ggml_ctx = ctx.get();
    runner.backend = backend;
    runner.flash_attn_enabled = flags;
    runner.training_graph = training;
    auto output = attention.forward(&runner, input, positions, mask);
    auto graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    int flash_count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            ++flash_count;
            require(ggml_flash_attn_ext_get_sage_attn(node) == ((flags & 2) != 0), "Sage preference lost");
        }
    }
    require(flash_count == ((flags != 0 && !training) ? 1 : 0), "LLM flash flag or training exclusion not respected");
    require(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS, "CPU attention failed");
    const auto values = static_cast<float*>(output->data);
    return std::vector<float>(values, values + ggml_nelements(output));
}

int main() {
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(backend != nullptr, "CPU backend unavailable");
        for (bool masked : {false, true}) {
            const auto reference = evaluate(backend.get(), 0, false, masked);
            for (uint8_t flags : {1, 3}) {
                for (bool training : {false, true}) {
                    const auto actual = evaluate(backend.get(), flags, training, masked);
                    require(actual.size() == reference.size(), "attention output shape changed");
                    for (size_t i = 0; i < actual.size(); ++i) {
                        require(std::isfinite(actual[i]) && std::abs(actual[i] - reference[i]) < 0.001f,
                                "attention output diverges from ordinary CPU path");
                    }
                }
            }
        }
        std::cout << "PASS LLM flash enable/disable, Sage preference, training exclusion, masked/unmasked CPU numerical parity\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
