#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "ggml-cpu.h"
#include "model/diffusion/minimax_h3.hpp"

static size_t forced_max_size = 0;
extern "C" size_t __real_ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t);
extern "C" size_t __wrap_ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    return forced_max_size ? forced_max_size : __real_ggml_backend_buft_get_max_size(buft);
}
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static std::vector<float> evaluate(ggml_backend_t backend, int batch, bool noncontiguous, int tokens_per_chunk) {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free);
    require(ctx != nullptr, "context unavailable");
    MiniMaxH3::MLP mlp(8,16);
    mlp.init(ctx.get());
    std::map<std::string,ggml_tensor*> weights;
    mlp.get_param_tensors(weights, "mlp");
    for (auto& entry : weights) {
        for (int64_t i = 0; i < ggml_nelements(entry.second); ++i) ggml_set_f32_1d(entry.second, i, std::sin(float(i) * .17f) * .2f);
    }
    auto input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, noncontiguous ? 14 : 7, batch);
    for (int64_t i = 0; i < ggml_nelements(input); ++i) ggml_set_f32_1d(input, i, std::cos(float(i) * .11f));
    if (noncontiguous) input = ggml_ext_slice(ctx.get(), input, 1, 0, 7, false);
    GGMLRunnerContext runner;
    runner.ggml_ctx = ctx.get();
    runner.backend = backend;
    const size_t token_bytes = 32 * sizeof(float) * batch;
    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
    forced_max_size = tokens_per_chunk > 0 ? token_bytes * tokens_per_chunk + alignment - 1 : 0;
    auto output = mlp.forward(&runner, input);
    forced_max_size = 0;
    require(output->ne[0] == 8 && output->ne[1] == 7 && output->ne[2] == batch, "MLP shape changed");
    auto graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    int expanded_matmuls = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT && node->ne[0] == 32) {
            ++expanded_matmuls;
            if (tokens_per_chunk > 0) require(ggml_nbytes(node) <= token_bytes * tokens_per_chunk, "expanded activation exceeds injected buffer limit");
        }
    }
    const int chunks = tokens_per_chunk > 0 ? (7 + tokens_per_chunk - 1) / tokens_per_chunk : 1;
    require(expanded_matmuls == chunks, "MLP did not split before expanded projection");
    require(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS, "CPU MLP compute failed");
    const auto* values = static_cast<float*>(output->data);
    return {values, values + ggml_nelements(output)};
}
int main() {
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(backend != nullptr, "CPU backend unavailable");
        for (int batch : {1,2}) {
            for (bool noncontiguous : {false,true}) {
                auto reference = evaluate(backend.get(), batch, noncontiguous, 0);
                float peak = 0.f;
                for (float value : reference) peak = std::max(peak, std::abs(value));
                require(peak > 0.000001f, "reference MLP output too small for meaningful parity check");
                for (int chunk : {1,2,3,7,8}) {
                    auto actual = evaluate(backend.get(), batch, noncontiguous, chunk);
                    require(actual.size() == reference.size(), "MLP size differs");
                    for (size_t i = 0; i < actual.size(); ++i) {
                        require(std::isfinite(actual[i]) && std::abs(actual[i] - reference[i]) <= 1e-10f + 1e-6f * std::abs(reference[i]),
                                "chunked MLP differs from original gated CPU path");
                    }
                }
            }
        }
        std::cout << "PASS MiniMax MLP actual CPU numerical parity, expanded-buffer limits, singleton/tail chunks and strided/batched inputs\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
