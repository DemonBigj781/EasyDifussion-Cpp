#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "ggml-cpu.h"
#include "model/common/rope.hpp"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static float input_value(int d, int head, int token, int batch) {
    return std::sin(float(1 + d + 5 * head + 11 * token + 23 * batch) * 0.17f);
}

static void run_case(int width, int heads, int tokens, int batches, bool interleaved, bool strided, bool feature_strided) {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
    require(ctx != nullptr, "context allocation failed");
    auto storage = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, width + (feature_strided ? 4 : 0), heads, tokens * (strided ? 2 : 1), batches);
    ggml_set_zero(storage);
    auto x = (strided || feature_strided) ? ggml_view_4d(ctx.get(), storage, width, heads, tokens, batches,
                                  storage->nb[1], storage->nb[2] * (strided ? 2 : 1), storage->nb[3], 0) : storage;
    for (int b = 0; b < batches; ++b)
        for (int t = 0; t < tokens; ++t)
            for (int h = 0; h < heads; ++h)
                for (int d = 0; d < width; ++d)
                    *reinterpret_cast<float*>(static_cast<char*>(x->data) + d * x->nb[0] + h * x->nb[1] + t * x->nb[2] + b * x->nb[3]) = input_value(d, h, t, b);
    auto pe = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 2, 2, width / 2, tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int d = 0; d < width / 2; ++d) {
            const float angle = float(d + 3 * t) * 0.13f;
            const float values[] = {std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle)};
            for (int i = 0; i < 4; ++i) ggml_set_f32_1d(pe, (t * width / 2 + d) * 4 + i, values[i]);
        }
    }
    auto output = Rope::apply_rope(ctx.get(), x, pe, interleaved);
    require(output->ne[0] == width && output->ne[1] == tokens && output->ne[2] == heads * batches,
            "RoPE output shape changed");
    auto graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    require(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS, "CPU RoPE failed");
    for (int b = 0; b < batches; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int t = 0; t < tokens; ++t) {
                for (int d = 0; d < width / 2; ++d) {
                    const int d0 = interleaved ? 2 * d : d;
                    const int d1 = interleaved ? 2 * d + 1 : d + width / 2;
                    const float angle = float(d + 3 * t) * 0.13f;
                    const float a = input_value(d0, h, t, b), z = input_value(d1, h, t, b);
                    const float expected0 = a * std::cos(angle) - z * std::sin(angle);
                    const float expected1 = a * std::sin(angle) + z * std::cos(angle);
                    const int base = ((b * heads + h) * tokens + t) * width;
                    require(std::abs(ggml_get_f32_1d(output, base + d0) - expected0) < 1e-6f, "first RoPE component mismatch");
                    require(std::abs(ggml_get_f32_1d(output, base + d1) - expected1) < 1e-6f, "second RoPE component mismatch");
                }
            }
        }
    }
    if (!interleaved) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            require(ggml_graph_node(graph, i)->op != GGML_OP_REPEAT, "non-interleaved RoPE still duplicates input halves");
    }
}

int main() {
    try {
        int cases = 0;
        for (int width : {4, 16, 64})
            for (int heads : {1, 3})
                for (int tokens : {1, 2, 7})
                    for (int batches : {1, 2})
                        for (bool interleaved : {false, true})
                            for (bool strided : {false, true})
                                for (bool feature_strided : {false, true}) {
                                    run_case(width, heads, tokens, batches, interleaved, strided, feature_strided);
                                    ++cases;
                                }
        std::cout << "PASS " << cases << " CPU RoPE layouts, analytic parity and non-interleaved repeat elimination\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
