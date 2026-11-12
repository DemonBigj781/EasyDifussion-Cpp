#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "ggml-cpu.h"
#include "core/ggml_extend.hpp"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void check_merge(int rank, int outputs, bool convolution) {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
    require(ctx != nullptr, "GGML context unavailable");
    const int inputs = convolution ? 18 : 5;
    auto down = convolution ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 3, 3, 2, rank)
                            : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, inputs, rank);
    auto up = convolution ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, 1, rank, outputs)
                          : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rank, outputs);
    auto d = static_cast<float*>(down->data);
    auto u = static_cast<float*>(up->data);
    for (int i = 0; i < inputs * rank; ++i) d[i] = static_cast<float>((i % 11) - 5) / 8;
    for (int i = 0; i < outputs * rank; ++i) u[i] = static_cast<float>((i % 7) - 3) / 4;
    auto merged = ggml_ext_merge_lora(ctx.get(), down, up);
    require(merged->ne[0] == inputs && merged->ne[1] == outputs, "LoRA output dimensions changed");
    auto graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, merged);
    require(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS, "CPU LoRA merge failed");
    auto result = static_cast<float*>(merged->data);
    for (int out = 0; out < outputs; ++out) {
        for (int in = 0; in < inputs; ++in) {
            float expected = 0;
            for (int k = 0; k < rank; ++k) expected += u[out * rank + k] * d[k * inputs + in];
            require(std::abs(result[out * inputs + in] - expected) < 1e-6f, "LoRA merge differs from scalar reference");
        }
    }
}

int main() {
    try {
        for (int rank : {1, 3}) {
            for (int outputs : {1, 4}) {
                check_merge(rank, outputs, false);
                // Singleton high dimensions in convolution tensors have different
                // legacy layout semantics; cover the ordinary multi-output case.
                if (outputs > 1) check_merge(rank, outputs, true);
            }
        }
        std::cout << "PASS singleton and multi-output linear LoRA, rank-one and rank-three, convolution control\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
