#include <iostream>
#include <memory>
#include <stdexcept>

#include "ggml-cpu.h"
#include "model/te/clip.hpp"

static void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void check_clip(ggml_backend_t backend, CLIPVersion version, bool force_f32) {
    String2TensorStorage storage;
    auto& source = storage["clip.embeddings.token_embedding.weight"];
    source.type = GGML_TYPE_BF16;
    source.expected_type = GGML_TYPE_COUNT;
    CLIPTextModelRunner runner(backend, storage, "clip", version, true, force_f32);
    auto weight = runner.model.get_token_embed_weight();
    require(weight->type == (force_f32 ? GGML_TYPE_F32 : GGML_TYPE_BF16), "BF16 CLIP dtype contract violated");
    std::map<ggml_tensor*, ggml_op> usages;
    runner.model.get_param_tensor_ops(usages);
    require(usages.count(weight) == 1 && usages.at(weight) == GGML_OP_GET_ROWS, "CLIP token weight missing get_rows usage annotation");
    std::map<std::string, ggml_tensor*> tensors;
    runner.model.get_param_tensors(tensors, "clip");
    auto position = tensors.at("clip.embeddings.position_embedding.weight");
    require(position->type == GGML_TYPE_F32 && usages.count(position) == 0, "position embedding behavior changed");
}

static void check_rows() {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({2 * 1024 * 1024, nullptr, false}), ggml_free);
    require(ctx != nullptr, "GGML context unavailable");
    constexpr int width = 7;
    constexpr int rows = 3;
    auto weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, width, rows);
    auto bytes = static_cast<ggml_bf16_t*>(weights->data);
    for (int i = 0; i < width * rows; ++i) {
        bytes[i] = ggml_fp32_to_bf16(static_cast<float>(i - 9) / 7.0f);
    }
    require(ggml_nbytes(weights) == width * rows * sizeof(ggml_bf16_t), "BF16 storage expanded");
    auto indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 3);
    auto ids = static_cast<int32_t*>(indices->data);
    ids[0] = 2;
    ids[1] = 0;
    ids[2] = 2;
    auto selected = ggml_get_rows(ctx.get(), weights, indices);
    auto graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, selected);
    require(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS, "BF16 CPU get_rows failed");
    auto values = static_cast<float*>(selected->data);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < width; ++col) {
            require(values[row * width + col] == ggml_bf16_to_fp32(bytes[ids[row] * width + col]),
                    "BF16 row gather differs from scalar reference");
        }
    }
}

int main() {
    auto backend = ggml_backend_cpu_init();
    int result = 0;
    try {
        require(backend != nullptr, "CPU backend unavailable");
        require(support_get_rows(GGML_TYPE_BF16), "BF16 embeddings incorrectly expanded to F32");
        check_rows();
        for (auto version : {OPENAI_CLIP_VIT_L_14, OPEN_CLIP_VIT_BIGG_14}) {
            check_clip(backend, version, false);
            check_clip(backend, version, true);
        }
        std::cout << "PASS BF16 CPU gathering, compact storage, CLIP-L/G dtype and operation contracts, forced F32 override\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        result = 1;
    }
    ggml_backend_free(backend);
    return result;
}
