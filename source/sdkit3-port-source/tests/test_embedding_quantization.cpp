#include <cmath>
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

static void check_cpu_rows(ggml_type type) {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({2 * 1024 * 1024, nullptr, false}), ggml_free);
    require(ctx != nullptr, "GGML context unavailable");
    constexpr int width = 256;
    constexpr int rows = 3;
    auto weights = ggml_new_tensor_2d(ctx.get(), type, width, rows);
    std::vector<float> source(width * rows);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < width; ++col) {
            source[row * width + col] = static_cast<float>(row + 1);
        }
    }
    require(ggml_quantize_chunk(type, source.data(), weights->data, 0, rows, width, nullptr) == ggml_nbytes(weights),
            "incorrect quantized storage size");
    require(ggml_nbytes(weights) < source.size() * sizeof(float), "quantized storage expanded");
    auto indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 3);
    auto ids = static_cast<int32_t*>(indices->data);
    ids[0] = 2;
    ids[1] = 0;
    ids[2] = 2;
    auto selected = ggml_get_rows(ctx.get(), weights, indices);
    auto graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, selected);
    require(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS, "CPU get_rows failed");
    auto values = static_cast<float*>(selected->data);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < width; ++col) {
            require(std::abs(values[row * width + col] - static_cast<float>(ids[row] + 1)) < 0.03f,
                    "CPU get_rows returned incorrect reordered/repeated row");
        }
    }
}

static void check_clip_type(ggml_backend_t backend, ggml_type type, bool force_f32, CLIPVersion version = OPENAI_CLIP_VIT_L_14) {
    String2TensorStorage storage;
    auto& tensor = storage["clip.embeddings.token_embedding.weight"];
    tensor.type = type;
    tensor.expected_type = GGML_TYPE_COUNT;
    CLIPTextModelRunner runner(backend, storage, "clip", version, true, force_f32);
    require(runner.model.get_token_embed_weight()->type == (force_f32 ? GGML_TYPE_F32 : type),
            "CLIP embedding dtype contract violated");
}

int main() {
    auto backend = ggml_backend_cpu_init();
    int result = 0;
    try {
        require(backend != nullptr, "CPU backend unavailable");
        for (auto type : {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
            require(support_get_rows(type), "K-quantized embeddings incorrectly rejected");
            check_cpu_rows(type);
            check_clip_type(backend, type, false);
            check_clip_type(backend, type, true);
            check_clip_type(backend, type, false, OPEN_CLIP_VIT_BIGG_14);
            check_clip_type(backend, type, true, OPEN_CLIP_VIT_BIGG_14);
            std::cout << "PASS " << ggml_type_name(type) << " CPU rows, preserved dtype, forced F32 override\n";
        }
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0}) {
            require(support_get_rows(type), "previously supported type rejected");
            check_clip_type(backend, type, false);
        }
        require(!support_get_rows(GGML_TYPE_I32) && !support_get_rows(GGML_TYPE_I64), "unrequested type enabled");
        std::cout << "PASS existing allowlist and unsupported-type behavior\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        result = 1;
    }
    ggml_backend_free(backend);
    return result;
}
