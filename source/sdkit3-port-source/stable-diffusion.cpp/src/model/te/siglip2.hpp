#ifndef SD_MODEL_TE_SIGLIP2_HPP
#define SD_MODEL_TE_SIGLIP2_HPP

#include "core/ggml_extend.hpp"

namespace SigLIP2 {

constexpr int image_size = 512;
constexpr int patch_size = 16;
constexpr int hidden_size = 768;
constexpr int token_count = 1024;
constexpr int layer_count = 12;

// Match Python round (ties to even), independently of the host rounding mode.
inline int resized_extent(int extent, int longest) {
    const double value = extent * (double(image_size) / longest);
    const int lower = int(std::floor(value));
    const double fraction = value - lower;
    return std::max(1, lower + (fraction > 0.5 || (fraction == 0.5 && lower % 2)));
}

struct ResizeTap {
    int first;
    std::vector<int32_t> weights;
};

inline std::vector<ResizeTap> bilinear_taps(int input, int output) {
    std::vector<ResizeTap> taps(output);
    const double step = double(input) / output;
    const double support = std::max(1.0, step);
    for (int i = 0; i < output; ++i) {
        const double center = (i + 0.5) * step;
        auto& tap = taps[i];
        tap.first = std::max(0, int(center - support + 0.5));
        const int end = std::min(input, int(center + support + 0.5));
        std::vector<double> weights(end - tap.first);
        double sum = 0;
        for (int j = tap.first; j < end; ++j) {
            const double weight = std::max(0.0, 1.0 - std::abs((j - center + 0.5) / support));
            weights[j - tap.first] = weight;
            sum += weight;
        }
        for (double weight : weights) {
            tap.weights.push_back(int32_t(weight / sum * (1 << 22) + 0.5));
        }
    }
    return taps;
}

inline sd::Tensor<float> preprocess(const uint8_t* rgb, int width, int height) {
    if (rgb == nullptr || width <= 0 || height <= 0) {
        throw std::invalid_argument("SigLIP2 requires a non-empty RGB image");
    }
    const int longest = std::max(width, height);
    const int w = resized_extent(width, longest);
    const int h = resized_extent(height, longest);
    const auto horizontal = bilinear_taps(width, w);
    const auto vertical = bilinear_taps(height, h);
    std::vector<uint8_t> intermediate(size_t(w) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < w; ++x) {
            const auto& tap = horizontal[x];
            for (int c = 0; c < 3; ++c) {
                int64_t total = 1 << 21;
                for (size_t k = 0; k < tap.weights.size(); ++k) {
                    total += int64_t(rgb[(size_t(y) * width + tap.first + k) * 3 + c]) * tap.weights[k];
                }
                intermediate[(size_t(y) * w + x) * 3 + c] = uint8_t(std::clamp<int64_t>(total >> 22, 0, 255));
            }
        }
    }
    sd::Tensor<float> pixels({image_size, image_size, 3, 1});
    std::fill(pixels.values().begin(), pixels.values().end(), -1.0f);
    const int left = (image_size - w) / 2;
    const int top = (image_size - h) / 2;
    for (int y = 0; y < h; ++y) {
        const auto& tap = vertical[y];
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                int64_t total = 1 << 21;
                for (size_t k = 0; k < tap.weights.size(); ++k) {
                    total += int64_t(intermediate[((tap.first + k) * w + x) * 3 + c]) * tap.weights[k];
                }
                const float byte = float(std::clamp<int64_t>(total >> 22, 0, 255));
                pixels.values()[(c * image_size + y + top) * image_size + x + left] = (byte / 255.0f - 0.5f) / 0.5f;
            }
        }
    }
    return pixels;
}

struct Embeddings : GGMLBlock {
    void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
        params["patch_embedding.weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, patch_size, patch_size, 3, hidden_size);
        params["patch_embedding.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        params["position_embedding.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, token_count);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* pixels) {
        auto g = ctx->ggml_ctx;
        auto weight = params.at("patch_embedding.weight");
        auto patches = ggml_im2col(g, weight, pixels, patch_size, patch_size, 0, 0, 1, 1, true, GGML_TYPE_F32);
        patches = ggml_reshape_3d(g, patches, 3 * patch_size * patch_size, token_count, pixels->ne[3]);
        weight = ggml_reshape_2d(g, weight, 3 * patch_size * patch_size, hidden_size);
        auto x = ggml_mul_mat(g, weight, patches);
        ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        x = ggml_add(g, x, params.at("patch_embedding.bias"));
        return ggml_add(g, x, params.at("position_embedding.weight"));
    }
};

struct MLP : GGMLBlock {
    MLP() {
        blocks["fc1"] = std::make_shared<Linear>(hidden_size, 3072, true, true, true);
        blocks["fc2"] = std::make_shared<Linear>(3072, hidden_size, true, true, true);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        x = std::static_pointer_cast<Linear>(blocks.at("fc1"))->forward(ctx, x);
        // GGML's CPU GELU uses a half-precision lookup table. SigLIP2's
        // gelu_pytorch_tanh must retain F32 precision through all 12 layers.
        auto g = ctx->ggml_ctx;
        auto cubic = ggml_mul(g, ggml_sqr(g, x), x);
        auto inner = ggml_add(g, x, ggml_scale(g, cubic, 0.044715f));
        inner = ggml_tanh(g, ggml_scale(g, inner, 0.7978845608028654f));
        x = ggml_mul(g, x, ggml_scale_bias(g, inner, 0.5f, 0.5f));
        return std::static_pointer_cast<Linear>(blocks.at("fc2"))->forward(ctx, x);
    }
};

struct Attention : GGMLBlock {
    Attention() {
        for (const auto* name : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            blocks[name] = std::make_shared<Linear>(hidden_size, hidden_size, true, true, true);
        }
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto q = std::static_pointer_cast<Linear>(blocks.at("q_proj"))->forward(ctx, x);
        auto k = std::static_pointer_cast<Linear>(blocks.at("k_proj"))->forward(ctx, x);
        auto v = std::static_pointer_cast<Linear>(blocks.at("v_proj"))->forward(ctx, x);
        x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 12, nullptr, false,
                                   0, 1.f, false, GGML_PREC_F32);
        return std::static_pointer_cast<Linear>(blocks.at("out_proj"))->forward(ctx, x);
    }
};

struct EncoderLayer : GGMLBlock {
    EncoderLayer() {
        blocks["layer_norm1"] = std::make_shared<LayerNorm>(hidden_size, 1e-6f);
        blocks["layer_norm2"] = std::make_shared<LayerNorm>(hidden_size, 1e-6f);
        blocks["self_attn"] = std::make_shared<Attention>();
        blocks["mlp"] = std::make_shared<MLP>();
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto norm1 = std::static_pointer_cast<LayerNorm>(blocks.at("layer_norm1"));
        auto norm2 = std::static_pointer_cast<LayerNorm>(blocks.at("layer_norm2"));
        auto attention = std::static_pointer_cast<Attention>(blocks.at("self_attn"));
        auto mlp = std::static_pointer_cast<MLP>(blocks.at("mlp"));
        x = ggml_add(ctx->ggml_ctx, x, attention->forward(ctx, norm1->forward(ctx, x)));
        return ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, norm2->forward(ctx, x)));
    }
};

struct VisionModel : GGMLBlock {
    VisionModel() {
        blocks["embeddings"] = std::make_shared<Embeddings>();
        for (int i = 0; i < layer_count; ++i) {
            blocks["encoder.layers." + std::to_string(i)] = std::make_shared<EncoderLayer>();
        }
        blocks["post_layernorm"] = std::make_shared<LayerNorm>(hidden_size, 1e-6f);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* pixels) {
        auto x = std::static_pointer_cast<Embeddings>(blocks.at("embeddings"))->forward(ctx, pixels);
        for (int i = 0; i < layer_count; ++i) {
            x = std::static_pointer_cast<EncoderLayer>(blocks.at("encoder.layers." + std::to_string(i)))->forward(ctx, x);
        }
        return std::static_pointer_cast<LayerNorm>(blocks.at("post_layernorm"))->forward(ctx, x);
    }
};

struct Runner : GGMLRunner {
    VisionModel model;
    const std::string prefix;

    Runner(ggml_backend_t backend, const String2TensorStorage& storage,
           std::shared_ptr<RunnerWeightManager> manager = nullptr,
           const std::string& prefix = "vision_model")
        : GGMLRunner(backend, manager), prefix(prefix) {
        if (!sd_backend_is_cpu(backend) && !sd_backend_is(backend, "CUDA")) {
            throw std::invalid_argument("SigLIP2 requires a verified full-precision CPU or CUDA backend; select clipvision=cpu or clipvision=CUDA0");
        }
        model.init(params_ctx, storage, prefix);
        std::map<std::string, ggml_tensor*> tensors;
        get_param_tensors(tensors);
        for (const auto& entry : tensors) {
            const auto found = storage.find(entry.first);
            if (found == storage.end()) {
                throw std::invalid_argument("Missing SigLIP2 tensor: " + entry.first);
            }
            for (int axis = 0; axis < 4; ++axis) {
                if (found->second.ne[axis] != entry.second->ne[axis]) {
                    throw std::invalid_argument("Unsupported SigLIP2 shape: " + entry.first);
                }
            }
        }
        for (const auto& entry : storage) {
            if (starts_with(entry.first, prefix + ".") &&
                !starts_with(entry.first, prefix + ".head.") && tensors.count(entry.first) == 0) {
                throw std::invalid_argument("Unsupported SigLIP2 tensor: " + entry.first);
            }
        }
    }
    std::string get_desc() override { return "siglip2"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        model.get_param_tensors(tensors, prefix);
    }
    sd::Tensor<float> compute(int threads, const sd::Tensor<float>& pixels) {
        if (pixels.shape() != std::vector<int64_t>{image_size, image_size, 3, 1}) {
            throw std::invalid_argument("SigLIP2 expects one 512x512 RGB image");
        }
        auto graph = [&]() {
            auto gf = ggml_new_graph(compute_ctx);
            auto ctx = get_context();
            auto output = model.forward(&ctx, make_input(pixels));
            ggml_build_forward_expand(gf, output);
            return gf;
        };
        return take_or_empty(GGMLRunner::compute<float>(graph, threads, true, true, true));
    }
};

}  // namespace SigLIP2
#endif
