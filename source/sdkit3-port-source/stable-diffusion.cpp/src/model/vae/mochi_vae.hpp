#ifndef __SD_MODEL_VAE_MOCHI_VAE_HPP__
#define __SD_MODEL_VAE_MOCHI_VAE_HPP__

#include <memory>
#include <vector>

#include "core/ggml_graph_cut.h"
#include "model/common/block.hpp"
#include "model/vae/vae.hpp"
#include "model_loader.h"

namespace MOCHI {

class MochiSpatialGroupNorm : public GroupNorm {
public:
    explicit MochiSpatialGroupNorm(int64_t channels)
        : GroupNorm(32, channels, 1e-6f, true) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // The reference applies GroupNorm independently to every video frame.
        x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
        x = GroupNorm::forward(ctx, x);
        return ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
    }
};

class MochiCausalConv3d : public Conv3d {
public:
    MochiCausalConv3d(int64_t in_channels,
                      int64_t out_channels,
                      std::tuple<int, int, int> kernel,
                      bool bias = true)
        : Conv3d(in_channels, out_channels, kernel, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, bias) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // Official Mochi VAE uses causal replicate padding: two copies of the
        // first frame and one replicated pixel on every spatial edge.
        if (std::get<0>(kernel_size) == 3) {
            auto first = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
            x          = ggml_concat(ctx->ggml_ctx, first, x, 2);
            x          = ggml_concat(ctx->ggml_ctx, first, x, 2);
        }
        if (std::get<2>(kernel_size) == 3) {
            auto left  = ggml_ext_slice(ctx->ggml_ctx, x, 0, 0, 1);
            auto right = ggml_ext_slice(ctx->ggml_ctx, x, 0, x->ne[0] - 1, x->ne[0]);
            x          = ggml_concat(ctx->ggml_ctx, left, x, 0);
            x          = ggml_concat(ctx->ggml_ctx, x, right, 0);
        }
        if (std::get<1>(kernel_size) == 3) {
            auto top    = ggml_ext_slice(ctx->ggml_ctx, x, 1, 0, 1);
            auto bottom = ggml_ext_slice(ctx->ggml_ctx, x, 1, x->ne[1] - 1, x->ne[1]);
            x           = ggml_concat(ctx->ggml_ctx, top, x, 1);
            x           = ggml_concat(ctx->ggml_ctx, x, bottom, 1);
        }
        return Conv3d::forward(ctx, x);
    }
};

class MochiVideoLinear : public Linear {
public:
    MochiVideoLinear(int64_t in_channels, int64_t out_channels, bool bias = true)
        : Linear(in_channels, out_channels, bias) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // [W,H,T,C] -> [C,W,H,T], apply linear to channels, then restore.
        x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 3, 0, 1, 2));
        x = Linear::forward(ctx, x);
        return ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 3, 0));
    }
};

class MochiVaeResBlock : public GGMLBlock {
public:
    explicit MochiVaeResBlock(int64_t channels) {
        blocks["stack.0"] = std::make_shared<MochiSpatialGroupNorm>(channels);
        blocks["stack.2"] = std::make_shared<MochiCausalConv3d>(channels, channels, std::make_tuple(3, 3, 3));
        blocks["stack.3"] = std::make_shared<MochiSpatialGroupNorm>(channels);
        blocks["stack.5"] = std::make_shared<MochiCausalConv3d>(channels, channels, std::make_tuple(3, 3, 3));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto residual = x;
        x = std::dynamic_pointer_cast<MochiSpatialGroupNorm>(blocks["stack.0"])->forward(ctx, x);
        x = ggml_silu(ctx->ggml_ctx, x);
        x = std::dynamic_pointer_cast<MochiCausalConv3d>(blocks["stack.2"])->forward(ctx, x);
        x = std::dynamic_pointer_cast<MochiSpatialGroupNorm>(blocks["stack.3"])->forward(ctx, x);
        x = ggml_silu(ctx->ggml_ctx, x);
        x = std::dynamic_pointer_cast<MochiCausalConv3d>(blocks["stack.5"])->forward(ctx, x);
        return ggml_add(ctx->ggml_ctx, residual, x);
    }
};

static ggml_tensor* mochi_depth_to_space_time(ggml_context* ctx,
                                              ggml_tensor* x,
                                              int64_t temporal_expansion,
                                              int64_t spatial_expansion) {
    // Exact inverse of rearrange(
    //   "B (C st sh sw) T H W -> B C (T st) (H sh) (W sw)").
    // GGML video tensors are [W, H, T, B*C].
    const int64_t t = x->ne[2];
    const int64_t h = x->ne[1];
    const int64_t w = x->ne[0];
    const int64_t c = x->ne[3] /
                      (temporal_expansion * spatial_expansion * spatial_expansion);

    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 1, 3, 2));
    x = ggml_reshape_4d(ctx,
                        x,
                        w,
                        h,
                        spatial_expansion,
                        spatial_expansion * temporal_expansion * c * t);
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));
    x = ggml_reshape_4d(ctx,
                        x,
                        spatial_expansion * w,
                        h,
                        spatial_expansion,
                        temporal_expansion * c * t);
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(ctx,
                        x,
                        spatial_expansion * w * spatial_expansion * h,
                        temporal_expansion,
                        c,
                        t);
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 1, 3, 2));
    x = ggml_reshape_4d(ctx,
                        x,
                        spatial_expansion * w,
                        spatial_expansion * h,
                        temporal_expansion * t,
                        c);

    // Causal decoding maps the first latent to one frame, not st frames.
    if (temporal_expansion > 1) {
        x = ggml_ext_slice(ctx, x, 2, temporal_expansion - 1, x->ne[2]);
    }
    return x;
}

class MochiVaeInitialBlock : public GGMLBlock {
public:
    MochiVaeInitialBlock() {
        blocks["0"] = std::make_shared<Conv3d>(12, 768, std::make_tuple(1, 1, 1));
        for (int i = 1; i <= 3; ++i) {
            blocks[std::to_string(i)] = std::make_shared<MochiVaeResBlock>(768);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        x = std::dynamic_pointer_cast<Conv3d>(blocks["0"])->forward(ctx, x);
        for (int i = 1; i <= 3; ++i) {
            x = std::dynamic_pointer_cast<MochiVaeResBlock>(blocks[std::to_string(i)])->forward(ctx, x);
        }
        return x;
    }
};

class MochiVaeUpsampleBlock : public GGMLBlock {
    int num_res_blocks;
    int temporal_expansion;
    int spatial_expansion;

public:
    MochiVaeUpsampleBlock(int64_t in_channels,
                          int64_t out_channels,
                          int num_res_blocks,
                          int temporal_expansion,
                          int spatial_expansion)
        : num_res_blocks(num_res_blocks),
          temporal_expansion(temporal_expansion),
          spatial_expansion(spatial_expansion) {
        for (int i = 0; i < num_res_blocks; ++i) {
            blocks["blocks." + std::to_string(i)] = std::make_shared<MochiVaeResBlock>(in_channels);
        }
        blocks["proj"] = std::make_shared<MochiVideoLinear>(
            in_channels,
            out_channels * temporal_expansion * spatial_expansion * spatial_expansion);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        for (int i = 0; i < num_res_blocks; ++i) {
            x = std::dynamic_pointer_cast<MochiVaeResBlock>(blocks["blocks." + std::to_string(i)])->forward(ctx, x);
        }
        x = std::dynamic_pointer_cast<MochiVideoLinear>(blocks["proj"])->forward(ctx, x);
        return mochi_depth_to_space_time(ctx->ggml_ctx, x, temporal_expansion, spatial_expansion);
    }
};

class MochiVaeFinalBlock : public GGMLBlock {
public:
    MochiVaeFinalBlock() {
        for (int i = 0; i < 3; ++i) {
            blocks[std::to_string(i)] = std::make_shared<MochiVaeResBlock>(128);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        for (int i = 0; i < 3; ++i) {
            x = std::dynamic_pointer_cast<MochiVaeResBlock>(blocks[std::to_string(i)])->forward(ctx, x);
        }
        return x;
    }
};

class MochiDecoder : public GGMLBlock {
public:
    MochiDecoder() {
        blocks["blocks.0"] = std::make_shared<MochiVaeInitialBlock>();
        blocks["blocks.1"] = std::make_shared<MochiVaeUpsampleBlock>(768, 512, 6, 3, 2);
        blocks["blocks.2"] = std::make_shared<MochiVaeUpsampleBlock>(512, 256, 4, 2, 2);
        blocks["blocks.3"] = std::make_shared<MochiVaeUpsampleBlock>(256, 128, 3, 1, 2);
        blocks["blocks.4"] = std::make_shared<MochiVaeFinalBlock>();
        blocks["output_proj"] = std::make_shared<MochiVideoLinear>(128, 3);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        x = std::dynamic_pointer_cast<MochiVaeInitialBlock>(blocks["blocks.0"])->forward(ctx, x);
        sd::ggml_graph_cut::mark_graph_cut(x, "mochi_vae.blocks.0", "x");
        for (int i = 1; i <= 3; ++i) {
            x = std::dynamic_pointer_cast<MochiVaeUpsampleBlock>(blocks["blocks." + std::to_string(i)])->forward(ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(x, "mochi_vae.blocks." + std::to_string(i), "x");
        }
        x = std::dynamic_pointer_cast<MochiVaeFinalBlock>(blocks["blocks.4"])->forward(ctx, x);
        x = ggml_silu(ctx->ggml_ctx, x);
        return std::dynamic_pointer_cast<MochiVideoLinear>(blocks["output_proj"])->forward(ctx, x);
    }
};

struct MochiVAERunner : public VAE {
    MochiDecoder decoder;
    std::string decoder_prefix;

    MochiVAERunner(ggml_backend_t backend,
                   const String2TensorStorage& tensor_storage_map      = {},
                   const std::string& prefix                           = "",
                   std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : VAE(VERSION_MOCHI, backend, prefix, weight_manager) {
        // Comfy's full VAE uses decoder.blocks.*, while Genmo's official
        // decoder-only checkpoint uses bare blocks.*.
        const std::string full_prefix = prefix + ".decoder";
        if (tensor_storage_map.find(full_prefix + ".blocks.0.0.weight") != tensor_storage_map.end()) {
            decoder_prefix = full_prefix;
        } else {
            decoder_prefix = prefix;
        }
        decoder.init(params_ctx, tensor_storage_map, decoder_prefix);
    }

    std::string get_desc() override {
        return "mochi_vae";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        decoder.get_param_tensors(tensors, decoder_prefix);
    }

    int get_encoder_output_channels(int input_channels) override {
        SD_UNUSED(input_channels);
        return 12;
    }

    sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output,
                                             std::shared_ptr<RNG> rng) override {
        SD_UNUSED(rng);
        return vae_to_diffusion_latents(vae_output);
    }

    std::pair<sd::Tensor<float>, sd::Tensor<float>> latent_stats(const sd::Tensor<float>& x) const {
        const int channel_dim = x.dim() == 5 ? 3 : 2;
        std::vector<int64_t> shape(static_cast<size_t>(x.dim()), 1);
        shape[static_cast<size_t>(channel_dim)] = 12;
        auto mean = sd::Tensor<float>::from_vector({
            -0.06730895953510081f, -0.038011381506090416f, -0.07477820912866141f,
            -0.05565264470995561f, 0.012767231469026969f, -0.04703542746246419f,
            0.043896967884726704f, -0.09346305707025976f, -0.09918314763016893f,
            -0.008729793427399178f, -0.011931556316503654f, -0.0321993391887285f});
        auto std = sd::Tensor<float>::from_vector({
            0.9263795028493863f, 0.9248894543193766f, 0.9393059390890617f,
            0.959253732819592f, 0.8244560132752793f, 0.917259975397747f,
            0.9294154431013696f, 1.3720942357788521f, 0.881393668867029f,
            0.9168315692124348f, 0.9185249279345552f, 0.9274757570805041f});
        mean.reshape_(shape);
        std.reshape_(shape);
        return {std::move(mean), std::move(std)};
    }

    sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
        auto stats = latent_stats(latents);
        return latents * stats.second + stats.first;
    }

    sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
        auto stats = latent_stats(latents);
        return (latents - stats.first) / stats.second;
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& z) {
        ggml_cgraph* graph = new_graph_custom(32768 + 4096 * z.shape()[2]);
        auto input         = make_input(z);
        auto runner_ctx    = get_context();
        auto output        = decoder.forward(&runner_ctx, input);
        ggml_build_forward_expand(graph, output);
        return graph;
    }

    sd::Tensor<float> _compute(const int n_threads,
                               const sd::Tensor<float>& z,
                               bool decode_graph) override {
        if (!decode_graph) {
            LOG_ERROR("Mochi 1 Preview is text-to-video; native Mochi VAE encoding is not enabled");
            return {};
        }
        auto get_graph = [&]() { return build_graph(z); };
        return restore_trailing_singleton_dims(
            GGMLRunner::compute<float>(get_graph, n_threads, false, false, false),
            z.dim());
    }
};

}  // namespace MOCHI

#endif  // __SD_MODEL_VAE_MOCHI_VAE_HPP__
