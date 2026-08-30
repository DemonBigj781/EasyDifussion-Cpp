#ifndef __SD_MODEL_DIFFUSION_CONTROL_HPP__
#define __SD_MODEL_DIFFUSION_CONTROL_HPP__

#include "model/common/block.hpp"
#include "model_loader.h"
#include "model_manager.h"

// SDXL ControlNets contain deep transformer stacks and exceed the original
// SD1.x-sized graph allocation. Keep enough headroom for full SDXL models and
// weight-adapter graph expansion without paying the full UNet graph overhead.
#define CONTROL_NET_GRAPH_SIZE 32768

// Xinsir's ControlNet Union adds a small CLIP-style transformer in front of
// the otherwise standard SDXL ControlNet.  Keep the implementation local to
// ControlNet so regular SD1.x/SDXL checkpoints do not gain parameters or graph
// nodes.
class ControlNetUnionMLP : public GGMLBlock {
public:
    explicit ControlNetUnionMLP(int64_t dim) {
        blocks["c_fc"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 4));
        blocks["c_proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim * 4, dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto c_fc   = std::dynamic_pointer_cast<Linear>(blocks["c_fc"]);
        auto c_proj = std::dynamic_pointer_cast<Linear>(blocks["c_proj"]);
        x           = c_fc->forward(ctx, x);
        x           = ggml_ext_gelu_quick(ctx->ggml_ctx, x, true);
        return c_proj->forward(ctx, x);
    }
};

class ControlNetUnionAttentionBlock : public GGMLBlock {
public:
    ControlNetUnionAttentionBlock(int64_t dim, int64_t heads) {
        // torch.nn.MultiheadAttention stores Q/K/V as one in_proj_weight.
        // Name conversion normalizes that spelling to this block hierarchy.
        blocks["attn"] = std::shared_ptr<GGMLBlock>(new MultiheadAttention(dim,
                                                                            heads,
                                                                            true,
                                                                            true,
                                                                            true,
                                                                            "q_proj",
                                                                            "k_proj",
                                                                            "v_proj",
                                                                            "in_proj",
                                                                            "out_proj"));
        blocks["ln_1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["mlp"]  = std::shared_ptr<GGMLBlock>(new ControlNetUnionMLP(dim));
        blocks["ln_2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto attn = std::dynamic_pointer_cast<MultiheadAttention>(blocks["attn"]);
        auto ln_1 = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_1"]);
        auto mlp  = std::dynamic_pointer_cast<ControlNetUnionMLP>(blocks["mlp"]);
        auto ln_2 = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_2"]);
        x         = ggml_add(ctx->ggml_ctx, x, attn->forward(ctx, ln_1->forward(ctx, x)));
        return ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, ln_2->forward(ctx, x)));
    }
};

class UniControlFDN : public GGMLBlock {
public:
    UniControlFDN(int64_t channels, int64_t feature_channels) {
        blocks["param_free_norm"] = std::shared_ptr<GGMLBlock>(new GroupNorm(32, channels, 1e-5f, false));
        blocks["conv_gamma"]      = std::shared_ptr<GGMLBlock>(new Conv2d(feature_channels, channels, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv_beta"]       = std::shared_ptr<GGMLBlock>(new Conv2d(feature_channels, channels, {3, 3}, {1, 1}, {1, 1}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* local_features) {
        auto norm  = std::dynamic_pointer_cast<GroupNorm>(blocks["param_free_norm"]);
        auto gamma = std::dynamic_pointer_cast<Conv2d>(blocks["conv_gamma"]);
        auto beta  = std::dynamic_pointer_cast<Conv2d>(blocks["conv_beta"]);
        auto normalized = norm->forward(ctx, x);
        auto gamma_out  = gamma->forward(ctx, local_features);
        auto beta_out   = beta->forward(ctx, local_features);
        auto modulated = ggml_mul(ctx->ggml_ctx, normalized, gamma_out);
        modulated      = ggml_add(ctx->ggml_ctx, modulated, normalized);
        return ggml_add(ctx->ggml_ctx, modulated, beta_out);
    }
};

class UniControlResBlock : public GGMLBlock {
protected:
    int64_t channels;
    int64_t out_channels;

public:
    UniControlResBlock(int64_t channels,
                       int64_t emb_channels,
                       int64_t out_channels,
                       int64_t feature_channels)
        : channels(channels), out_channels(out_channels) {
        blocks["norm_in"]      = std::shared_ptr<GGMLBlock>(new UniControlFDN(channels, feature_channels));
        blocks["in_layers.2"]  = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
        blocks["emb_layers.1"] = std::shared_ptr<GGMLBlock>(new Linear(emb_channels, out_channels));
        blocks["norm_out"]     = std::shared_ptr<GGMLBlock>(new UniControlFDN(out_channels, feature_channels));
        blocks["out_layers.3"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
        if (channels != out_channels) {
            blocks["skip_connection"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {1, 1}));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* emb,
                         ggml_tensor* local_features) {
        auto norm_in  = std::dynamic_pointer_cast<UniControlFDN>(blocks["norm_in"]);
        auto in_conv  = std::dynamic_pointer_cast<Conv2d>(blocks["in_layers.2"]);
        auto emb_proj = std::dynamic_pointer_cast<Linear>(blocks["emb_layers.1"]);
        auto norm_out = std::dynamic_pointer_cast<UniControlFDN>(blocks["norm_out"]);
        auto out_conv = std::dynamic_pointer_cast<Conv2d>(blocks["out_layers.3"]);

        auto h = norm_in->forward(ctx, x, local_features);
        h      = ggml_silu_inplace(ctx->ggml_ctx, h);
        h      = in_conv->forward(ctx, h);

        auto emb_out = ggml_silu(ctx->ggml_ctx, emb);
        emb_out      = emb_proj->forward(ctx, emb_out);
        emb_out      = ggml_reshape_4d(ctx->ggml_ctx, emb_out, 1, 1, emb_out->ne[0], emb_out->ne[1]);
        h            = ggml_add(ctx->ggml_ctx, h, emb_out);

        h = norm_out->forward(ctx, h, local_features);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);
        h = out_conv->forward(ctx, h);

        ggml_tensor* skip = x;
        auto skip_it = blocks.find("skip_connection");
        if (skip_it != blocks.end()) {
            skip = std::dynamic_pointer_cast<Conv2d>(skip_it->second)->forward(ctx, x);
        }
        return ggml_add(ctx->ggml_ctx, skip, h);
    }
};

class UniControlFeatureExtractor : public GGMLBlock {
public:
    UniControlFeatureExtractor() {
        blocks["pre_extractor.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(21, 32, {3, 3}, {1, 1}, {1, 1}));
        blocks["pre_extractor.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(32, 64, {3, 3}, {2, 2}, {1, 1}));
        blocks["pre_extractor.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(64, 64, {3, 3}, {1, 1}, {1, 1}));
        blocks["pre_extractor.6"] = std::shared_ptr<GGMLBlock>(new Conv2d(64, 128, {3, 3}, {2, 2}, {1, 1}));
        blocks["pre_extractor.8"] = std::shared_ptr<GGMLBlock>(new Conv2d(128, 128, {3, 3}, {1, 1}, {1, 1}));

        const int64_t feature_channels[] = {128, 192, 256, 384, 512};
        for (int i = 0; i < 4; ++i) {
            blocks["extractors." + std::to_string(i) + ".0"] = std::shared_ptr<GGMLBlock>(new Conv2d(feature_channels[i],
                                                                                                       feature_channels[i + 1],
                                                                                                       {3, 3},
                                                                                                       {2, 2},
                                                                                                       {1, 1}));
            blocks["zero_convs." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Conv2d(feature_channels[i + 1],
                                                                                               feature_channels[i + 1],
                                                                                               {3, 3},
                                                                                               {1, 1},
                                                                                               {1, 1}));
        }
    }

    std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        for (int i = 0; i <= 8; i += 2) {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["pre_extractor." + std::to_string(i)]);
            x         = conv->forward(ctx, x);
            x         = ggml_silu_inplace(ctx->ggml_ctx, x);
        }

        std::vector<ggml_tensor*> features;
        features.reserve(4);
        for (int i = 0; i < 4; ++i) {
            auto extractor = std::dynamic_pointer_cast<Conv2d>(blocks["extractors." + std::to_string(i) + ".0"]);
            auto zero_conv = std::dynamic_pointer_cast<Conv2d>(blocks["zero_convs." + std::to_string(i)]);
            x              = extractor->forward(ctx, x);
            x              = ggml_silu_inplace(ctx->ggml_ctx, x);
            features.push_back(zero_conv->forward(ctx, x));
        }
        return features;
    }
};

class ControlLiteBlock : public GGMLBlock {
protected:
    bool convolutional;
    bool use_diffusion_input;
    int model_channels = 320;

public:
    ControlLiteBlock(bool convolutional, bool use_diffusion_input)
        : convolutional(convolutional), use_diffusion_input(use_diffusion_input) {
        blocks["input_hint_block.0"]  = std::shared_ptr<GGMLBlock>(new Conv2d(3, 16, {3, 3}, {1, 1}, {1, 1}));
        blocks["input_hint_block.2"]  = std::shared_ptr<GGMLBlock>(new Conv2d(16, 16, {3, 3}, {1, 1}, {1, 1}));
        blocks["input_hint_block.4"]  = std::shared_ptr<GGMLBlock>(new Conv2d(16, 32, {3, 3}, {2, 2}, {1, 1}));
        blocks["input_hint_block.6"]  = std::shared_ptr<GGMLBlock>(new Conv2d(32, 32, {3, 3}, {1, 1}, {1, 1}));
        blocks["input_hint_block.8"]  = std::shared_ptr<GGMLBlock>(new Conv2d(32, 96, {3, 3}, {2, 2}, {1, 1}));
        blocks["input_hint_block.10"] = std::shared_ptr<GGMLBlock>(new Conv2d(96, 96, {3, 3}, {1, 1}, {1, 1}));
        blocks["input_hint_block.12"] = std::shared_ptr<GGMLBlock>(new Conv2d(96, 256, {3, 3}, {2, 2}, {1, 1}));
        blocks["input_hint_block.14"] = std::shared_ptr<GGMLBlock>(new Conv2d(256, model_channels, {3, 3}, {1, 1}, {1, 1}));

        int index = 0;
        if (use_diffusion_input) {
            blocks["input_blocks_control.0.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(4, model_channels, {3, 3}, {1, 1}, {1, 1}));
            blocks["zero_convs.0.0"]           = std::shared_ptr<GGMLBlock>(new Conv2d(model_channels, model_channels, {1, 1}));
            index = 1;
        }

        const int channel_mult[] = {1, 2, 4, 4};
        int channels             = model_channels;
        for (int level = 0; level < 4; ++level, ++index) {
            const int out_channels = channel_mult[level] * model_channels;
            const int kernel       = convolutional ? 3 : 1;
            const int padding      = convolutional ? 1 : 0;
            blocks["input_blocks_control." + std::to_string(index) + ".0"] =
                std::shared_ptr<GGMLBlock>(new Conv2d(channels,
                                                      out_channels,
                                                      {kernel, kernel},
                                                      {1, 1},
                                                      {padding, padding}));
            blocks["zero_convs." + std::to_string(index) + ".0"] =
                std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {1, 1}));
            channels = out_channels;
        }
    }

    ggml_tensor* hint_forward(GGMLRunnerContext* ctx, ggml_tensor* hint) {
        auto h = hint;
        for (int i = 0; i <= 14; i += 2) {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["input_hint_block." + std::to_string(i)]);
            h         = conv->forward(ctx, h);
            if (i != 14) {
                h = ggml_silu_inplace(ctx->ggml_ctx, h);
            }
        }
        return h;
    }

    std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                      ggml_tensor* x,
                                      ggml_tensor* hint) {
        auto guided_hint = hint_forward(ctx, hint);
        std::vector<ggml_tensor*> outs;
        outs.reserve(5);

        ggml_tensor* h = nullptr;
        int index      = 0;
        if (use_diffusion_input) {
            h = x;
            auto input = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks_control.0.0"]);
            auto zero  = std::dynamic_pointer_cast<Conv2d>(blocks["zero_convs.0.0"]);
            h          = input->forward(ctx, h);
            h          = ggml_add(ctx->ggml_ctx, h, guided_hint);
            guided_hint = nullptr;
            outs.push_back(zero->forward(ctx, h));
            index = 1;
        } else {
            // This first feature is intentionally not zero-convolved in the
            // reference implementation. It is injected before UNet block 1.
            h = guided_hint;
            guided_hint = nullptr;
            outs.push_back(h);
        }

        const int strides[] = {2, 2, 2, 1};
        for (int level = 0; level < 4; ++level, ++index) {
            auto block = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks_control." + std::to_string(index) + ".0"]);
            auto zero  = std::dynamic_pointer_cast<Conv2d>(blocks["zero_convs." + std::to_string(index) + ".0"]);
            h          = block->forward(ctx, h);
            h          = ggml_silu_inplace(ctx->ggml_ctx, h);
            h          = ggml_pool_2d(ctx->ggml_ctx,
                                      h,
                                      GGML_OP_POOL_AVG,
                                      strides[level],
                                      strides[level],
                                      strides[level],
                                      strides[level],
                                      0.f,
                                      0.f);
            outs.push_back(zero->forward(ctx, h));
        }
        return outs;
    }
};

class UniControlFeedForward : public GGMLBlock {
public:
    UniControlFeedForward(int64_t dim_in,
                          int64_t inner_dim,
                          int64_t dim_out) {
        blocks["net.0"] = std::shared_ptr<GGMLBlock>(new GEGLU(dim_in, inner_dim));
        blocks["net.2"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim_out));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto geglu = std::dynamic_pointer_cast<GEGLU>(blocks["net.0"]);
        auto out   = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);
        x          = geglu->forward(ctx, x);
        return out->forward(ctx, x);
    }
};

class UniControlGlobalAdapterBlock : public GGMLBlock {
public:
    UniControlGlobalAdapterBlock() {
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(768));
        blocks["ff1"]   = std::shared_ptr<GGMLBlock>(new UniControlFeedForward(768, 3072, 1536));
        blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(1536));
        blocks["ff2"]   = std::shared_ptr<GGMLBlock>(new UniControlFeedForward(1536, 6144, 3072));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto ff1   = std::dynamic_pointer_cast<UniControlFeedForward>(blocks["ff1"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto ff2   = std::dynamic_pointer_cast<UniControlFeedForward>(blocks["ff2"]);
        x          = ff1->forward(ctx, norm1->forward(ctx, x));
        x          = ff2->forward(ctx, norm2->forward(ctx, x));
        // PyTorch: rearrange(x, 'b (n d) -> b n d', n=4, d=768).
        return ggml_reshape_3d(ctx->ggml_ctx, x, 768, 4, x->ne[1]);
    }
};

struct UniControlGlobalAdapter : public GGMLRunner {
    UniControlGlobalAdapterBlock adapter;
    std::string weight_prefix;
    ggml_tensor* output_ggml = nullptr;

    UniControlGlobalAdapter(ggml_backend_t backend,
                            const String2TensorStorage& tensor_storage_map,
                            const std::string& prefix,
                            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager), weight_prefix(prefix) {
        adapter.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "uni_control_global_adapter";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        adapter.get_param_tensors(tensors, weight_prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& clip_embedding_tensor) {
        ggml_cgraph* gf      = new_graph_custom(4096);
        ggml_tensor* clip_embedding = make_input(clip_embedding_tensor);
        auto runner_ctx      = get_context();
        output_ggml          = adapter.forward(&runner_ctx, clip_embedding);
        ggml_set_output(output_ggml);
        ggml_build_forward_expand(gf, output_ggml);
        return gf;
    }

    sd::Tensor<float> compute(int n_threads,
                              const sd::Tensor<float>& clip_embedding) {
        if (clip_embedding.empty()) {
            return {};
        }
        auto input = clip_embedding;
        if (input.dim() == 1) {
            input = input.reshape({input.shape()[0], 1});
        }
        if (input.dim() != 2 || input.shape()[0] != 768) {
            LOG_ERROR("Uni-ControlNet global adapter expects a 768-dimensional CLIP image embedding");
            return {};
        }
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(input);
        };
        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true, true, true);
        if (!result.has_value()) {
            return {};
        }
        return restore_trailing_singleton_dims(sd::make_sd_tensor_from_ggml<float>(output_ggml), 3);
    }
};

/*
    =================================== ControlNet ===================================
    Reference: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/cldm/cldm.py

*/
class ControlNetBlock : public GGMLBlock {
protected:
    SDVersion version = VERSION_SD1;
    bool union_model  = false;
    bool uni_model    = false;
    bool lite_model   = false;
    bool lite_conv    = false;
    bool lite_uses_x  = false;
    bool middle_attention = true;
    // network hparams
    int in_channels                        = 4;
    int out_channels                       = 4;
    int hint_channels                      = 3;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult          = {1, 2, 4, 4};
    std::vector<int> transformer_depth     = {1, 1, 1, 1};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = 8;
    int num_head_channels                  = -1;   // channels // num_heads
    int context_dim                        = 768;  // 1024 for VERSION_SD2, 2048 for VERSION_SDXL
    bool use_linear_projection             = false;

    void init_params(ggml_context* ctx,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "") override {
        if (union_model) {
            ggml_type wtype        = get_type(prefix + "task_embedding", tensor_storage_map, GGML_TYPE_F32);
            params["task_embedding"] = ggml_new_tensor_2d(ctx, wtype, model_channels, 6);
        }
    }

    ggml_tensor* spatial_mean(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // Image tensors are [W,H,C,N]. ggml_mean reduces dimension zero, so
        // reduce W, swap the remaining H into dimension zero, then reduce H.
        x = ggml_mean(ctx->ggml_ctx, x);  // [1,H,C,N]
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        x = ggml_mean(ctx->ggml_ctx, x);  // [1,1,C,N]
        return ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[2], 1, x->ne[3]);  // [N,1,C]
    }

    ggml_tensor* union_task_embedding(GGMLRunnerContext* ctx, int control_type, int64_t batch) {
        GGML_ASSERT(control_type >= 0 && control_type < 6);
        ggml_tensor* task_embedding = params["task_embedding"];
        ggml_tensor* task = ggml_view_2d(ctx->ggml_ctx,
                                         task_embedding,
                                         model_channels,
                                         1,
                                         task_embedding->nb[1],
                                         static_cast<size_t>(control_type) * task_embedding->nb[1]);
        task = ggml_reshape_3d(ctx->ggml_ctx, task, model_channels, 1, 1);
        if (batch > 1) {
            task = ggml_repeat(ctx->ggml_ctx,
                               task,
                               ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, model_channels, 1, batch));
        }
        return task;
    }

public:
    int model_channels  = 320;
    int adm_in_channels = 2816;  // only for VERSION_SDXL

    ControlNetBlock(SDVersion version = VERSION_SD1,
                    bool union_model = false,
                    bool uni_model   = false,
                    int lite_kind    = 0,
                    int sdxl_depth   = 3)
        : version(version),
          union_model(union_model),
          uni_model(uni_model),
          lite_model(lite_kind != 0),
          lite_conv(lite_kind == 2 || lite_kind == 4),
          lite_uses_x(lite_kind == 3 || lite_kind == 4) {
        GGML_ASSERT(static_cast<int>(union_model) + static_cast<int>(uni_model) + static_cast<int>(lite_model) <= 1);
        if (lite_model) {
            GGML_ASSERT(sd_version_is_sd1(version));
            // The empty child key deliberately keeps the checkpoint's original
            // top-level names (input_hint_block.*, input_blocks_control.*,
            // zero_convs.*) instead of introducing a synthetic "lite." prefix.
            blocks[""] = std::shared_ptr<GGMLBlock>(new ControlLiteBlock(lite_conv, lite_uses_x));
            return;
        }
        if (sd_version_is_sd2(version)) {
            context_dim       = 1024;
            num_head_channels = 64;
            num_heads         = -1;
        } else if (sd_version_is_sdxl(version)) {
            context_dim           = 2048;
            channel_mult          = {1, 2, 4};
            num_head_channels     = 64;
            num_heads             = -1;
            if (!union_model && sdxl_depth <= 1) {
                // Diffusers' "small" SDXL ControlNet removes every spatial
                // transformer, including the middle-block attention.
                attention_resolutions.clear();
                transformer_depth = {0, 0, 0};
                middle_attention  = false;
            } else if (!union_model && sdxl_depth == 2) {
                // The "mid" topology keeps one transformer only at the
                // deepest resolution and one in the middle block.
                attention_resolutions = {4};
                transformer_depth     = {0, 0, 1};
            } else {
                attention_resolutions = {4, 2};
                transformer_depth     = {1, 2, 10};
            }
        } else if (version == VERSION_SVD) {
            in_channels       = 8;
            out_channels      = 4;
            context_dim       = 1024;
            adm_in_channels   = 768;
            num_head_channels = 64;
            num_heads         = -1;
        }

        blocks["time_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(model_channels, time_embed_dim));
        // time_embed_1 is nn.SiLU()
        blocks["time_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));

        if (union_model) {
            blocks["transformer_layes.0"] = std::shared_ptr<GGMLBlock>(new ControlNetUnionAttentionBlock(model_channels, 8));
            blocks["spatial_ch_projs"]    = std::shared_ptr<GGMLBlock>(new Linear(model_channels, model_channels));
            blocks["control_add_embedding.linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(256 * 6, time_embed_dim));
            blocks["control_add_embedding.linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));
        }
        if (uni_model) {
            GGML_ASSERT(sd_version_is_sd1(version));
            blocks["feature_extractor"] = std::shared_ptr<GGMLBlock>(new UniControlFeatureExtractor());
        }

        if (sd_version_is_sdxl(version) || version == VERSION_SVD) {
            blocks["label_emb.0.0"] = std::shared_ptr<GGMLBlock>(new Linear(adm_in_channels, time_embed_dim));
            // label_emb_1 is nn.SiLU()
            blocks["label_emb.0.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));
        }

        // input_blocks
        blocks["input_blocks.0.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, model_channels, {3, 3}, {1, 1}, {1, 1}));

        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch              = model_channels;
        int input_block_idx = 0;
        int ds              = 1;

        auto get_resblock = [&](int64_t channels, int64_t emb_channels, int64_t out_channels) -> ResBlock* {
            return new ResBlock(channels, emb_channels, out_channels);
        };

        auto get_attention_layer = [&](int64_t in_channels,
                                       int64_t n_head,
                                       int64_t d_head,
                                       int64_t depth,
                                       int64_t context_dim) -> SpatialTransformer* {
            return new SpatialTransformer(in_channels, n_head, d_head, depth, context_dim, use_linear_projection);
        };

        auto make_zero_conv = [&](int64_t channels) {
            return new Conv2d(channels, channels, {1, 1});
        };

        blocks["zero_convs.0.0"] = std::shared_ptr<GGMLBlock>(make_zero_conv(model_channels));

        if (!uni_model) {
            blocks["input_hint_block.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(hint_channels, 16, {3, 3}, {1, 1}, {1, 1}));
            // nn.SiLU()
            blocks["input_hint_block.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(16, 16, {3, 3}, {1, 1}, {1, 1}));
            // nn.SiLU()
            blocks["input_hint_block.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(16, 32, {3, 3}, {2, 2}, {1, 1}));
            // nn.SiLU()
            blocks["input_hint_block.6"] = std::shared_ptr<GGMLBlock>(new Conv2d(32, 32, {3, 3}, {1, 1}, {1, 1}));
            // nn.SiLU()
            blocks["input_hint_block.8"] = std::shared_ptr<GGMLBlock>(new Conv2d(32, 96, {3, 3}, {2, 2}, {1, 1}));
            // nn.SiLU()
            blocks["input_hint_block.10"] = std::shared_ptr<GGMLBlock>(new Conv2d(96, 96, {3, 3}, {1, 1}, {1, 1}));
            // nn.SiLU()
            blocks["input_hint_block.12"] = std::shared_ptr<GGMLBlock>(new Conv2d(96, 256, {3, 3}, {2, 2}, {1, 1}));
            // nn.SiLU()
            blocks["input_hint_block.14"] = std::shared_ptr<GGMLBlock>(new Conv2d(256, model_channels, {3, 3}, {1, 1}, {1, 1}));
        }

        size_t len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                if (uni_model && (input_block_idx == 1 || input_block_idx == 4 || input_block_idx == 7 || input_block_idx == 10)) {
                    static const int64_t inject_channels[] = {192, 256, 384, 512};
                    blocks[name] = std::shared_ptr<GGMLBlock>(new UniControlResBlock(ch,
                                                                                    time_embed_dim,
                                                                                    mult * model_channels,
                                                                                    inject_channels[i]));
                } else {
                    blocks[name] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, mult * model_channels));
                }

                ch = mult * model_channels;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    blocks[name]     = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                      n_head,
                                                                                      d_head,
                                                                                      transformer_depth[i],
                                                                                      context_dim));
                }
                blocks["zero_convs." + std::to_string(input_block_idx) + ".0"] = std::shared_ptr<GGMLBlock>(make_zero_conv(ch));
                input_block_chans.push_back(ch);
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(ch, ch));

                blocks["zero_convs." + std::to_string(input_block_idx) + ".0"] = std::shared_ptr<GGMLBlock>(make_zero_conv(ch));

                input_block_chans.push_back(ch);
                ds *= 2;
            }
        }

        // middle blocks
        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        blocks["middle_block.0"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));
        if (middle_attention) {
            blocks["middle_block.1"] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                      n_head,
                                                                                      d_head,
                                                                                      transformer_depth[transformer_depth.size() - 1],
                                                                                      context_dim));
        }
        blocks["middle_block.2"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));

        // middle_block_out
        blocks["middle_block_out.0"] = std::shared_ptr<GGMLBlock>(make_zero_conv(ch));
    }

    ggml_tensor* resblock_forward(std::string name,
                                  GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* emb) {
        auto block = std::dynamic_pointer_cast<ResBlock>(blocks[name]);
        return block->forward(ctx, x, emb);
    }

    ggml_tensor* attention_layer_forward(std::string name,
                                         GGMLRunnerContext* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* context) {
        auto block = std::dynamic_pointer_cast<SpatialTransformer>(blocks[name]);
        return block->forward(ctx, x, context);
    }

    ggml_tensor* input_hint_block_forward(GGMLRunnerContext* ctx,
                                          ggml_tensor* hint,
                                          ggml_tensor* emb,
                                          ggml_tensor* context) {
        int num_input_blocks = 15;
        auto h               = hint;
        for (int i = 0; i < num_input_blocks; i++) {
            if (i % 2 == 0) {
                auto block = std::dynamic_pointer_cast<Conv2d>(blocks["input_hint_block." + std::to_string(i)]);

                h = block->forward(ctx, h);
            } else {
                h = ggml_silu_inplace(ctx->ggml_ctx, h);
            }
        }
        return h;
    }

    std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                      ggml_tensor* x,
                                      ggml_tensor* hint,
                                      ggml_tensor* guided_hint,
                                      ggml_tensor* timesteps,
                                      ggml_tensor* context,
                                      ggml_tensor* y            = nullptr,
                                      ggml_tensor* control_types = nullptr,
                                      int control_type           = -1) {
        // x: [N, in_channels, h, w] or [N, in_channels/2, h, w]
        // timesteps: [N,]
        // context: [N, max_position, hidden_size] or [1, max_position, hidden_size]. for example, [N, 77, 768]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        if (lite_model) {
            auto lite = std::dynamic_pointer_cast<ControlLiteBlock>(blocks[""]);
            return lite->forward(ctx, x, hint);
        }

        if (context != nullptr) {
            if (context->ne[2] != x->ne[3]) {
                context = ggml_repeat(ctx->ggml_ctx, context, ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, context->ne[0], context->ne[1], x->ne[3]));
            }
        }

        if (y != nullptr) {
            if (y->ne[1] != x->ne[3]) {
                y = ggml_repeat(ctx->ggml_ctx, y, ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32, y->ne[0], x->ne[3]));
            }
        }

        auto time_embed_0     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.0"]);
        auto time_embed_2     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.2"]);
        auto input_blocks_0_0 = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks.0.0"]);
        auto zero_convs_0     = std::dynamic_pointer_cast<Conv2d>(blocks["zero_convs.0.0"]);

        auto middle_block_out = std::dynamic_pointer_cast<Conv2d>(blocks["middle_block_out.0"]);

        auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, model_channels);  // [N, model_channels]

        auto emb = time_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx->ggml_ctx, emb);
        emb      = time_embed_2->forward(ctx, emb);  // [N, time_embed_dim]

        if (union_model) {
            GGML_ASSERT(control_types != nullptr);
            auto control_add_1 = std::dynamic_pointer_cast<Linear>(blocks["control_add_embedding.linear_1"]);
            auto control_add_2 = std::dynamic_pointer_cast<Linear>(blocks["control_add_embedding.linear_2"]);
            auto flat_types = ggml_reshape_1d(ctx->ggml_ctx, control_types, ggml_nelements(control_types));
            auto type_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, flat_types, 256);
            type_emb = ggml_reshape_2d(ctx->ggml_ctx, type_emb, 256 * 6, x->ne[3]);
            type_emb = control_add_1->forward(ctx, type_emb);
            type_emb = ggml_silu_inplace(ctx->ggml_ctx, type_emb);
            type_emb = control_add_2->forward(ctx, type_emb);
            emb      = ggml_add(ctx->ggml_ctx, emb, type_emb);
        }

        // SDXL/SVD
        if (y != nullptr) {
            auto label_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.0"]);
            auto label_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.2"]);

            auto label_emb = label_embed_0->forward(ctx, y);
            label_emb      = ggml_silu_inplace(ctx->ggml_ctx, label_emb);
            label_emb      = label_embed_2->forward(ctx, label_emb);  // [N, time_embed_dim]

            emb = ggml_add(ctx->ggml_ctx, emb, label_emb);  // [N, time_embed_dim]
        }

        std::vector<ggml_tensor*> outs;
        std::vector<ggml_tensor*> local_features;

        if (uni_model) {
            auto feature_extractor = std::dynamic_pointer_cast<UniControlFeatureExtractor>(blocks["feature_extractor"]);
            local_features         = feature_extractor->forward(ctx, hint);
            GGML_ASSERT(local_features.size() == 4);
            // The runner reserves output zero as a non-ControlNet marker. It is
            // never cached for Uni-ControlNet, but keeps the standard thirteen
            // control outputs aligned at indices 1..13.
            outs.push_back(local_features[0]);
        } else {
            if (guided_hint == nullptr) {
                guided_hint = input_hint_block_forward(ctx, hint, emb, context);
            }
            sd::ggml_graph_cut::mark_graph_cut(guided_hint, "controlnet.input_hint_block", "guided_hint");
            outs.push_back(guided_hint);
        }

        // input_blocks

        // input block 0
        auto h = input_blocks_0_0->forward(ctx, x);
        if (union_model) {
            auto transformer = std::dynamic_pointer_cast<ControlNetUnionAttentionBlock>(blocks["transformer_layes.0"]);
            auto spatial_proj = std::dynamic_pointer_cast<Linear>(blocks["spatial_ch_projs"]);
            auto condition_token = spatial_mean(ctx, guided_hint);
            condition_token = ggml_add(ctx->ggml_ctx,
                                       condition_token,
                                       union_task_embedding(ctx, control_type, x->ne[3]));
            auto sample_token = spatial_mean(ctx, h);
            auto tokens       = ggml_concat(ctx->ggml_ctx, condition_token, sample_token, 1);
            tokens            = transformer->forward(ctx, tokens);
            auto condition_bias = ggml_view_3d(ctx->ggml_ctx,
                                                tokens,
                                                model_channels,
                                                1,
                                                tokens->ne[2],
                                                tokens->nb[1],
                                                tokens->nb[2],
                                                0);
            condition_bias = spatial_proj->forward(ctx, condition_bias);
            condition_bias = ggml_reshape_4d(ctx->ggml_ctx,
                                              condition_bias,
                                              1,
                                              1,
                                              model_channels,
                                              x->ne[3]);
            guided_hint = ggml_add(ctx->ggml_ctx, guided_hint, condition_bias);
        }
        if (!uni_model) {
            h = ggml_add(ctx->ggml_ctx, h, guided_hint);
        }
        sd::ggml_graph_cut::mark_graph_cut(h, "controlnet.input_blocks.0", "h");
        outs.push_back(zero_convs_0->forward(ctx, h));

        // input block 1-11
        size_t len_mults    = channel_mult.size();
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                if (uni_model && (input_block_idx == 1 || input_block_idx == 4 || input_block_idx == 7 || input_block_idx == 10)) {
                    const int feature_index = input_block_idx == 1 ? 0 : input_block_idx == 4 ? 1 : input_block_idx == 7 ? 2 : 3;
                    auto block = std::dynamic_pointer_cast<UniControlResBlock>(blocks[name]);
                    h          = block->forward(ctx, h, emb, local_features[feature_index]);
                } else {
                    h = resblock_forward(name, ctx, h, emb);  // [N, mult*model_channels, h, w]
                }
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    h                = attention_layer_forward(name, ctx, h, context);  // [N, mult*model_channels, h, w]
                }
                sd::ggml_graph_cut::mark_graph_cut(h,
                                                   "controlnet.input_blocks." + std::to_string(input_block_idx),
                                                   "h");

                auto zero_conv = std::dynamic_pointer_cast<Conv2d>(blocks["zero_convs." + std::to_string(input_block_idx) + ".0"]);

                outs.push_back(zero_conv->forward(ctx, h));
            }
            if (i != len_mults - 1) {
                ds *= 2;
                input_block_idx += 1;

                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                auto block       = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = block->forward(ctx, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]
                sd::ggml_graph_cut::mark_graph_cut(h,
                                                   "controlnet.input_blocks." + std::to_string(input_block_idx),
                                                   "h");

                auto zero_conv = std::dynamic_pointer_cast<Conv2d>(blocks["zero_convs." + std::to_string(input_block_idx) + ".0"]);

                outs.push_back(zero_conv->forward(ctx, h));
            }
        }
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        h = resblock_forward("middle_block.0", ctx, h, emb);             // [N, 4*model_channels, h/8, w/8]
        if (middle_attention) {
            h = attention_layer_forward("middle_block.1", ctx, h, context);  // [N, 4*model_channels, h/8, w/8]
        }
        h = resblock_forward("middle_block.2", ctx, h, emb);             // [N, 4*model_channels, h/8, w/8]
        sd::ggml_graph_cut::mark_graph_cut(h, "controlnet.middle_block", "h");

        // out
        outs.push_back(middle_block_out->forward(ctx, h));
        return outs;
    }

    bool is_union() const {
        return union_model;
    }

    bool is_uni() const {
        return uni_model;
    }

    bool is_lite() const {
        return lite_model;
    }

    bool can_cache_guided_hint() const {
        return !union_model && !uni_model && !lite_model;
    }
};

struct ControlNet : public GGMLRunner {
    SDVersion version = VERSION_SD1;
    ControlNetBlock control_net;
    std::string weight_prefix;

    std::vector<ggml_tensor*> control_outputs_ggml;
    ggml_tensor* guided_hint_output_ggml = nullptr;
    std::vector<sd::Tensor<float>> controls;
    bool guided_hint_cached = false;
    std::shared_ptr<ModelManager> owned_model_manager;
    ggml_backend_t params_backend = nullptr;

    static const char* guided_hint_cache_name() {
        return "controlnet.guided_hint";
    }

    static bool has_tensor(const String2TensorStorage& tensors,
                           const std::string& prefix,
                           const std::string& name) {
        return tensors.find(prefix.empty() ? name : prefix + "." + name) != tensors.end();
    }

    static int detect_lite_kind(const String2TensorStorage& tensors,
                                const std::string& prefix) {
        if (!has_tensor(tensors, prefix, "input_blocks_control.0.0.weight")) {
            return 0;
        }
        const bool uses_x = has_tensor(tensors, prefix, "input_blocks_control.4.0.weight");
        const std::string probe_name = "input_blocks_control." + std::to_string(uses_x ? 1 : 0) + ".0.weight";
        auto it = tensors.find(prefix.empty() ? probe_name : prefix + "." + probe_name);
        if (it == tensors.end() || it->second.n_dims < 4) {
            return 0;
        }
        const bool conv = it->second.ne[0] == 3 && it->second.ne[1] == 3;
        return (uses_x ? 3 : 1) + (conv ? 1 : 0);
    }

    static bool has_tensor_prefix(const String2TensorStorage& tensors,
                                  const std::string& prefix,
                                  const std::string& name_prefix) {
        const std::string full_prefix = prefix.empty() ? name_prefix : prefix + "." + name_prefix;
        return std::any_of(tensors.begin(), tensors.end(), [&](const auto& item) {
            return item.first.rfind(full_prefix, 0) == 0;
        });
    }

    static int detect_sdxl_depth(const String2TensorStorage& tensors,
                                 const std::string& prefix,
                                 SDVersion version) {
        if (!sd_version_is_sdxl(version)) {
            return 3;
        }
        // Mid checkpoints retain norm/projection shells at input block 4 but
        // omit its transformer layers, so topology detection must probe an
        // actual transformer block rather than the surrounding attention
        // module. Full has depth two at block 4; mid has depth one only at
        // block 7; small has no transformer blocks at either resolution.
        if (has_tensor_prefix(tensors, prefix, "input_blocks.4.1.transformer_blocks.1.")) {
            return 3;
        }
        if (has_tensor_prefix(tensors, prefix, "input_blocks.7.1.transformer_blocks.0.")) {
            return 2;
        }
        return 1;
    }

    static int union_slot_for_type(int type) {
        switch (type) {
            case CONTROL_NET_TYPE_OPENPOSE:
                return 0;
            case CONTROL_NET_TYPE_DEPTH:
                return 1;
            case CONTROL_NET_TYPE_HED:
            case CONTROL_NET_TYPE_SKETCH:
                return 2;
            case CONTROL_NET_TYPE_CANNY:
            case CONTROL_NET_TYPE_MLSD:
                return 3;
            case CONTROL_NET_TYPE_NORMAL:
                return 4;
            case CONTROL_NET_TYPE_SEGMENT:
                return 5;
            default:
                return -1;
        }
    }

    static int uni_slot_for_type(int type) {
        switch (type) {
            case CONTROL_NET_TYPE_CANNY:
                return 0;
            case CONTROL_NET_TYPE_MLSD:
                return 1;
            case CONTROL_NET_TYPE_HED:
                return 2;
            case CONTROL_NET_TYPE_SKETCH:
                return 3;
            case CONTROL_NET_TYPE_OPENPOSE:
                return 4;
            case CONTROL_NET_TYPE_DEPTH:
                return 5;
            case CONTROL_NET_TYPE_SEGMENT:
                return 6;
            default:
                return -1;
        }
    }

    ControlNet(ggml_backend_t backend,
               ggml_backend_t params_backend_,
               const String2TensorStorage& tensor_storage_map      = {},
               SDVersion version                                   = VERSION_SD1,
               const std::string& prefix                           = "",
               std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          version(version),
          control_net(version,
                      has_tensor(tensor_storage_map, prefix, "task_embedding"),
                      has_tensor(tensor_storage_map, prefix, "feature_extractor.pre_extractor.0.weight"),
                      detect_lite_kind(tensor_storage_map, prefix),
                      detect_sdxl_depth(tensor_storage_map, prefix, version)),
          weight_prefix(prefix),
          params_backend(params_backend_) {
        control_net.init(params_ctx, tensor_storage_map, prefix);
        if (control_net.is_union()) {
            LOG_INFO("detected Xinsir ControlNet Union (six SDXL control types)");
        }
        if (control_net.is_uni()) {
            LOG_INFO("detected Uni-ControlNet local adapter (seven SD1.5 control slots)");
        }
        if (control_net.is_lite()) {
            LOG_INFO("detected ControlNet-LITE (%s, diffusion input: %s)",
                     detect_lite_kind(tensor_storage_map, prefix) % 2 == 0 ? "3x3 convolution" : "1x1 MLP",
                     detect_lite_kind(tensor_storage_map, prefix) >= 3 ? "yes" : "no");
        }
        if (sd_version_is_sdxl(version) && !control_net.is_union()) {
            static const char* depth_names[] = {"unknown", "small", "mid", "full"};
            LOG_INFO("detected %s SDXL ControlNet topology",
                     depth_names[detect_sdxl_depth(tensor_storage_map, prefix, version)]);
        }
    }

    ~ControlNet() override {
        free_control_ctx();
    }

    void free_control_ctx() {
        guided_hint_output_ggml = nullptr;
        guided_hint_cached      = false;
        control_outputs_ggml.clear();
        controls.clear();
        free_cache_ctx_and_buffer();
    }

    std::string get_desc() override {
        return "control_net";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        control_net.get_param_tensors(tensors, weight_prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                             const sd::Tensor<float>& hint_tensor,
                             const sd::Tensor<float>& timesteps_tensor,
                             const sd::Tensor<float>& context_tensor = {},
                             const sd::Tensor<float>& y_tensor       = {},
                             const sd::Tensor<float>& control_types_tensor = {},
                             int control_type = -1) {
        ggml_cgraph* gf = new_graph_custom(CONTROL_NET_GRAPH_SIZE);

        ggml_tensor* x         = make_input(x_tensor);
        ggml_tensor* hint      = nullptr;
        ggml_tensor* timesteps = make_input(timesteps_tensor);
        ggml_tensor* context   = make_optional_input(context_tensor);
        ggml_tensor* y         = make_optional_input(y_tensor);
        ggml_tensor* control_types = make_optional_input(control_types_tensor);

        guided_hint_output_ggml = nullptr;
        control_outputs_ggml.clear();

        ggml_tensor* guided_hint_input = nullptr;
        if (guided_hint_cached) {
            guided_hint_input = get_cache_tensor_by_name(guided_hint_cache_name());
            if (guided_hint_input == nullptr) {
                guided_hint_cached = false;
            }
        }
        if (guided_hint_input == nullptr) {
            hint = make_input(hint_tensor);
        }

        auto runner_ctx = get_context();

        auto outs = control_net.forward(&runner_ctx,
                                        x,
                                        hint,
                                        guided_hint_input,
                                        timesteps,
                                        context,
                                        y,
                                        control_types,
                                        control_type);

        if (guided_hint_input == nullptr && !outs.empty() && control_net.can_cache_guided_hint()) {
            guided_hint_output_ggml = outs[0];
            ggml_set_output(guided_hint_output_ggml);
            cache(guided_hint_cache_name(), guided_hint_output_ggml);
            ggml_build_forward_expand(gf, guided_hint_output_ggml);
        }

        const size_t first_control = control_net.is_lite() ? 0 : 1;
        control_outputs_ggml.reserve(outs.size() > first_control ? outs.size() - first_control : 0);
        for (size_t i = first_control; i < outs.size(); i++) {
            ggml_tensor* control_output = outs[i];
            ggml_set_output(control_output);
            ggml_build_forward_expand(gf, control_output);
            control_outputs_ggml.push_back(control_output);
        }

        return gf;
    }

    std::optional<std::vector<sd::Tensor<float>>> compute(int n_threads,
                                                          const sd::Tensor<float>& x,
                                                          const sd::Tensor<float>& hint,
                                                          const sd::Tensor<float>& timesteps,
                                                          const sd::Tensor<float>& context = {},
                                                          const sd::Tensor<float>& y       = {},
                                                          int control_type                 = CONTROL_NET_TYPE_CANNY) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 77, 768]) or [1, max_position, hidden_size]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        int model_control_type = control_type;
        sd::Tensor<float> model_hint = hint;
        sd::Tensor<float> control_types;
        if (control_net.is_union()) {
            model_control_type = union_slot_for_type(control_type);
            if (model_control_type < 0) {
                LOG_ERROR("unsupported semantic control type %d for ControlNet Union", control_type);
                return std::nullopt;
            }
            const int64_t batch = x.shape().size() > 3 ? x.shape()[3] : 1;
            control_types = sd::zeros<float>({6, batch});
            for (int64_t i = 0; i < batch; ++i) {
                control_types.values()[static_cast<size_t>(i * 6 + model_control_type)] = 1.f;
            }
        } else if (control_net.is_uni()) {
            model_control_type = uni_slot_for_type(control_type);
            if (model_control_type < 0) {
                LOG_ERROR("unsupported semantic control type %d for Uni-ControlNet", control_type);
                return std::nullopt;
            }
            if (hint.shape().size() != 4 || hint.shape()[2] != 3) {
                LOG_ERROR("Uni-ControlNet expects an RGB condition image");
                return std::nullopt;
            }
            const int64_t width  = hint.shape()[0];
            const int64_t height = hint.shape()[1];
            const int64_t batch  = hint.shape()[3];
            const size_t plane   = static_cast<size_t>(width * height);
            model_hint           = sd::zeros<float>({width, height, 21, batch});
            for (int64_t n = 0; n < batch; ++n) {
                for (int64_t c = 0; c < 3; ++c) {
                    const size_t src_offset = static_cast<size_t>(n * 3 + c) * plane;
                    const size_t dst_offset = static_cast<size_t>(n * 21 + model_control_type * 3 + c) * plane;
                    std::copy_n(hint.values().begin() + static_cast<std::ptrdiff_t>(src_offset),
                                static_cast<std::ptrdiff_t>(plane),
                                model_hint.values().begin() + static_cast<std::ptrdiff_t>(dst_offset));
                }
            }
        }
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(x, model_hint, timesteps, context, y, control_types, model_control_type);
        };

        auto compute_result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false, true);
        if (!compute_result.has_value()) {
            return std::nullopt;
        }

        guided_hint_cached = control_net.can_cache_guided_hint() && get_cache_tensor_by_name(guided_hint_cache_name()) != nullptr;
        controls.clear();
        controls.reserve(control_outputs_ggml.size());
        for (ggml_tensor* control : control_outputs_ggml) {
            auto control_host = restore_trailing_singleton_dims(sd::make_sd_tensor_from_ggml<float>(control), 4);
            GGML_ASSERT(!control_host.empty());
            controls.push_back(std::move(control_host));
        }
        return controls;
    }

    bool load_from_file(const std::string& file_path, int n_threads) {
        LOG_INFO("loading control net from '%s'", file_path.c_str());
        std::map<std::string, ggml_tensor*> tensors;
        control_net.get_param_tensors(tensors);

        auto manager = std::dynamic_pointer_cast<ModelManager>(weight_manager.lock());
        if (manager == nullptr) {
            owned_model_manager = std::make_shared<ModelManager>();
            weight_manager      = owned_model_manager;
            manager             = owned_model_manager;
        }

        ModelLoader& model_loader = manager->loader();
        if (!model_loader.init_from_file_and_convert_name(file_path)) {
            LOG_ERROR("init control net model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        manager->set_n_threads(n_threads);
        if (!manager->register_param_tensors("ControlNet",
                                             std::move(tensors),
                                             ModelManager::ResidencyMode::ParamBackend,
                                             runtime_backend,
                                             params_backend) ||
            !manager->validate_registered_tensors()) {
            LOG_ERROR("register control net tensors with model manager failed");
            return false;
        }

        LOG_INFO("control net model loaded");
        return true;
    }
};

#endif  // __SD_MODEL_DIFFUSION_CONTROL_HPP__
