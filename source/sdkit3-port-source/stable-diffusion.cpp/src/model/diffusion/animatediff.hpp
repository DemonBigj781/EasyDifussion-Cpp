#ifndef __SD_MODEL_DIFFUSION_ANIMATEDIFF_HPP__
#define __SD_MODEL_DIFFUSION_ANIMATEDIFF_HPP__

#include "core/ggml_extend.hpp"
#include "model/common/block.hpp"

// AnimateDiff (https://arxiv.org/abs/2307.04725) motion modules for SD 1.5
// and the SDXL beta topology.
namespace AnimateDiff {

    struct MotionModuleConfig {
        int max_frames                     = 32;
        int64_t num_heads                  = 8;
        int norm_num_groups                = 32;
        std::vector<int64_t> down_channels = {320, 640, 1280, 1280};
        std::vector<int64_t> up_channels   = {1280, 1280, 640, 320};
        int num_down_motion_per_block      = 2;
        int num_up_motion_per_block        = 3;
        bool enable_mid_block              = false;
        int64_t mid_channels               = 1280;
        bool use_diffusers_names            = false;
    };

    class TemporalAttention : public GGMLBlock {
    protected:
        int64_t channels;
        int64_t num_heads;
        int max_frames;
        bool use_diffusers_names;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            const char* name = use_diffusers_names ? "pos_embed.pe" : "pos_encoder.pe";
            params[name]      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, max_frames, 1);
        }

    public:
        TemporalAttention(int64_t channels, int64_t num_heads, int max_frames, bool use_diffusers_names)
            : channels(channels),
              num_heads(num_heads),
              max_frames(max_frames),
              use_diffusers_names(use_diffusers_names) {
            blocks["to_q"]     = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, false));
            blocks["to_k"]     = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, false));
            blocks["to_v"]     = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, false));
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto to_q   = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k   = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v   = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            int64_t C = x->ne[0];
            int64_t F = x->ne[1];

            auto pe   = params[use_diffusers_names ? "pos_embed.pe" : "pos_encoder.pe"];
            auto pe_f = (F == pe->ne[1])
                            ? pe
                            : ggml_view_3d(ctx->ggml_ctx, pe, C, F, 1, pe->nb[1], pe->nb[2], 0);
            auto x_pe = ggml_add(ctx->ggml_ctx, x, ggml_repeat(ctx->ggml_ctx, pe_f, x));

            auto q = to_q->forward(ctx, x_pe);
            auto k = to_k->forward(ctx, x_pe);
            auto v = to_v->forward(ctx, x_pe);

            auto a = ggml_ext_attention_ext(ctx->ggml_ctx,
                                            ctx->backend,
                                            q,
                                            k,
                                            v,
                                            (int)num_heads,
                                            nullptr,
                                            false,
                                            ctx->flash_attn_enabled);
            return to_out->forward(ctx, a);
        }
    };

    class TemporalTransformerBlock : public GGMLBlock {
    public:
        TemporalTransformerBlock(int64_t channels, int64_t num_heads, int max_frames, bool use_diffusers_names) {
            blocks[use_diffusers_names ? "attn1" : "attention_blocks.0"] =
                std::make_shared<TemporalAttention>(channels, num_heads, max_frames, use_diffusers_names);
            blocks[use_diffusers_names ? "attn2" : "attention_blocks.1"] =
                std::make_shared<TemporalAttention>(channels, num_heads, max_frames, use_diffusers_names);
            blocks[use_diffusers_names ? "norm1" : "norms.0"] = std::shared_ptr<GGMLBlock>(new LayerNorm(channels));
            blocks[use_diffusers_names ? "norm2" : "norms.1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(channels));
            blocks["ff"] = std::make_shared<FeedForward>(channels, channels, 4, FeedForward::Activation::GEGLU);
            blocks[use_diffusers_names ? "norm3" : "ff_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(channels));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            const char* attn0_name = blocks.find("attn1") != blocks.end() ? "attn1" : "attention_blocks.0";
            const char* attn1_name = blocks.find("attn2") != blocks.end() ? "attn2" : "attention_blocks.1";
            const char* norm0_name = blocks.find("norm1") != blocks.end() ? "norm1" : "norms.0";
            const char* norm1_name = blocks.find("norm2") != blocks.end() ? "norm2" : "norms.1";
            const char* norm2_name = blocks.find("norm3") != blocks.end() ? "norm3" : "ff_norm";
            auto attn0   = std::dynamic_pointer_cast<TemporalAttention>(blocks[attn0_name]);
            auto attn1   = std::dynamic_pointer_cast<TemporalAttention>(blocks[attn1_name]);
            auto norm0   = std::dynamic_pointer_cast<LayerNorm>(blocks[norm0_name]);
            auto norm1   = std::dynamic_pointer_cast<LayerNorm>(blocks[norm1_name]);
            auto ff      = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);
            auto ff_norm = std::dynamic_pointer_cast<LayerNorm>(blocks[norm2_name]);

            auto r = x;
            x      = ggml_add(ctx->ggml_ctx, attn0->forward(ctx, norm0->forward(ctx, x)), r);

            r = x;
            x = ggml_add(ctx->ggml_ctx, attn1->forward(ctx, norm1->forward(ctx, x)), r);

            r = x;
            x = ggml_add(ctx->ggml_ctx, ff->forward(ctx, ff_norm->forward(ctx, x)), r);

            return x;
        }
    };

    class TemporalTransformer : public GGMLBlock {
    public:
        TemporalTransformer(int64_t channels,
                            int64_t num_heads,
                            int norm_num_groups,
                            int max_frames,
                            bool use_diffusers_names) {
            blocks["norm"]                 = std::shared_ptr<GGMLBlock>(new GroupNorm(norm_num_groups, channels));
            blocks["proj_in"]              = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, true));
            blocks["transformer_blocks.0"] =
                std::make_shared<TemporalTransformerBlock>(channels, num_heads, max_frames, use_diffusers_names);
            blocks["proj_out"]             = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, int64_t num_frames) {
            auto norm     = std::dynamic_pointer_cast<GroupNorm>(blocks["norm"]);
            auto proj_in  = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
            auto tb0      = std::dynamic_pointer_cast<TemporalTransformerBlock>(blocks["transformer_blocks.0"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            GGML_ASSERT(x->ne[3] == num_frames);

            auto residual = x;
            auto h        = norm->forward(ctx, x);

            h = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h, 2, 3, 0, 1));
            h = ggml_reshape_3d(ctx->ggml_ctx, h, C, num_frames, W * H);
            h = proj_in->forward(ctx, h);
            h = tb0->forward(ctx, h);
            h = proj_out->forward(ctx, h);
            h = ggml_reshape_4d(ctx->ggml_ctx, h, C, num_frames, W, H);
            h = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h, 2, 3, 0, 1));

            return ggml_add(ctx->ggml_ctx, h, residual);
        }
    };

    class MotionModule : public GGMLBlock {
    public:
        MotionModule(int64_t channels,
                     int64_t num_heads,
                     int norm_num_groups,
                     int max_frames,
                     bool use_diffusers_names) {
            blocks["temporal_transformer"] =
                std::make_shared<TemporalTransformer>(channels,
                                                     num_heads,
                                                     norm_num_groups,
                                                     max_frames,
                                                     use_diffusers_names);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, int64_t num_frames) {
            auto tt = std::dynamic_pointer_cast<TemporalTransformer>(blocks["temporal_transformer"]);
            return tt->forward(ctx, x, num_frames);
        }
    };

    class AnimateDiffModel : public GGMLBlock {
    public:
        MotionModuleConfig config;

        AnimateDiffModel(const MotionModuleConfig& cfg)
            : config(cfg) {
            for (int i = 0; i < static_cast<int>(cfg.down_channels.size()); ++i) {
                int64_t ch = cfg.down_channels[i];
                for (int j = 0; j < cfg.num_down_motion_per_block; ++j) {
                    blocks["down_blocks." + std::to_string(i) + ".motion_modules." + std::to_string(j)] =
                        std::make_shared<MotionModule>(ch,
                                                       cfg.num_heads,
                                                       cfg.norm_num_groups,
                                                       cfg.max_frames,
                                                       cfg.use_diffusers_names);
                }
            }
            for (int i = 0; i < static_cast<int>(cfg.up_channels.size()); ++i) {
                int64_t ch = cfg.up_channels[i];
                for (int j = 0; j < cfg.num_up_motion_per_block; ++j) {
                    blocks["up_blocks." + std::to_string(i) + ".motion_modules." + std::to_string(j)] =
                        std::make_shared<MotionModule>(ch,
                                                       cfg.num_heads,
                                                       cfg.norm_num_groups,
                                                       cfg.max_frames,
                                                       cfg.use_diffusers_names);
                }
            }
            if (cfg.enable_mid_block) {
                blocks["mid_block.motion_modules.0"] =
                    std::make_shared<MotionModule>(cfg.mid_channels,
                                                   cfg.num_heads,
                                                   cfg.norm_num_groups,
                                                   cfg.max_frames,
                                                   cfg.use_diffusers_names);
            }
        }

        std::shared_ptr<MotionModule> motion(const std::string& key) {
            auto it = blocks.find(key);
            if (it == blocks.end())
                return nullptr;
            return std::dynamic_pointer_cast<MotionModule>(it->second);
        }
    };

}  // namespace AnimateDiff

#endif  // __SD_MODEL_DIFFUSION_ANIMATEDIFF_HPP__
