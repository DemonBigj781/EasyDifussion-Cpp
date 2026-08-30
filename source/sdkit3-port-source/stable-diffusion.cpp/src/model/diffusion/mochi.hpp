#ifndef __SD_MODEL_DIFFUSION_MOCHI_HPP__
#define __SD_MODEL_DIFFUSION_MOCHI_HPP__

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "core/ggml_graph_cut.h"
#include "model/common/block.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

// Native Mochi 1 Preview inference.  Tensor names and math intentionally follow
// genmo/mochi so the original and ComfyUI safetensors checkpoints load directly.
namespace MOCHI {

constexpr int MOCHI_GRAPH_SIZE = 65536;

static ggml_tensor* modulated_rms_norm(ggml_context* ctx,
                                       ggml_tensor* x,
                                       ggml_tensor* scale) {
    x          = ggml_rms_norm(ctx, x, 1e-6f);
    auto one   = ggml_ext_ones(ctx, 1, 1, 1, 1);
    auto gain  = ggml_add(ctx, scale, one);
    return ggml_mul(ctx, x, gain);
}

static ggml_tensor* gated_residual(ggml_context* ctx,
                                   ggml_tensor* x,
                                   ggml_tensor* residual,
                                   ggml_tensor* gate) {
    residual = ggml_rms_norm(ctx, residual, 1e-6f);
    gate     = ggml_tanh(ctx, gate);
    return ggml_add(ctx, x, ggml_mul(ctx, residual, gate));
}

class MochiFeedForward : public GGMLBlock {
public:
    MochiFeedForward(int64_t dim, int64_t hidden_dim) {
        blocks["w1"] = std::make_shared<Linear>(dim, hidden_dim * 2, false);
        blocks["w2"] = std::make_shared<Linear>(hidden_dim, dim, false);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
        auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
        auto uv = ggml_ext_chunk(ctx->ggml_ctx, w1->forward(ctx, x), 2, 0);
        return w2->forward(ctx, ggml_mul(ctx->ggml_ctx,
                                        ggml_silu(ctx->ggml_ctx, uv[0]),
                                        uv[1]));
    }
};

class MochiAttention : public GGMLBlock {
    int64_t num_heads = 24;
    int64_t head_dim  = 128;
    bool update_y     = true;

    ggml_tensor* norm_heads(GGMLRunnerContext* ctx,
                            ggml_tensor* x,
                            const std::string& name,
                            int64_t tokens) {
        auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks[name]);
        x         = ggml_reshape_4d(ctx->ggml_ctx, x, head_dim, num_heads, tokens, x->ne[2]);
        return norm->forward(ctx, x);
    }

    ggml_tensor* apply_mixed_rope(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* rope_cos,
                                  ggml_tensor* rope_sin) {
        const int64_t tokens = x->ne[2];
        const int64_t batch  = x->ne[3];
        x                    = ggml_reshape_4d(ctx->ggml_ctx, x, 2, head_dim / 2, num_heads, tokens * batch);
        auto even            = ggml_ext_slice(ctx->ggml_ctx, x, 0, 0, 1);
        auto odd             = ggml_ext_slice(ctx->ggml_ctx, x, 0, 1, 2);
        rope_cos             = ggml_reshape_4d(ctx->ggml_ctx, rope_cos, 1, head_dim / 2, num_heads, tokens);
        rope_sin             = ggml_reshape_4d(ctx->ggml_ctx, rope_sin, 1, head_dim / 2, num_heads, tokens);
        auto out_even        = ggml_sub(ctx->ggml_ctx,
                                ggml_mul(ctx->ggml_ctx, even, rope_cos),
                                ggml_mul(ctx->ggml_ctx, odd, rope_sin));
        auto out_odd         = ggml_add(ctx->ggml_ctx,
                               ggml_mul(ctx->ggml_ctx, even, rope_sin),
                               ggml_mul(ctx->ggml_ctx, odd, rope_cos));
        x                    = ggml_concat(ctx->ggml_ctx, out_even, out_odd, 0);
        return ggml_reshape_4d(ctx->ggml_ctx, x, head_dim, num_heads, tokens, batch);
    }

public:
    explicit MochiAttention(bool update_y)
        : update_y(update_y) {
        blocks["qkv_x"]    = std::make_shared<Linear>(3072, 3 * 3072, false);
        blocks["qkv_y"]    = std::make_shared<Linear>(1536, 3 * 3072, false);
        blocks["q_norm_x"] = std::make_shared<RMSNorm>(head_dim, 1e-5f);
        blocks["k_norm_x"] = std::make_shared<RMSNorm>(head_dim, 1e-5f);
        blocks["q_norm_y"] = std::make_shared<RMSNorm>(head_dim, 1e-5f);
        blocks["k_norm_y"] = std::make_shared<RMSNorm>(head_dim, 1e-5f);
        blocks["proj_x"]   = std::make_shared<Linear>(3072, 3072);
        if (update_y) {
            blocks["proj_y"] = std::make_shared<Linear>(3072, 1536);
        }
    }

    std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* y,
                                                   ggml_tensor* scale_x,
                                                   ggml_tensor* scale_y,
                                                   ggml_tensor* rope_cos,
                                                   ggml_tensor* rope_sin) {
        const int64_t nx = x->ne[1];
        const int64_t ny = y->ne[1];
        auto qkv_x       = std::dynamic_pointer_cast<Linear>(blocks["qkv_x"]);
        auto qkv_y       = std::dynamic_pointer_cast<Linear>(blocks["qkv_y"]);

        auto qkvx = ggml_ext_chunk(ctx->ggml_ctx,
                                   qkv_x->forward(ctx, modulated_rms_norm(ctx->ggml_ctx, x, scale_x)),
                                   3,
                                   0);
        auto qx = norm_heads(ctx, qkvx[0], "q_norm_x", nx);
        auto kx = norm_heads(ctx, qkvx[1], "k_norm_x", nx);
        qx      = apply_mixed_rope(ctx, qx, rope_cos, rope_sin);
        kx      = apply_mixed_rope(ctx, kx, rope_cos, rope_sin);
        qx      = ggml_reshape_3d(ctx->ggml_ctx, qx, 3072, nx, x->ne[2]);
        kx      = ggml_reshape_3d(ctx->ggml_ctx, kx, 3072, nx, x->ne[2]);

        auto qkvy = ggml_ext_chunk(ctx->ggml_ctx,
                                   qkv_y->forward(ctx, modulated_rms_norm(ctx->ggml_ctx, y, scale_y)),
                                   3,
                                   0);
        auto qy = norm_heads(ctx, qkvy[0], "q_norm_y", ny);
        auto ky = norm_heads(ctx, qkvy[1], "k_norm_y", ny);
        qy      = ggml_reshape_3d(ctx->ggml_ctx, qy, 3072, ny, y->ne[2]);
        ky      = ggml_reshape_3d(ctx->ggml_ctx, ky, 3072, ny, y->ne[2]);

        auto q = ggml_concat(ctx->ggml_ctx, qx, qy, 1);
        auto k = ggml_concat(ctx->ggml_ctx, kx, ky, 1);
        auto v = ggml_concat(ctx->ggml_ctx, qkvx[2], qkvy[2], 1);
        auto a = ggml_ext_attention_ext(ctx->ggml_ctx,
                                        ctx->backend,
                                        q,
                                        k,
                                        v,
                                        num_heads,
                                        nullptr,
                                        false,
                                        ctx->flash_attn_enabled);

        auto ax = ggml_ext_slice(ctx->ggml_ctx, a, 1, 0, nx);
        auto ay = ggml_ext_slice(ctx->ggml_ctx, a, 1, nx, nx + ny);
        ax      = std::dynamic_pointer_cast<Linear>(blocks["proj_x"])->forward(ctx, ax);
        if (update_y) {
            ay = std::dynamic_pointer_cast<Linear>(blocks["proj_y"])->forward(ctx, ay);
        }
        return {ax, ay};
    }
};

class MochiJointBlock : public GGMLBlock {
    bool update_y = true;

public:
    explicit MochiJointBlock(bool update_y)
        : update_y(update_y) {
        blocks["mod_x"] = std::make_shared<Linear>(3072, 4 * 3072);
        blocks["mod_y"] = std::make_shared<Linear>(3072, update_y ? 4 * 1536 : 1536);
        blocks["attn"]  = std::make_shared<MochiAttention>(update_y);
        blocks["mlp_x"] = std::make_shared<MochiFeedForward>(3072, 8192);
        if (update_y) {
            blocks["mlp_y"] = std::make_shared<MochiFeedForward>(1536, 4096);
        }
    }

    std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* c,
                                                   ggml_tensor* y,
                                                   ggml_tensor* rope_cos,
                                                   ggml_tensor* rope_sin) {
        auto c_act = ggml_silu(ctx->ggml_ctx, c);
        auto mx    = ggml_ext_chunk(ctx->ggml_ctx,
                                std::dynamic_pointer_cast<Linear>(blocks["mod_x"])->forward(ctx, c_act),
                                4,
                                0);
        auto my_raw = std::dynamic_pointer_cast<Linear>(blocks["mod_y"])->forward(ctx, c_act);
        std::vector<ggml_tensor*> my;
        if (update_y) {
            my = ggml_ext_chunk(ctx->ggml_ctx, my_raw, 4, 0);
        } else {
            my.push_back(my_raw);
        }

        auto attn = std::dynamic_pointer_cast<MochiAttention>(blocks["attn"])
                        ->forward(ctx, x, y, mx[0], my[0], rope_cos, rope_sin);
        x = gated_residual(ctx->ggml_ctx, x, attn.first, mx[1]);
        if (update_y) {
            y = gated_residual(ctx->ggml_ctx, y, attn.second, my[1]);
        }

        auto x_ff = std::dynamic_pointer_cast<MochiFeedForward>(blocks["mlp_x"])
                        ->forward(ctx, modulated_rms_norm(ctx->ggml_ctx, x, mx[2]));
        x = gated_residual(ctx->ggml_ctx, x, x_ff, mx[3]);
        if (update_y) {
            auto y_ff = std::dynamic_pointer_cast<MochiFeedForward>(blocks["mlp_y"])
                            ->forward(ctx, modulated_rms_norm(ctx->ggml_ctx, y, my[2]));
            y = gated_residual(ctx->ggml_ctx, y, y_ff, my[3]);
        }
        return {x, y};
    }
};

class MochiAttentionPool : public GGMLBlock {
public:
    MochiAttentionPool() {
        blocks["to_kv"]  = std::make_shared<Linear>(4096, 8192);
        blocks["to_q"]   = std::make_shared<Linear>(4096, 4096);
        blocks["to_out"] = std::make_shared<Linear>(4096, 3072);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* text) {
        // text: [4096, valid_tokens, batch].  Mean over the token dimension.
        auto token_major = ggml_ext_cont(ctx->ggml_ctx,
                                         ggml_permute(ctx->ggml_ctx, text, 1, 0, 2, 3));
        auto pooled      = ggml_mean(ctx->ggml_ctx, token_major);
        pooled           = ggml_ext_cont(ctx->ggml_ctx,
                               ggml_permute(ctx->ggml_ctx, pooled, 1, 0, 2, 3));
        pooled           = ggml_reshape_3d(ctx->ggml_ctx, pooled, 4096, 1, text->ne[2]);
        auto all_tokens  = ggml_concat(ctx->ggml_ctx, pooled, text, 1);
        auto kv          = ggml_ext_chunk(ctx->ggml_ctx,
                                std::dynamic_pointer_cast<Linear>(blocks["to_kv"])->forward(ctx, all_tokens),
                                2,
                                0);
        auto q           = std::dynamic_pointer_cast<Linear>(blocks["to_q"])->forward(ctx, pooled);
        auto out         = ggml_ext_attention_ext(ctx->ggml_ctx,
                                         ctx->backend,
                                         q,
                                         kv[0],
                                         kv[1],
                                         8,
                                         nullptr,
                                         false,
                                         ctx->flash_attn_enabled);
        out = std::dynamic_pointer_cast<Linear>(blocks["to_out"])->forward(ctx, out);
        return ggml_reshape_2d(ctx->ggml_ctx, out, 3072, text->ne[2]);
    }
};

class MochiFinalLayer : public GGMLBlock {
public:
    MochiFinalLayer() {
        blocks["mod"]    = std::make_shared<Linear>(3072, 6144);
        blocks["linear"] = std::make_shared<Linear>(3072, 48);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* c) {
        auto mods = ggml_ext_chunk(ctx->ggml_ctx,
                                   std::dynamic_pointer_cast<Linear>(blocks["mod"])
                                       ->forward(ctx, ggml_silu(ctx->ggml_ctx, c)),
                                   2,
                                   0);
        x = ggml_norm(ctx->ggml_ctx, x, 1e-6f);
        x = ggml_add(ctx->ggml_ctx,
                     ggml_mul(ctx->ggml_ctx,
                              x,
                              ggml_add(ctx->ggml_ctx,
                                       mods[1],
                                       ggml_ext_ones(ctx->ggml_ctx, 1, 1, 1, 1))),
                     mods[0]);
        return std::dynamic_pointer_cast<Linear>(blocks["linear"])->forward(ctx, x);
    }
};

class MochiModel : public GGMLBlock {
    void init_params(ggml_context* ctx,
                     const String2TensorStorage& tensor_storage_map,
                     const std::string prefix) override {
        params["pos_frequencies"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 24, 3);
    }

public:
    MochiModel() {
        blocks["x_embedder.proj"] = std::make_shared<Conv2d>(12, 3072, std::make_pair(2, 2), std::make_pair(2, 2));
        blocks["t_embedder.mlp.0"] = std::make_shared<Linear>(256, 3072);
        blocks["t_embedder.mlp.2"] = std::make_shared<Linear>(3072, 3072);
        blocks["t5_y_embedder"]    = std::make_shared<MochiAttentionPool>();
        blocks["t5_yproj"]         = std::make_shared<Linear>(4096, 1536);
        for (int i = 0; i < 48; ++i) {
            blocks["blocks." + std::to_string(i)] = std::make_shared<MochiJointBlock>(i < 47);
        }
        blocks["final_layer"] = std::make_shared<MochiFinalLayer>();
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* latent,
                         ggml_tensor* timestep,
                         ggml_tensor* text,
                         ggml_tensor* positions) {
        const int64_t w = latent->ne[0] / 2;
        const int64_t h = latent->ne[1] / 2;
        const int64_t t = latent->ne[2];
        const int64_t b = latent->ne[3] / 12;
        GGML_ASSERT(b == 1);

        // Conv2d sees video frames as its batch dimension.
        auto x = ggml_ext_cont(ctx->ggml_ctx,
                               ggml_ext_torch_permute(ctx->ggml_ctx, latent, 0, 1, 3, 2));
        x = std::dynamic_pointer_cast<Conv2d>(blocks["x_embedder.proj"])->forward(ctx, x);
        x = ggml_reshape_3d(ctx->ggml_ctx, x, w * h, 3072, t * b);
        x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        x = ggml_reshape_3d(ctx->ggml_ctx, x, 3072, t * h * w, b);

        // Official Mochi embeds (1 - sigma) * 1000. Core flow timesteps are sigma*1000.
        auto thousand = ggml_ext_scale(ctx->ggml_ctx,
                                       ggml_ext_ones(ctx->ggml_ctx, 1, 1, 1, 1),
                                       1000.f);
        auto one_minus_sigma = ggml_add(ctx->ggml_ctx,
                                        ggml_ext_scale(ctx->ggml_ctx, timestep, -1.f),
                                        thousand);
        auto c = ggml_ext_timestep_embedding(ctx->ggml_ctx, one_minus_sigma, 256);
        c      = std::dynamic_pointer_cast<Linear>(blocks["t_embedder.mlp.0"])->forward(ctx, c);
        c      = ggml_silu(ctx->ggml_ctx, c);
        c      = std::dynamic_pointer_cast<Linear>(blocks["t_embedder.mlp.2"])->forward(ctx, c);
        c      = ggml_add(ctx->ggml_ctx,
                     c,
                     std::dynamic_pointer_cast<MochiAttentionPool>(blocks["t5_y_embedder"])->forward(ctx, text));
        auto y = std::dynamic_pointer_cast<Linear>(blocks["t5_yproj"])->forward(ctx, text);

        auto freq = ggml_ext_cont(ctx->ggml_ctx,
                                  ggml_permute(ctx->ggml_ctx, params["pos_frequencies"], 2, 0, 1, 3));
        freq      = ggml_reshape_2d(ctx->ggml_ctx, freq, 3, 64 * 24);
        auto angle = ggml_mul_mat(ctx->ggml_ctx, freq, positions);
        angle      = ggml_reshape_3d(ctx->ggml_ctx, angle, 64, 24, t * h * w);
        auto rope_cos = ggml_cos(ctx->ggml_ctx, angle);
        auto rope_sin = ggml_sin(ctx->ggml_ctx, angle);

        sd::ggml_graph_cut::mark_graph_cut(x, "mochi.prelude", "x");
        sd::ggml_graph_cut::mark_graph_cut(y, "mochi.prelude", "y");
        for (int i = 0; i < 48; ++i) {
            auto block = std::dynamic_pointer_cast<MochiJointBlock>(blocks["blocks." + std::to_string(i)]);
            auto xy    = block->forward(ctx, x, c, y, rope_cos, rope_sin);
            x          = xy.first;
            y          = xy.second;
            sd::ggml_graph_cut::mark_graph_cut(x, "mochi.blocks." + std::to_string(i), "x");
            if (i < 47) {
                sd::ggml_graph_cut::mark_graph_cut(y, "mochi.blocks." + std::to_string(i), "y");
            }
        }
        x = std::dynamic_pointer_cast<MochiFinalLayer>(blocks["final_layer"])->forward(ctx, x, c);
        x = DiT::unpatchify(ctx->ggml_ctx, x, t, h, w, 1, 2, 2);

        // genmo/mochi predicts z0-epsilon; the core FLOW denoiser consumes epsilon-z0.
        return ggml_neg(ctx->ggml_ctx, x);
    }
};

struct MochiRunner : public DiffusionModelRunner {
    MochiModel model;
    std::vector<float> position_data;

    MochiRunner(ggml_backend_t backend,
                const String2TensorStorage& tensor_storage_map      = {},
                const std::string& prefix                           = "",
                std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : DiffusionModelRunner(backend, prefix, weight_manager) {
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "Mochi 1 Preview";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                           const std::string& prefix) override {
        model.get_param_tensors(tensors, prefix);
    }

    static sd::Tensor<float> trim_text(const sd::Tensor<float>& text,
                                       const sd::Tensor<float>& mask) {
        if (text.empty() || mask.empty() || text.dim() < 2) {
            return text;
        }
        int64_t valid = 0;
        while (valid < mask.numel() && std::isfinite(mask[valid])) {
            ++valid;
        }
        valid = std::max<int64_t>(1, std::min<int64_t>(valid, text.shape()[1]));
        return valid == text.shape()[1] ? text : sd::ops::slice(text, 1, 0, valid);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& latent,
                             const sd::Tensor<float>& timestep,
                             const sd::Tensor<float>& text) {
        ggml_cgraph* graph = new_graph_custom(MOCHI_GRAPH_SIZE);
        auto x             = make_input(latent);
        auto ts            = make_input(timestep);
        auto context       = make_input(text);

        const int64_t pw = latent.shape()[0] / 2;
        const int64_t ph = latent.shape()[1] / 2;
        const int64_t pt = latent.shape()[2];
        const float scale = std::sqrt(36864.f / static_cast<float>(pw * ph));
        position_data.clear();
        position_data.reserve(static_cast<size_t>(3 * pw * ph * pt));
        for (int64_t ti = 0; ti < pt; ++ti) {
            for (int64_t hi = 0; hi < ph; ++hi) {
                const float hp = (static_cast<float>(hi) + 0.5f - static_cast<float>(ph) * 0.5f) * scale;
                for (int64_t wi = 0; wi < pw; ++wi) {
                    const float wp = (static_cast<float>(wi) + 0.5f - static_cast<float>(pw) * 0.5f) * scale;
                    position_data.push_back(static_cast<float>(ti));
                    position_data.push_back(hp);
                    position_data.push_back(wp);
                }
            }
        }
        auto positions = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 3, pw * ph * pt);
        set_backend_tensor_data(positions, position_data.data());

        auto runner_ctx = get_context();
        auto out        = model.forward(&runner_ctx, x, ts, context, positions);
        ggml_build_forward_expand(graph, out);
        return graph;
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& params) override {
        GGML_ASSERT(params.x != nullptr);
        GGML_ASSERT(params.timesteps != nullptr);
        GGML_ASSERT(params.context != nullptr);
        const auto text = trim_text(*params.context, tensor_or_empty(params.y));
        auto get_graph  = [&]() { return build_graph(*params.x, *params.timesteps, text); };
        return restore_trailing_singleton_dims(
            GGMLRunner::compute<float>(get_graph, n_threads, false, false, false),
            params.x->dim());
    }
};

}  // namespace MOCHI

#endif  // __SD_MODEL_DIFFUSION_MOCHI_HPP__
