#ifndef SD_ANIMA_LLLITE_HPP
#define SD_ANIMA_LLLITE_HPP

#include "model/common/block.hpp"
#include <set>
#include <stdexcept>

namespace AnimaLLLite {

class Norm : public GGMLBlock {
    int64_t channels;
    void init_params(ggml_context* ctx, const String2TensorStorage&, std::string) override {
        params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    }
public:
    explicit Norm(int64_t channels) : channels(channels) {}
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        int groups = 8;
        while (channels % groups) groups /= 2;
        x = ggml_group_norm(ctx->ggml_ctx, x, groups, 1e-5f);
        auto w = ggml_reshape_4d(ctx->ggml_ctx, params["weight"], 1, 1, channels, 1);
        auto b = ggml_reshape_4d(ctx->ggml_ctx, params["bias"], 1, 1, channels, 1);
        return ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, w), b);
    }
};

class Residual : public GGMLBlock {
public:
    explicit Residual(int64_t channels) {
        blocks["norm1"] = std::make_shared<Norm>(channels);
        blocks["norm2"] = std::make_shared<Norm>(channels);
        blocks["conv1"] = std::make_shared<Conv2d>(channels, channels, std::pair<int,int>{3,3}, std::pair<int,int>{1,1}, std::pair<int,int>{1,1});
        blocks["conv2"] = std::make_shared<Conv2d>(channels, channels, std::pair<int,int>{3,3}, std::pair<int,int>{1,1}, std::pair<int,int>{1,1});
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto h = x;
        for (const auto& index : {"1", "2"}) {
            h = std::dynamic_pointer_cast<Norm>(blocks[std::string("norm") + index])->forward(ctx, h);
            h = ggml_silu(ctx->ggml_ctx, h);
            h = std::dynamic_pointer_cast<Conv2d>(blocks[std::string("conv") + index])->forward(ctx, h);
        }
        return ggml_add(ctx->ggml_ctx, x, h);
    }
};

class Conditioning : public GGMLBlock {
    int residuals = 0;
public:
    int channels;
    explicit Conditioning(const String2TensorStorage& storage) {
        const std::string prefix = "lllite.lllite_conditioning1.";
        const auto& conv1 = storage.at(prefix + "conv1.weight");
        const auto& conv3 = storage.at(prefix + "conv3.weight");
        channels = int(conv1.ne[2]);
        if (channels != 3) throw std::invalid_argument("Anima LLLite inpainting requires a separate mask input; RGB conditioning weights are required here");
        const auto half = conv1.ne[3], dim = conv3.ne[3];
        const auto embedding = storage.at(prefix + "proj.weight").ne[3];
        for (const auto& entry : storage) {
            if (entry.first.rfind(prefix + "aspp.", 0) == 0)
                throw std::invalid_argument("Anima LLLite ASPP weights are not supported");
        }
        blocks["conv1"] = std::make_shared<Conv2d>(channels, half, std::pair<int,int>{4,4}, std::pair<int,int>{4,4});
        blocks["conv2"] = std::make_shared<Conv2d>(half, half, std::pair<int,int>{3,3}, std::pair<int,int>{1,1}, std::pair<int,int>{1,1});
        blocks["conv3"] = std::make_shared<Conv2d>(half, dim, std::pair<int,int>{4,4}, std::pair<int,int>{4,4});
        blocks["norm1"] = std::make_shared<Norm>(half);
        blocks["norm2"] = std::make_shared<Norm>(half);
        blocks["norm3"] = std::make_shared<Norm>(dim);
        while (storage.count(prefix + "resblocks." + std::to_string(residuals) + ".conv1.weight")) {
            blocks["resblocks." + std::to_string(residuals++)] = std::make_shared<Residual>(dim);
        }
        blocks["proj"] = std::make_shared<Conv2d>(dim, embedding, std::pair<int,int>{1,1});
        blocks["out_norm"] = std::make_shared<LayerNorm>(embedding);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        for (const auto& index : {"1", "2", "3"}) {
            x = std::dynamic_pointer_cast<Conv2d>(blocks[std::string("conv") + index])->forward(ctx, x);
            x = std::dynamic_pointer_cast<Norm>(blocks[std::string("norm") + index])->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
        }
        for (int i = 0; i < residuals; ++i)
            x = std::dynamic_pointer_cast<Residual>(blocks["resblocks." + std::to_string(i)])->forward(ctx, x);
        x = std::dynamic_pointer_cast<Conv2d>(blocks["proj"])->forward(ctx, x);
        const auto width = x->ne[0], height = x->ne[1], channels = x->ne[2], batch = x->ne[3];
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));
        x = ggml_reshape_3d(ctx->ggml_ctx, x, channels, width * height, batch);
        return std::dynamic_pointer_cast<LayerNorm>(blocks["out_norm"])->forward(ctx, x);
    }
};

class Module : public GGMLBlock {
    int64_t embedding, mlp;
    void init_params(ggml_context* ctx, const String2TensorStorage&, std::string) override {
        params["depth_embed"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embedding);
    }
public:
    Module(const String2TensorStorage& storage, const std::string& prefix) {
        const auto& down = storage.at(prefix + ".down.weight");
        embedding = storage.at(prefix + ".cond_to_film.weight").ne[0];
        mlp = down.ne[1];
        if (down.ne[0] != 2048) throw std::invalid_argument("Anima LLLite requires 2048-wide DiT projections");
        blocks["down"] = std::make_shared<Linear>(down.ne[0], mlp);
        blocks["mid"] = std::make_shared<Linear>(embedding + mlp, mlp);
        blocks["cond_to_film"] = std::make_shared<Linear>(embedding, 2 * mlp);
        blocks["up"] = std::make_shared<Linear>(mlp, down.ne[0]);
    }
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* condition, float strength) {
        if (x->ne[1] != condition->ne[1]) throw std::invalid_argument("Anima LLLite control image token count does not match the image; reference-image concatenation is not supported");
        auto local = ggml_add(ctx->ggml_ctx, condition, params["depth_embed"]);
        auto h = ggml_silu(ctx->ggml_ctx, std::dynamic_pointer_cast<Linear>(blocks["down"])->forward(ctx, x));
        auto film = std::dynamic_pointer_cast<Linear>(blocks["cond_to_film"])->forward(ctx, local);
        auto gamma = ggml_ext_slice(ctx->ggml_ctx, film, 0, 0, mlp);
        auto beta = ggml_ext_slice(ctx->ggml_ctx, film, 0, mlp, 2 * mlp);
        h = std::dynamic_pointer_cast<Linear>(blocks["mid"])->forward(ctx, ggml_concat(ctx->ggml_ctx, local, h, 0));
        h = ggml_add(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, h, ggml_mul(ctx->ggml_ctx, h, gamma)), beta);
        h = std::dynamic_pointer_cast<Linear>(blocks["up"])->forward(ctx, ggml_silu(ctx->ggml_ctx, h));
        return ggml_add(ctx->ggml_ctx, x, ggml_ext_scale(ctx->ggml_ctx, h, strength));
    }
};

class Adapter {
    std::shared_ptr<Conditioning> encoder;
    std::map<std::string, std::shared_ptr<Module>> modules;
public:
    void init(ggml_context* ctx, const String2TensorStorage& storage) {
        encoder = std::make_shared<Conditioning>(storage);
        encoder->init(ctx, storage, "lllite.lllite_conditioning1");
        const std::string prefix = "lllite.lllite_dit_blocks_";
        std::set<std::string> names;
        for (const auto& item : storage) {
            if (item.first.rfind(prefix, 0) == 0) names.insert(item.first.substr(0, item.first.find('.' , prefix.size())));
        }
        if (names.empty()) throw std::invalid_argument("Anima LLLite has no DiT modules");
        for (const auto& name : names) {
            auto module = std::make_shared<Module>(storage, name);
            module->init(ctx, storage, name);
            modules[name] = module;
        }
    }
    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        encoder->get_param_tensors(tensors, "lllite.lllite_conditioning1");
        for (const auto& item : modules) item.second->get_param_tensors(tensors, item.first);
    }
    ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* image) { return encoder->forward(ctx, image); }
    ggml_tensor* apply(GGMLRunnerContext* ctx, std::string name, ggml_tensor* x, ggml_tensor* condition, float strength) {
        const std::string prefix = "model.diffusion_model.net.";
        if (name.rfind(prefix, 0) != 0) return x;
        name.erase(0, prefix.size());
        std::replace(name.begin(), name.end(), '.', '_');
        auto found = modules.find("lllite.lllite_dit_" + name);
        return found == modules.end() ? x : found->second->forward(ctx, x, condition, strength);
    }
};
}
#endif
