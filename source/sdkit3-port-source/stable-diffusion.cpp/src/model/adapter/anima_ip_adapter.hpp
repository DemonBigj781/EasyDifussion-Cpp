#ifndef SD_MODEL_ADAPTER_ANIMA_IP_ADAPTER_HPP
#define SD_MODEL_ADAPTER_ANIMA_IP_ADAPTER_HPP

#include "core/ggml_extend.hpp"

namespace AnimaIP {

constexpr int num_blocks = 28;
constexpr int hidden_size = 2048;
constexpr int feature_size = 768;
constexpr int rank = 32;

inline void validate_checkpoint(const String2TensorStorage& storage,
                                const std::map<std::string, std::string>& metadata,
                                const std::string& prefix = "anima_ip.") {
    const std::map<std::string, std::string> required = {
        {"ip_inject_before_mlp", "False"}, {"ip_norm_keys", "False"},
        {"lora_alpha", "32"}, {"lora_rank", "32"}};
    for (const auto& entry : required) {
        auto found = metadata.find(entry.first);
        if (found == metadata.end() || found->second != entry.second) {
            throw std::invalid_argument("Unsupported Anima IP metadata: " + entry.first);
        }
    }
    std::set<std::string> expected;
    auto require = [&](const std::string& name, std::initializer_list<int64_t> shape) {
        const auto full_name = prefix + name;
        auto found = storage.find(full_name);
        if (found == storage.end()) throw std::invalid_argument("Missing Anima IP tensor: " + full_name);
        int64_t dimensions[4] = {1, 1, 1, 1};
        std::copy(shape.begin(), shape.end(), dimensions);
        for (int axis = 0; axis < 4; ++axis) {
            if (found->second.ne[axis] != dimensions[axis]) {
                throw std::invalid_argument("Unsupported Anima IP shape: " + full_name);
            }
        }
        expected.insert(full_name);
    };
    for (int i = 0; i < num_blocks; ++i) {
        auto block = "blocks." + std::to_string(i) + ".";
        for (const auto& name : {"ip_k_proj", "ip_v_proj", "adaln_ip.1"}) {
            require(block + name + ".weight", {std::string(name) == "adaln_ip.1" ? hidden_size : feature_size, hidden_size});
            require(block + name + ".bias", {hidden_size});
        }
        auto lora = "lora.base_model.model." + block;
        for (const auto& attention : {"cross_attn", "self_attn"}) {
            for (const auto& projection : {"q_proj", "k_proj", "v_proj", "output_proj"}) {
                const bool context = std::string(attention) == "cross_attn" &&
                                     (std::string(projection) == "k_proj" || std::string(projection) == "v_proj");
                const auto name = lora + attention + "." + projection;
                require(name + ".lora_A.weight", {context ? 1024 : hidden_size, rank});
                require(name + ".lora_B.weight", {rank, hidden_size});
            }
        }
        require(lora + "mlp.layer1.lora_A.weight", {hidden_size, rank});
        require(lora + "mlp.layer1.lora_B.weight", {rank, hidden_size * 4});
        require(lora + "mlp.layer2.lora_A.weight", {hidden_size * 4, rank});
        require(lora + "mlp.layer2.lora_B.weight", {rank, hidden_size});
    }
    for (const auto& entry : storage) {
        if (starts_with(entry.first, prefix) && expected.count(entry.first) == 0) {
            throw std::invalid_argument("Unsupported Anima IP tensor: " + entry.first);
        }
    }
}

struct Block : GGMLBlock {
    Block() {
        blocks["ip_k_proj"] = std::make_shared<Linear>(feature_size, hidden_size, true, true, true);
        blocks["ip_v_proj"] = std::make_shared<Linear>(feature_size, hidden_size, true, true, true);
        blocks["adaln_ip.1"] = std::make_shared<Linear>(hidden_size, hidden_size, true, true, true);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* normalized_query,
                         ggml_tensor* tokens, ggml_tensor* timestep, float strength) {
        auto g = ctx->ggml_ctx;
        // No projection of the attention output: this is the author's residual.
        auto scaled = ggml_scale(g, tokens, strength);
        auto k = std::static_pointer_cast<Linear>(blocks.at("ip_k_proj"))->forward(ctx, scaled);
        auto v = std::static_pointer_cast<Linear>(blocks.at("ip_v_proj"))->forward(ctx, scaled);
        auto out = ggml_ext_attention_ext(g, ctx->backend, normalized_query, k, v, 16, nullptr, false,
                                          ctx->flash_attn_enabled, 1.f, false, GGML_PREC_F32);
        auto gate = std::static_pointer_cast<Linear>(blocks.at("adaln_ip.1"))->forward(ctx, ggml_silu(g, timestep));
        return ggml_mul(g, out, gate);
    }
};

struct Adapter : GGMLBlock {
    Adapter() {
        for (int i = 0; i < num_blocks; ++i) {
            const auto index = std::to_string(i);
            blocks["blocks." + index] = std::make_shared<Block>();
            for (const auto& projection : {"q_proj", "k_proj", "v_proj", "output_proj"}) {
                const auto name = "lora.base_model.model.blocks." + index + ".cross_attn." + projection;
                const bool context = std::string(projection) == "k_proj" || std::string(projection) == "v_proj";
                blocks[name + ".lora_A"] = std::make_shared<Linear>(context ? 1024 : hidden_size, rank, false, true, true);
                blocks[name + ".lora_B"] = std::make_shared<Linear>(rank, hidden_size, false, true, true);
            }
        }
    }

    ggml_tensor* lora(GGMLRunnerContext* ctx, int index, const std::string& projection, ggml_tensor* x) {
        const auto name = "lora.base_model.model.blocks." + std::to_string(index) + ".cross_attn." + projection;
        auto a = std::static_pointer_cast<Linear>(blocks.at(name + ".lora_A"));
        auto b = std::static_pointer_cast<Linear>(blocks.at(name + ".lora_B"));
        return b->forward(ctx, a->forward(ctx, x));
    }

    ggml_tensor* image_attention(GGMLRunnerContext* ctx, int index, ggml_tensor* normalized_query,
                                 ggml_tensor* tokens, ggml_tensor* timestep, float strength) {
        if (strength == 0 || tokens == nullptr) {
            return ggml_scale(ctx->ggml_ctx, normalized_query, 0);
        }
        return std::static_pointer_cast<Block>(blocks.at("blocks." + std::to_string(index)))->forward(
            ctx, normalized_query, tokens, timestep, strength);
    }
};

}  // namespace AnimaIP
#endif
