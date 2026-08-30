#ifndef __SD_MODEL_ADAPTER_LLLITE_HPP__
#define __SD_MODEL_ADAPTER_LLLITE_HPP__

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>

#include "model/common/block.hpp"

// Native inference implementation for kohya-ss ControlNet-LLLite modules.
// LLLite injects a small, image-conditioned residual into selected UNet
// attention Q/K/V projections. Weight names follow the original ComfyUI
// extension's `lllite_unet_...` convention.
class LLLiteModule : public GGMLBlock {
protected:
    std::string name;
    bool is_conv2d = false;
    int depth      = 1;

public:
    LLLiteModule(const std::string& name,
                 const String2TensorStorage& tensor_storage_map)
        : name(name) {
        const std::string prefix = "lllite." + name + ".";
        const auto down_it       = tensor_storage_map.find(prefix + "down.0.weight");
        const auto cond0_it      = tensor_storage_map.find(prefix + "conditioning1.0.weight");
        const auto cond2_it      = tensor_storage_map.find(prefix + "conditioning1.2.weight");
        GGML_ASSERT(down_it != tensor_storage_map.end());
        GGML_ASSERT(cond0_it != tensor_storage_map.end());
        GGML_ASSERT(cond2_it != tensor_storage_map.end());

        const TensorStorage& down  = down_it->second;
        const TensorStorage& cond0 = cond0_it->second;
        const TensorStorage& cond2 = cond2_it->second;
        is_conv2d                   = down.n_dims == 4;

        if (tensor_storage_map.find(prefix + "conditioning1.4.weight") != tensor_storage_map.end()) {
            depth = 3;
        } else if (cond2.n_dims == 4 && cond2.ne[0] == 4) {
            depth = 2;
        }

        const int64_t cond_half = cond0.ne[3];
        const int64_t cond_dim  = cond_half * 2;
        const int64_t in_dim    = is_conv2d ? down.ne[2] : down.ne[0];
        const int64_t mlp_dim   = is_conv2d ? down.ne[3] : down.ne[1];

        blocks["conditioning1.0"] = std::make_shared<Conv2d>(3, cond_half, std::pair<int, int>{4, 4}, std::pair<int, int>{4, 4});
        if (depth == 1) {
            blocks["conditioning1.2"] = std::make_shared<Conv2d>(cond_half, cond_dim, std::pair<int, int>{2, 2}, std::pair<int, int>{2, 2});
        } else if (depth == 2) {
            blocks["conditioning1.2"] = std::make_shared<Conv2d>(cond_half, cond_dim, std::pair<int, int>{4, 4}, std::pair<int, int>{4, 4});
        } else {
            blocks["conditioning1.2"] = std::make_shared<Conv2d>(cond_half, cond_half, std::pair<int, int>{4, 4}, std::pair<int, int>{4, 4});
            blocks["conditioning1.4"] = std::make_shared<Conv2d>(cond_half, cond_dim, std::pair<int, int>{2, 2}, std::pair<int, int>{2, 2});
        }

        if (is_conv2d) {
            blocks["down.0"] = std::make_shared<Conv2d>(in_dim, mlp_dim, std::pair<int, int>{1, 1});
            blocks["mid.0"]  = std::make_shared<Conv2d>(mlp_dim + cond_dim, mlp_dim, std::pair<int, int>{1, 1});
            blocks["up.0"]   = std::make_shared<Conv2d>(mlp_dim, in_dim, std::pair<int, int>{1, 1});
        } else {
            blocks["down.0"] = std::make_shared<Linear>(in_dim, mlp_dim);
            blocks["mid.0"]  = std::make_shared<Linear>(mlp_dim + cond_dim, mlp_dim);
            blocks["up.0"]   = std::make_shared<Linear>(mlp_dim, in_dim);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* condition,
                         float strength) {
        const std::string cache_name = "lllite.condition." + name;
        ggml_tensor* cx              = ctx->load_cache_tensor(cache_name);
        if (cx == nullptr) {
            auto conditioning_0 = std::dynamic_pointer_cast<Conv2d>(blocks["conditioning1.0"]);
            auto conditioning_2 = std::dynamic_pointer_cast<Conv2d>(blocks["conditioning1.2"]);
            cx                  = conditioning_0->forward(ctx, condition);
            cx                  = ggml_relu_inplace(ctx->ggml_ctx, cx);
            cx                  = conditioning_2->forward(ctx, cx);
            if (depth == 3) {
                auto conditioning_4 = std::dynamic_pointer_cast<Conv2d>(blocks["conditioning1.4"]);
                cx                  = ggml_relu_inplace(ctx->ggml_ctx, cx);
                cx                  = conditioning_4->forward(ctx, cx);
            }

            if (!is_conv2d) {
                const int64_t width    = cx->ne[0];
                const int64_t height   = cx->ne[1];
                const int64_t channels = cx->ne[2];
                const int64_t batch    = cx->ne[3];
                cx = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, cx, 1, 2, 0, 3));
                cx = ggml_reshape_3d(ctx->ggml_ctx, cx, channels, width * height, batch);
            }
            ctx->persist_cache_tensor(cache_name, cx);
        }

        if (is_conv2d) {
            if (cx->ne[3] != x->ne[3]) {
                auto repeated = ggml_new_tensor_4d(ctx->ggml_ctx, cx->type, cx->ne[0], cx->ne[1], cx->ne[2], x->ne[3]);
                cx            = ggml_repeat(ctx->ggml_ctx, cx, repeated);
            }
        } else if (cx->ne[2] != x->ne[2]) {
            auto repeated = ggml_new_tensor_3d(ctx->ggml_ctx, cx->type, cx->ne[0], cx->ne[1], x->ne[2]);
            cx            = ggml_repeat(ctx->ggml_ctx, cx, repeated);
        }

        auto down = std::dynamic_pointer_cast<UnaryBlock>(blocks["down.0"]);
        auto mid  = std::dynamic_pointer_cast<UnaryBlock>(blocks["mid.0"]);
        auto up   = std::dynamic_pointer_cast<UnaryBlock>(blocks["up.0"]);

        auto h = down->forward(ctx, x);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = ggml_concat(ctx->ggml_ctx, cx, h, is_conv2d ? 2 : 0);
        h      = mid->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = up->forward(ctx, h);
        return ggml_ext_scale(ctx->ggml_ctx, h, strength, true);
    }
};

class LLLiteModel {
protected:
    std::map<std::string, std::shared_ptr<LLLiteModule>> modules;

    static std::string attention_to_module_name(std::string attention_name) {
        const std::string unet_prefix = "model.diffusion_model.";
        if (attention_name.rfind(unet_prefix, 0) != 0) {
            return {};
        }
        attention_name.erase(0, unet_prefix.size());
        std::replace(attention_name.begin(), attention_name.end(), '.', '_');
        return "lllite_unet_" + attention_name;
    }

public:
    void init(ggml_context* ctx,
              const String2TensorStorage& tensor_storage_map) {
        std::set<std::string> module_names;
        const std::string prefix = "lllite.lllite_unet_";
        for (const auto& [tensor_name, storage] : tensor_storage_map) {
            (void)storage;
            if (tensor_name.rfind(prefix, 0) != 0) {
                continue;
            }
            const size_t module_end = tensor_name.find('.', prefix.size());
            if (module_end != std::string::npos) {
                module_names.insert(tensor_name.substr(std::string("lllite.").size(), module_end - std::string("lllite.").size()));
            }
        }

        for (const auto& module_name : module_names) {
            auto module = std::make_shared<LLLiteModule>(module_name, tensor_storage_map);
            module->init(ctx, tensor_storage_map, "lllite." + module_name);
            modules[module_name] = std::move(module);
        }
        if (!modules.empty()) {
            LOG_INFO("ControlNet-LLLite: initialized %zu attention modules", modules.size());
        }
    }

    bool empty() const {
        return modules.empty();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        for (const auto& [module_name, module] : modules) {
            module->get_param_tensors(tensors, "lllite." + module_name);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         const std::string& attention_name,
                         ggml_tensor* x,
                         ggml_tensor* condition,
                         float strength) {
        const std::string module_name = attention_to_module_name(attention_name);
        auto it                       = modules.find(module_name);
        if (it == modules.end()) {
            return x;
        }
        return ggml_add(ctx->ggml_ctx, x, it->second->forward(ctx, x, condition, strength));
    }
};

#endif  // __SD_MODEL_ADAPTER_LLLITE_HPP__
