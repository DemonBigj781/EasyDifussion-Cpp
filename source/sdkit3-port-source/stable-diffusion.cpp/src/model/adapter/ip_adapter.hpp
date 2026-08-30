#ifndef __SD_MODEL_ADAPTER_IP_ADAPTER_HPP__
#define __SD_MODEL_ADAPTER_IP_ADAPTER_HPP__

#include <cinttypes>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/common/block.hpp"
#include "model/te/clip.hpp"

// Native inference for the original IP-Adapter projection checkpoints.  The
// adapter adds a second, independently normalized image cross-attention result
// to each UNet text cross-attention block before the existing output
// projection.  This deliberately does not concatenate image K/V with text
// K/V: doing so changes the softmax normalization and is not IP-Adapter.
class IPAdapterBlock : public GGMLBlock {
public:
    explicit IPAdapterBlock(const String2TensorStorage& tensor_storage_map,
                            SDVersion version)
        : version_(version) {
        const auto proj_it = tensor_storage_map.find("image_proj.proj.weight");
        const auto key_it  = tensor_storage_map.find("ip_adapter.1.to_k_ip.weight");
        if (proj_it == tensor_storage_map.end() || key_it == tensor_storage_map.end()) {
            LOG_ERROR("IP-Adapter: checkpoint is missing base image projection tensors");
            return;
        }
        if (tensor_storage_map.find("image_proj.latents") != tensor_storage_map.end()) {
            LOG_ERROR("IP-Adapter Plus requires the Perceiver projection path; use a base IP-Adapter checkpoint for now");
            return;
        }

        clip_embedding_dim_ = proj_it->second.ne[0];
        const int64_t projection_size = proj_it->second.ne[1];
        cross_attention_dim_          = key_it->second.ne[0];
        if (cross_attention_dim_ <= 0 || projection_size % cross_attention_dim_ != 0) {
            LOG_ERROR("IP-Adapter: invalid projection shape [in=%" PRId64 ", out=%" PRId64 "] for context dim %" PRId64,
                      clip_embedding_dim_, projection_size, cross_attention_dim_);
            return;
        }
        token_count_ = static_cast<int>(projection_size / cross_attention_dim_);
        if (token_count_ <= 0 || token_count_ > 64) {
            LOG_ERROR("IP-Adapter: invalid image token count %d", token_count_);
            return;
        }

        if (clip_embedding_dim_ == 1280) {
            clip_version_ = OPEN_CLIP_VIT_BIGG_14;
        } else if (clip_embedding_dim_ == 1024) {
            clip_version_ = OPEN_CLIP_VIT_H_14;
        } else {
            LOG_ERROR("IP-Adapter: unsupported CLIP projection dimension %" PRId64,
                      clip_embedding_dim_);
            return;
        }

        blocks["image_proj.proj"] = std::make_shared<Linear>(clip_embedding_dim_, projection_size, true);
        blocks["image_proj.norm"] = std::make_shared<LayerNorm>(cross_attention_dim_);

        build_attention_layer_map();
        for (const auto& [attention_prefix, layer_index] : attention_layer_indices_) {
            const std::string block_prefix = "ip_adapter." + std::to_string(layer_index);
            const auto k_it = tensor_storage_map.find(block_prefix + ".to_k_ip.weight");
            const auto v_it = tensor_storage_map.find(block_prefix + ".to_v_ip.weight");
            if (k_it == tensor_storage_map.end() || v_it == tensor_storage_map.end()) {
                LOG_ERROR("IP-Adapter: missing K/V weights for layer %d (%s)",
                          layer_index, attention_prefix.c_str());
                blocks.clear();
                attention_layer_indices_.clear();
                return;
            }
            if (k_it->second.ne[0] != cross_attention_dim_ ||
                v_it->second.ne[0] != cross_attention_dim_ ||
                k_it->second.ne[1] != v_it->second.ne[1]) {
                LOG_ERROR("IP-Adapter: incompatible K/V shape at layer %d", layer_index);
                blocks.clear();
                attention_layer_indices_.clear();
                return;
            }
            blocks[block_prefix + ".to_k_ip"] =
                std::make_shared<Linear>(cross_attention_dim_, k_it->second.ne[1], false);
            blocks[block_prefix + ".to_v_ip"] =
                std::make_shared<Linear>(cross_attention_dim_, v_it->second.ne[1], false);
        }

        valid_ = !attention_layer_indices_.empty();
        if (valid_) {
            LOG_INFO("IP-Adapter: initialized %zu UNet attention layers, %d image tokens, CLIP dim %" PRId64,
                     attention_layer_indices_.size(), token_count_, clip_embedding_dim_);
        }
    }

    bool valid() const {
        return valid_;
    }

    int token_count() const {
        return token_count_;
    }

    int64_t cross_attention_dim() const {
        return cross_attention_dim_;
    }

    int64_t clip_embedding_dim() const {
        return clip_embedding_dim_;
    }

    CLIPVersion clip_version() const {
        return clip_version_;
    }

    ggml_tensor* project(GGMLRunnerContext* ctx, ggml_tensor* clip_embedding) {
        auto proj = std::dynamic_pointer_cast<Linear>(blocks["image_proj.proj"]);
        auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks["image_proj.norm"]);
        ggml_tensor* tokens = proj->forward(ctx, clip_embedding);
        const int64_t batch = ggml_nelements(tokens) /
                              (cross_attention_dim_ * static_cast<int64_t>(token_count_));
        tokens = ggml_reshape_3d(ctx->ggml_ctx,
                                 tokens,
                                 cross_attention_dim_,
                                 token_count_,
                                 batch);
        return norm->forward(ctx, tokens);
    }

    ggml_tensor* forward_attention(GGMLRunnerContext* ctx,
                                   const std::string& attention_prefix,
                                   ggml_tensor* query,
                                   ggml_tensor* image_context,
                                   int64_t n_head,
                                   float strength) {
        const auto layer_it = attention_layer_indices_.find(attention_prefix);
        if (layer_it == attention_layer_indices_.end() || image_context == nullptr || strength == 0.f) {
            return nullptr;
        }
        const std::string block_prefix = "ip_adapter." + std::to_string(layer_it->second);
        auto to_k = std::dynamic_pointer_cast<Linear>(blocks[block_prefix + ".to_k_ip"]);
        auto to_v = std::dynamic_pointer_cast<Linear>(blocks[block_prefix + ".to_v_ip"]);
        if (!to_k || !to_v) {
            return nullptr;
        }

        ggml_tensor* key   = to_k->forward(ctx, image_context);
        ggml_tensor* value = to_v->forward(ctx, image_context);
        ggml_tensor* out   = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                  ctx->backend,
                                                  query,
                                                  key,
                                                  value,
                                                  n_head,
                                                  nullptr,
                                                  false,
                                                  ctx->flash_attn_enabled);
        return ggml_ext_scale(ctx->ggml_ctx, out, strength, true);
    }

private:
    void add_layer(const std::string& prefix, int ordinal) {
        attention_layer_indices_["model.diffusion_model." + prefix + ".attn2."] = ordinal * 2 + 1;
    }

    void build_attention_layer_map() {
        int ordinal = 0;
        if (sd_version_is_sdxl(version_)) {
            for (int block_id : {4, 5, 7, 8}) {
                const int depth = (block_id == 4 || block_id == 5) ? 2 : 10;
                for (int transformer_index = 0; transformer_index < depth; ++transformer_index) {
                    add_layer("input_blocks." + std::to_string(block_id) +
                                  ".1.transformer_blocks." + std::to_string(transformer_index),
                              ordinal++);
                }
            }
            for (int block_id = 0; block_id < 6; ++block_id) {
                const int depth = block_id < 3 ? 10 : 2;
                for (int transformer_index = 0; transformer_index < depth; ++transformer_index) {
                    add_layer("output_blocks." + std::to_string(block_id) +
                                  ".1.transformer_blocks." + std::to_string(transformer_index),
                              ordinal++);
                }
            }
            for (int transformer_index = 0; transformer_index < 10; ++transformer_index) {
                add_layer("middle_block.1.transformer_blocks." + std::to_string(transformer_index),
                          ordinal++);
            }
        } else if (sd_version_is_sd1(version_)) {
            for (int block_id : {1, 2, 4, 5, 7, 8}) {
                add_layer("input_blocks." + std::to_string(block_id) + ".1.transformer_blocks.0",
                          ordinal++);
            }
            for (int block_id = 3; block_id <= 11; ++block_id) {
                add_layer("output_blocks." + std::to_string(block_id) + ".1.transformer_blocks.0",
                          ordinal++);
            }
            add_layer("middle_block.1.transformer_blocks.0", ordinal++);
        } else {
            LOG_ERROR("IP-Adapter: only SD1.x and SDXL UNets are supported");
        }
    }

    SDVersion version_                  = VERSION_COUNT;
    int64_t clip_embedding_dim_         = 0;
    int64_t cross_attention_dim_        = 0;
    int token_count_                    = 0;
    CLIPVersion clip_version_           = OPEN_CLIP_VIT_H_14;
    bool valid_                         = false;
    std::map<std::string, int> attention_layer_indices_;
};

class IPAdapterModel : public GGMLRunner {
public:
    IPAdapterModel(ggml_backend_t backend,
                   const String2TensorStorage& tensor_storage_map,
                   SDVersion version,
                   std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          adapter_(tensor_storage_map, version) {
        if (adapter_.valid()) {
            adapter_.init(params_ctx, tensor_storage_map);
        }
    }

    std::string get_desc() override {
        return "ip_adapter";
    }

    bool valid() const {
        return adapter_.valid();
    }

    int token_count() const {
        return adapter_.token_count();
    }

    int64_t cross_attention_dim() const {
        return adapter_.cross_attention_dim();
    }

    CLIPVersion clip_version() const {
        return adapter_.clip_version();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        adapter_.get_param_tensors(tensors);
    }

    sd::Tensor<float> project(int n_threads, const sd::Tensor<float>& clip_embedding) {
        auto get_graph = [&]() -> ggml_cgraph* {
            ggml_cgraph* graph      = ggml_new_graph(compute_ctx);
            ggml_tensor* embedding  = make_input(clip_embedding);
            auto runner_ctx         = get_context();
            ggml_tensor* projection = adapter_.project(&runner_ctx, embedding);
            ggml_build_forward_expand(graph, projection);
            return graph;
        };
        return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true, true, true));
    }

    ggml_tensor* forward_attention(GGMLRunnerContext* ctx,
                                   const std::string& attention_prefix,
                                   ggml_tensor* query,
                                   ggml_tensor* image_context,
                                   int64_t n_head,
                                   float strength) {
        return adapter_.forward_attention(ctx,
                                          attention_prefix,
                                          query,
                                          image_context,
                                          n_head,
                                          strength);
    }

    // IP-Adapter attention layers are injected into the UNet graph, whose
    // runner owns a different params context. Hold one explicit compute-backend
    // reference for every adapter tensor while sampling so those external leaf
    // tensors remain allocated and initialized for the whole denoising pass.
    bool prepare_for_unet() {
        if (prepared_for_unet_) {
            return true;
        }
        auto manager = weight_manager.lock();
        if (manager == nullptr) {
            LOG_ERROR("IP-Adapter: weight manager is unavailable");
            return false;
        }
        std::map<std::string, ggml_tensor*> tensors;
        adapter_.get_param_tensors(tensors);
        unet_param_tensors_.clear();
        unet_param_tensors_.reserve(tensors.size());
        for (const auto& [name, tensor] : tensors) {
            (void)name;
            if (tensor != nullptr) {
                unet_param_tensors_.push_back(tensor);
            }
        }
        if (!manager->prepare_params(unet_param_tensors_)) {
            LOG_ERROR("IP-Adapter: failed to prepare attention weights");
            unet_param_tensors_.clear();
            return false;
        }
        prepared_for_unet_ = true;
        return true;
    }

    void release_from_unet() {
        if (!prepared_for_unet_) {
            return;
        }
        auto manager = weight_manager.lock();
        if (manager != nullptr) {
            manager->release_compute_backend_params(unet_param_tensors_);
        }
        unet_param_tensors_.clear();
        prepared_for_unet_ = false;
    }

private:
    IPAdapterBlock adapter_;
    std::vector<ggml_tensor*> unet_param_tensors_;
    bool prepared_for_unet_ = false;
};

#endif  // __SD_MODEL_ADAPTER_IP_ADAPTER_HPP__
