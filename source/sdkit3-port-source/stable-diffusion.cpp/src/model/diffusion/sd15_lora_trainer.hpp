#ifndef __SD_MODEL_DIFFUSION_SD15_LORA_TRAINER_HPP__
#define __SD_MODEL_DIFFUSION_SD15_LORA_TRAINER_HPP__

#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "ggml-opt.h"
#include "model/adapter/lora.hpp"
#include "model/diffusion/unet.hpp"
#include "model/te/clip.hpp"

// Minimal native SD 1.x epsilon-prediction trainer. Dataset rows are packed as
// [noisy latent, timestep, CLIP hidden states]; labels are the target noise.
// VAE latents are cached. With a positive text-encoder rate the packed rows
// contain token IDs instead of cached CLIP states, and CLIP joins the loss graph.
class SD15LoraTrainingRunner : public UNetModelRunner {
public:
    struct TrainingGraph {
        ggml_cgraph* graph = nullptr;
        ggml_tensor* inputs = nullptr;  // [packed_features, batch]
        ggml_tensor* outputs = nullptr; // [latent_features, batch]
        int64_t packed_features = 0;
        int64_t latent_features = 0;
    };

private:
    std::shared_ptr<MultiLoraAdapter> trainable_adapter_;
    ggml_opt_context_t optimizer_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    ggml_opt_optimizer_params optimizer_params_{};
    TrainingGraph graph_{};
    std::unique_ptr<CLIPTextModel> text_encoder_;
    float text_encoder_learning_rate_ = 0.0f;
    float text_encoder_lr_scale_ = 0.0f;
    ggml_backend_buffer_t mask_buffer_ = nullptr;

public:
    SD15LoraTrainingRunner(ggml_backend_t backend,
                           const String2TensorStorage& tensor_storage_map,
                           const std::string& prefix,
                           int rank,
                           float alpha = -1.0f,
                           uint32_t seed = 0x51D15u,
                           std::shared_ptr<RunnerWeightManager> weight_manager = nullptr,
                           float text_encoder_learning_rate = 0.0f)
        : UNetModelRunner(backend, tensor_storage_map, prefix, VERSION_SD1, std::move(weight_manager)) {
        if (!sd_version_is_sd1(config.version) || config.in_channels != 4 || config.out_channels != 4) {
            throw std::invalid_argument("native LoRA trainer currently requires a 4-channel SD 1.x UNet");
        }
        trainable_adapter_ = std::make_shared<MultiLoraAdapter>(std::vector<std::shared_ptr<LoraModel>>{});
        if (!trainable_adapter_->enable_trainable_attention_lora(rank, alpha, seed)) {
            throw std::invalid_argument("invalid SD 1.x LoRA rank or alpha");
        }
        set_weight_adapter(trainable_adapter_);
        set_training_graph_enabled(true);
        optimizer_params_ = ggml_opt_get_default_optimizer_params(nullptr);
        if (!std::isfinite(text_encoder_learning_rate) || text_encoder_learning_rate < 0.0f) {
            throw std::invalid_argument("text encoder learning rate must be finite and non-negative");
        }
        text_encoder_learning_rate_ = text_encoder_learning_rate;
        if (text_encoder_learning_rate_ > 0.0f) {
            text_encoder_ = std::make_unique<CLIPTextModel>(OPENAI_CLIP_VIT_L_14, true);
            text_encoder_->init(params_ctx, tensor_storage_map, "cond_stage_model.transformer.text_model");
        }
    }

    ~SD15LoraTrainingRunner() override {
        if (optimizer_ != nullptr) {
            ggml_opt_free(optimizer_);
            optimizer_ = nullptr;
        }
        ggml_backend_buffer_free(input_buffer_);
        input_buffer_ = nullptr;
        ggml_backend_buffer_free(mask_buffer_);
        mask_buffer_ = nullptr;
        runner_done();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
        UNetModelRunner::get_param_tensors(tensors, prefix);
        if (text_encoder_) {
            text_encoder_->get_param_tensors(tensors, "cond_stage_model.transformer.text_model");
        }
    }

    const TrainingGraph& prepare_training_graph(int64_t latent_width,
                                                int64_t latent_height,
                                                int64_t batch_size,
                                                int64_t context_tokens = 77) {
        if (graph_.graph != nullptr || optimizer_ != nullptr || latent_width <= 0 || latent_height <= 0 || batch_size <= 0 || context_tokens <= 0) {
            throw std::invalid_argument("invalid or repeated SD 1.x training graph request");
        }
        if (config.context_dim != 768) {
            throw std::invalid_argument("this first SD 1.x training path expects 768-wide CLIP context");
        }

        reset_compute_ctx();
        prepare_build_in_tensor_before();
        const int64_t latent_features = latent_width * latent_height * config.in_channels;
        if (text_encoder_ && context_tokens != text_encoder_->n_token) {
            throw std::invalid_argument("text encoder training requires its native 77-token context");
        }
        const int64_t context_features = text_encoder_ ? context_tokens : context_tokens * config.context_dim;
        const int64_t packed_features = latent_features + 1 + context_features;
        ggml_tensor* packed = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, packed_features, batch_size);
        if (packed == nullptr) {
            throw std::runtime_error("could not allocate packed SD 1.x training input");
        }
        ggml_set_name(packed, "sd15_training_packed_inputs");
        ggml_set_input(packed);

        const size_t sample_stride = packed->nb[1];
        ggml_tensor* latent_view = ggml_view_4d(compute_ctx,
                                                packed,
                                                latent_width,
                                                latent_height,
                                                config.in_channels,
                                                batch_size,
                                                latent_width * sizeof(float),
                                                latent_width * latent_height * sizeof(float),
                                                sample_stride,
                                                0);
        ggml_tensor* latent = ggml_cont(compute_ctx, latent_view);
        ggml_tensor* timestep_view = ggml_view_2d(compute_ctx,
                                                  packed,
                                                  1,
                                                  batch_size,
                                                  sample_stride,
                                                  static_cast<size_t>(latent_features) * sizeof(float));
        ggml_tensor* timesteps = ggml_reshape_1d(compute_ctx, ggml_cont(compute_ctx, timestep_view), batch_size);
        const size_t context_offset = static_cast<size_t>(latent_features + 1) * sizeof(float);
        auto runner_ctx = get_context();
        ggml_tensor* context = nullptr;
        if (text_encoder_) {
            auto token_view = ggml_view_2d(compute_ctx, packed, context_tokens, batch_size,
                                           sample_stride, context_offset);
            auto token_ids = ggml_cast(compute_ctx, ggml_cont(compute_ctx, token_view), GGML_TYPE_I32);
            auto mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, context_tokens, context_tokens);
            ggml_set_input(mask);
            mask_buffer_ = ggml_backend_alloc_buffer(runtime_backend, ggml_nbytes(mask));
            if (!mask_buffer_ || ggml_backend_tensor_alloc(mask_buffer_, mask,
                    ggml_backend_buffer_get_base(mask_buffer_)) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("could not allocate CLIP causal mask");
            }
            std::vector<float> values(static_cast<size_t>(context_tokens * context_tokens), 0.0f);
            for (int64_t query = 0; query < context_tokens; ++query) {
                for (int64_t key = query + 1; key < context_tokens; ++key) {
                    values[query * context_tokens + key] = -INFINITY;
                }
            }
            ggml_backend_tensor_set(mask, values.data(), 0, ggml_nbytes(mask));
            context = text_encoder_->forward(&runner_ctx, token_ids, nullptr, mask, 0, false, 1);
        } else {
            ggml_tensor* context_view = ggml_view_3d(compute_ctx,
                                                     packed,
                                                     config.context_dim,
                                                     context_tokens,
                                                     batch_size,
                                                     config.context_dim * sizeof(float),
                                                     sample_stride, // Full packed sample, not just its CLIP context.
                                                     context_offset);
            context = ggml_reshape_3d(compute_ctx,
                                       ggml_cont(compute_ctx, context_view),
                                       config.context_dim,
                                       context_tokens,
                                       batch_size);
        }

        ggml_cgraph* graph = new_graph_custom(UNET_GRAPH_SIZE);
        ggml_tensor* prediction = unet.forward(&runner_ctx,
                                               latent,
                                               timesteps,
                                               context,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               {},
                                               0.0f);
        ggml_tensor* flattened_prediction = ggml_reshape_2d(compute_ctx,
                                                            prediction,
                                                            latent_features,
                                                            batch_size);
        ggml_set_name(flattened_prediction, "sd15_epsilon_prediction_flat");
        ggml_build_forward_expand(graph, flattened_prediction);

        for (const auto& entry : debug_tensors) {
            if (entry.first != nullptr) {
                ggml_build_forward_expand(graph, entry.first);
            }
        }
        for (const auto& entry : cache_tensor_map) {
            if (entry.second != nullptr) {
                ggml_build_forward_expand(graph, entry.second);
            }
        }
        prepare_build_in_tensor_after(graph);

        // The optimizer scheduler also receives the much larger gradient and
        // update graphs, so reserve its node/leaf tables beyond the UNet
        // forward graph size before ggml_opt builds those graphs.
        // Training must execute entirely on the selected accelerator. If SYCL
        // lacks an operation required by this graph, fail instead of silently
        // dispatching that operation to the CPU.
        if (!ensure_sched(graph, MAX_GRAPH_SIZE, false)) {
            throw std::runtime_error("could not prepare SD 1.x training backend scheduler");
        }
        if (!assign_graph_cut_layer_split_backends(graph)) {
            throw std::runtime_error("could not assign SD 1.x training graph backends");
        }
        std::vector<ggml_tensor*> graph_params;
        std::vector<ggml_tensor*> params_to_prepare;
        const std::vector<ggml_tensor*> trainable_params = trainable_adapter_->trainable_parameters();
        const std::unordered_set<const ggml_tensor*> externally_managed_params(trainable_params.begin(), trainable_params.end());
        if (!prepare_execute_graph_weights(graph, graph_params, params_to_prepare, false, &externally_managed_params)) {
            throw std::runtime_error("could not prepare frozen SD 1.x model weights");
        }
        graph_ = {graph, packed, flattened_prediction, packed_features, latent_features};
        return graph_;
    }

    bool initialize_optimizer(float learning_rate = 1e-4f,
                              float weight_decay = 0.0f,
                              int32_t accumulation_steps = 1) {
        if (graph_.graph == nullptr || optimizer_ != nullptr || !std::isfinite(learning_rate) ||
            learning_rate <= 0.0f || !std::isfinite(weight_decay) || weight_decay < 0.0f ||
            accumulation_steps < 1) {
            return false;
        }
        optimizer_params_.adamw.alpha = learning_rate;
        optimizer_params_.adamw.wd = weight_decay;
        if (text_encoder_ && (!std::isfinite(text_encoder_learning_rate_ / learning_rate) ||
                              text_encoder_learning_rate_ / learning_rate <= 0.0f)) {
            return false;
        }
        text_encoder_lr_scale_ = text_encoder_learning_rate_ / learning_rate;
        if (graph_.inputs->buffer == nullptr) {
            input_buffer_ = ggml_backend_alloc_buffer(runtime_backend, ggml_nbytes(graph_.inputs));
            if (input_buffer_ == nullptr ||
                ggml_backend_tensor_alloc(input_buffer_,
                                          graph_.inputs,
                                          input_buffer_ != nullptr ? ggml_backend_buffer_get_base(input_buffer_) : nullptr) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(input_buffer_);
                input_buffer_ = nullptr;
                return false;
            }
        }
        auto params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR);
        params.ctx_compute = compute_ctx;
        params.inputs = graph_.inputs;
        params.outputs = graph_.outputs;
        params.graph_size = MAX_GRAPH_SIZE;
        params.opt_period = accumulation_steps;
        params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
        params.get_opt_pars_ud = &optimizer_params_;
        params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
        if (text_encoder_) {
            params.get_param_lr_scale = [](const ggml_tensor* parameter, void* userdata) -> float {
                auto* self = static_cast<SD15LoraTrainingRunner*>(userdata);
                return starts_with(parameter->name, "lora.cond_stage_model.")
                    ? self->text_encoder_lr_scale_ : 1.0f;
            };
            params.get_param_lr_scale_ud = this;
        }
        optimizer_ = ggml_opt_init(params);
        return optimizer_ != nullptr;
    }

    bool train_batch(const float* packed_inputs, const float* target_noise, float* loss_out = nullptr) {
        if (optimizer_ == nullptr || packed_inputs == nullptr || target_noise == nullptr) {
            return false;
        }
        ggml_tensor* inputs = ggml_opt_inputs(optimizer_);
        ggml_tensor* labels = ggml_opt_labels(optimizer_);
        if (inputs == nullptr || labels == nullptr) {
            return false;
        }
        ggml_backend_tensor_set(inputs, packed_inputs, 0, ggml_nbytes(inputs));
        ggml_backend_tensor_set(labels, target_noise, 0, ggml_nbytes(labels));
        ggml_opt_alloc(optimizer_, true);
        ggml_opt_eval(optimizer_, nullptr);
        if (loss_out != nullptr) {
            ggml_backend_tensor_get(ggml_opt_loss(optimizer_), loss_out, 0, sizeof(float));
        }
        return true;
    }

    std::vector<ggml_tensor*> trainable_parameters() const {
        return trainable_adapter_->trainable_parameters();
    }

    std::vector<float> predict_noise(const std::vector<float>& packed_inputs) {
        if (!optimizer_ || packed_inputs.size() != static_cast<size_t>(graph_.packed_features))
            throw std::invalid_argument("invalid sampling input");
        ggml_backend_tensor_set(ggml_opt_inputs(optimizer_), packed_inputs.data(), 0,
                                packed_inputs.size() * sizeof(float));
        ggml_opt_alloc(optimizer_, false);
        ggml_opt_eval(optimizer_, nullptr);
        std::vector<float> prediction(static_cast<size_t>(graph_.latent_features));
        ggml_backend_tensor_get(ggml_opt_outputs(optimizer_), prediction.data(), 0,
                                prediction.size() * sizeof(float));
        return prediction;
    }

    std::map<std::string, ggml_tensor*> trainable_parameter_map() const {
        return trainable_adapter_->trainable_parameter_map();
    }

    void set_learning_rate(float learning_rate) {
        if (!std::isfinite(learning_rate) || learning_rate < 0.0f) {
            throw std::invalid_argument("learning rate must be finite and non-negative");
        }
        optimizer_params_.adamw.alpha = learning_rate;
    }

    int64_t packed_feature_count(int64_t latent_width, int64_t latent_height, int64_t context_tokens = 77) const {
        return latent_width * latent_height * config.in_channels + 1 +
               context_tokens * (text_encoder_ ? 1 : config.context_dim);
    }
};

#endif  // __SD_MODEL_DIFFUSION_SD15_LORA_TRAINER_HPP__
