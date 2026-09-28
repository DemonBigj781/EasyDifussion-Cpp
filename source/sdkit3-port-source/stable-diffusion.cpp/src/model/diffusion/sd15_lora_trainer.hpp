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

// Minimal native SD 1.x epsilon-prediction trainer. Dataset rows are packed as
// [noisy latent, timestep, CLIP hidden states]; labels are the target noise.
// Image/VAE and text-encoder preprocessing are intentionally kept outside this
// graph so they can be cached once and reused across optimizer steps.
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

public:
    SD15LoraTrainingRunner(ggml_backend_t backend,
                           const String2TensorStorage& tensor_storage_map,
                           const std::string& prefix,
                           int rank,
                           float alpha = -1.0f,
                           uint32_t seed = 0x51D15u,
                           std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
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
    }

    ~SD15LoraTrainingRunner() override {
        if (optimizer_ != nullptr) {
            ggml_opt_free(optimizer_);
            optimizer_ = nullptr;
        }
        ggml_backend_buffer_free(input_buffer_);
        input_buffer_ = nullptr;
        runner_done();
    }

    const TrainingGraph& prepare_training_graph(int64_t latent_width,
                                                int64_t latent_height,
                                                int64_t batch_size,
                                                int64_t context_tokens = 77) {
        if (optimizer_ != nullptr || latent_width <= 0 || latent_height <= 0 || batch_size <= 0 || context_tokens <= 0) {
            throw std::invalid_argument("invalid or repeated SD 1.x training graph request");
        }
        if (config.context_dim != 768) {
            throw std::invalid_argument("this first SD 1.x training path expects 768-wide CLIP context");
        }

        reset_compute_ctx();
        prepare_build_in_tensor_before();
        const int64_t latent_features = latent_width * latent_height * config.in_channels;
        const int64_t context_features = context_tokens * config.context_dim;
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
        ggml_tensor* context_view = ggml_view_3d(compute_ctx,
                                                 packed,
                                                 config.context_dim,
                                                 context_tokens,
                                                 batch_size,
                                                 config.context_dim * sizeof(float),
                                                 static_cast<size_t>(context_features) * sizeof(float),
                                                 context_offset);
        ggml_tensor* context = ggml_reshape_3d(compute_ctx,
                                               ggml_cont(compute_ctx, context_view),
                                               config.context_dim,
                                               context_tokens,
                                               batch_size);

        ggml_cgraph* graph = new_graph_custom(UNET_GRAPH_SIZE);
        auto runner_ctx = get_context();
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

    std::map<std::string, ggml_tensor*> trainable_parameter_map() const {
        return trainable_adapter_->trainable_parameter_map();
    }

    void set_learning_rate(float learning_rate) {
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0f) {
            throw std::invalid_argument("learning rate must be finite and positive");
        }
        optimizer_params_.adamw.alpha = learning_rate;
    }

    int64_t packed_feature_count(int64_t latent_width, int64_t latent_height, int64_t context_tokens = 77) const {
        return latent_width * latent_height * config.in_channels + 1 + context_tokens * config.context_dim;
    }
};

#endif  // __SD_MODEL_DIFFUSION_SD15_LORA_TRAINER_HPP__
