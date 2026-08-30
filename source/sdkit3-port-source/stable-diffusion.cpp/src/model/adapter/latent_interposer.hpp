#ifndef __SD_MODEL_ADAPTER_LATENT_INTERPOSER_HPP__
#define __SD_MODEL_ADAPTER_LATENT_INTERPOSER_HPP__

#include <memory>
#include <string>

#include "model/common/block.hpp"

// Native inference implementation of city96/SD-Latent-Interposer revision 4.0.
// The original models are small FP32 Conv2d/BatchNorm networks. Keeping the
// weights in FP32 is intentional and matches the reference ComfyUI node.
class LatentInterposerConv2d : public Conv2d {
protected:
    void init_params(ggml_context* ctx,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "") override {
        (void)tensor_storage_map;
        this->prefix     = prefix;
        params["weight"] = ggml_new_tensor_4d(ctx,
                                                GGML_TYPE_F32,
                                                kernel_size.second,
                                                kernel_size.first,
                                                in_channels,
                                                out_channels);
        if (bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

public:
    LatentInterposerConv2d(int64_t in_channels,
                           int64_t out_channels,
                           std::pair<int, int> kernel_size,
                           std::pair<int, int> stride  = {1, 1},
                           std::pair<int, int> padding = {0, 0},
                           bool bias                   = true)
        : Conv2d(in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 {1, 1},
                 bias) {}
};

class LatentInterposerBatchNorm2d : public UnaryBlock {
protected:
    int64_t channels;

    void init_params(ggml_context* ctx,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "") override {
        (void)tensor_storage_map;
        (void)prefix;
        params["weight"]              = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        params["bias"]                = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        params["running_mean"]        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        params["running_var"]         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        params["num_batches_tracked"] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    }

    ggml_tensor* channel_broadcast(ggml_context* ctx, ggml_tensor* x) {
        return ggml_reshape_4d(ctx, x, 1, 1, channels, 1);
    }

public:
    explicit LatentInterposerBatchNorm2d(int64_t channels)
        : channels(channels) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto mean   = channel_broadcast(ctx->ggml_ctx, params["running_mean"]);
        auto var    = ggml_scale_bias(ctx->ggml_ctx, params["running_var"], 1.f, 1e-5f);
        auto stddev = channel_broadcast(ctx->ggml_ctx, ggml_sqrt(ctx->ggml_ctx, var));
        auto weight = channel_broadcast(ctx->ggml_ctx, params["weight"]);
        auto bias   = channel_broadcast(ctx->ggml_ctx, params["bias"]);

        x = ggml_div(ctx->ggml_ctx, ggml_sub(ctx->ggml_ctx, x, mean), stddev);
        return ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, weight), bias);
    }
};

class LatentInterposerExtractBlock : public UnaryBlock {
public:
    LatentInterposerExtractBlock(int64_t in_channels, int64_t out_channels) {
        blocks["short"]  = std::make_shared<LatentInterposerConv2d>(in_channels, out_channels, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
        blocks["long.0"] = std::make_shared<LatentInterposerConv2d>(in_channels, out_channels, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
        blocks["long.2"] = std::make_shared<LatentInterposerConv2d>(out_channels, out_channels, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
        blocks["long.4"] = std::make_shared<LatentInterposerConv2d>(out_channels, out_channels, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto short_path = std::dynamic_pointer_cast<UnaryBlock>(blocks["short"])->forward(ctx, x);
        auto long_path  = std::dynamic_pointer_cast<UnaryBlock>(blocks["long.0"])->forward(ctx, x);
        long_path       = ggml_silu_inplace(ctx->ggml_ctx, long_path);
        long_path       = std::dynamic_pointer_cast<UnaryBlock>(blocks["long.2"])->forward(ctx, long_path);
        long_path       = ggml_silu_inplace(ctx->ggml_ctx, long_path);
        long_path       = std::dynamic_pointer_cast<UnaryBlock>(blocks["long.4"])->forward(ctx, long_path);
        return ggml_relu_inplace(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, long_path, short_path));
    }
};

class LatentInterposerResBlock : public UnaryBlock {
public:
    explicit LatentInterposerResBlock(int64_t channels) {
        blocks["norm"]   = std::make_shared<LatentInterposerBatchNorm2d>(channels);
        blocks["long.0"] = std::make_shared<LatentInterposerConv2d>(channels, channels, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
        blocks["long.2"] = std::make_shared<LatentInterposerConv2d>(channels, channels, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
        blocks["long.4"] = std::make_shared<LatentInterposerConv2d>(channels, channels, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto normalized = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm"])->forward(ctx, x);
        auto long_path  = std::dynamic_pointer_cast<UnaryBlock>(blocks["long.0"])->forward(ctx, normalized);
        long_path       = ggml_silu_inplace(ctx->ggml_ctx, long_path);
        long_path       = std::dynamic_pointer_cast<UnaryBlock>(blocks["long.2"])->forward(ctx, long_path);
        long_path       = ggml_silu_inplace(ctx->ggml_ctx, long_path);
        long_path       = std::dynamic_pointer_cast<UnaryBlock>(blocks["long.4"])->forward(ctx, long_path);
        return ggml_relu_inplace(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, long_path, normalized));
    }
};

class LatentInterposerModel : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    int64_t mid_channels;
    int blocks_count;

public:
    LatentInterposerModel(int64_t in_channels,
                          int64_t out_channels,
                          int64_t mid_channels = 64,
                          int blocks_count      = 12)
        : in_channels(in_channels),
          out_channels(out_channels),
          mid_channels(mid_channels),
          blocks_count(blocks_count) {
        blocks["head"] = std::make_shared<LatentInterposerExtractBlock>(in_channels, mid_channels);
        for (int i = 1; i <= blocks_count; ++i) {
            blocks["core." + std::to_string(i)] = std::make_shared<LatentInterposerResBlock>(mid_channels);
        }
        blocks["core." + std::to_string(blocks_count + 1)] = std::make_shared<LatentInterposerBatchNorm2d>(mid_channels);
        blocks["tail"] = std::make_shared<LatentInterposerConv2d>(mid_channels,
                                                                   out_channels,
                                                                   std::pair<int, int>{3, 3},
                                                                   std::pair<int, int>{1, 1},
                                                                   std::pair<int, int>{1, 1});
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        x = std::dynamic_pointer_cast<UnaryBlock>(blocks["head"])->forward(ctx, x);
        for (int i = 1; i <= blocks_count; ++i) {
            x = std::dynamic_pointer_cast<UnaryBlock>(blocks["core." + std::to_string(i)])->forward(ctx, x);
        }
        x = std::dynamic_pointer_cast<UnaryBlock>(blocks["core." + std::to_string(blocks_count + 1)])->forward(ctx, x);
        x = ggml_silu_inplace(ctx->ggml_ctx, x);
        return std::dynamic_pointer_cast<UnaryBlock>(blocks["tail"])->forward(ctx, x);
    }
};

struct LatentInterposerRunner : public GGMLRunner {
    int64_t in_channels  = 0;
    int64_t out_channels = 0;
    std::string weight_prefix;
    std::unique_ptr<LatentInterposerModel> model;

    LatentInterposerRunner(ggml_backend_t backend,
                           const String2TensorStorage& tensor_storage_map,
                           const std::string& prefix = "latent_interposer",
                           std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager), weight_prefix(prefix) {
        const auto head = tensor_storage_map.find(weight_prefix + ".head.long.0.weight");
        const auto tail = tensor_storage_map.find(weight_prefix + ".tail.weight");
        if (head == tensor_storage_map.end() || tail == tensor_storage_map.end()) {
            throw std::invalid_argument("latent interposer weights are missing head/tail tensors");
        }
        in_channels              = head->second.ne[2];
        const int64_t mid_channels = head->second.ne[3];
        out_channels             = tail->second.ne[3];
        model                    = std::make_unique<LatentInterposerModel>(in_channels, out_channels, mid_channels, 12);
        model->init(params_ctx, tensor_storage_map, weight_prefix);
        LOG_INFO("Latent Interposer '%s': initialized v4.0 network (%" PRId64 " -> %" PRId64 " channels)",
                 weight_prefix.c_str(), in_channels, out_channels);
    }

    std::string get_desc() override {
        return "latent_interposer";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        model->get_param_tensors(tensors, weight_prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& input) {
        ggml_cgraph* graph = new_graph_custom(1 << 15);
        auto x             = make_input(input);
        auto runner_ctx    = get_context();
        auto output        = model->forward(&runner_ctx, x);
        ggml_build_forward_expand(graph, output);
        return graph;
    }

    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& input) {
        auto get_graph = [&]() -> ggml_cgraph* { return build_graph(input); };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads), input.dim());
    }
};

#endif  // __SD_MODEL_ADAPTER_LATENT_INTERPOSER_HPP__
