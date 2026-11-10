#include "api/gguf_cpu_weights.hpp"
#include "core/ggml_extend.hpp"
#include "ggml-cpu.h"
#include "gguf.h"
#include "model_io/gguf_io.h"
#include "model_manager.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

static_assert(GGML_MAX_NAME >= 128, "Application tensor-name ABI must be inherited");
static constexpr int inputs = 768, outputs = 3072, tokens = 2;
static const std::string prefix = "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1";
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static float weight_value(int in, int out) { return float((in + 3 * out) % 17 - 8) * .001f; }
static float bias_value(int out) { return float(out % 13 - 6) * .01f; }
static float input_value(int in, int token, int pass) { return float((in + token + pass) % 11 - 5) * .02f; }

static void write_linear_fixture(const std::filesystem::path& file, bool wrong_shape = false) {
    const int width = wrong_shape ? inputs - 1 : inputs;
    auto context = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({std::size_t(width * outputs + outputs) * sizeof(float) + 1024 * 1024, nullptr, false}), &ggml_free);
    auto metadata = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_empty(), &gguf_free);
    require(context && metadata, "fixture allocation failed");
    auto* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, outputs);
    auto* bias = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, outputs);
    ggml_set_name(weight, (prefix + ".weight").c_str());
    ggml_set_name(bias, (prefix + ".bias").c_str());
    for (int out = 0; out < outputs; ++out) {
        for (int in = 0; in < width; ++in)
            static_cast<float*>(weight->data)[out * width + in] = weight_value(in, out);
        static_cast<float*>(bias->data)[out] = bias_value(out);
    }
    gguf_add_tensor(metadata.get(), weight);
    gguf_add_tensor(metadata.get(), bias);
    require(gguf_write_to_file(metadata.get(), file.string().c_str(), false), "fixture write failed");
}
struct ApplicationRunner final : GGMLRunner {
    std::shared_ptr<ModelManager> manager;
    Linear projection{inputs, outputs, true, true, true};
    bool registered = false;
    ApplicationRunner(ggml_backend_t backend, std::shared_ptr<ModelManager> owner)
        : GGMLRunner(backend, owner), manager(std::move(owner)) {
        projection.init(params_ctx, manager->loader().get_tensor_storage_map(), prefix);
    }
    ~ApplicationRunner() override {
        // The manager must release bindings while the runner's tensor context
        // still exists; base GGMLRunner destruction frees that context.
        if (registered && !manager->unregister_param_tensors(get_desc())) std::terminate();
    }
    std::string get_desc() override { return "theory-application-linear-contract"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        projection.get_param_tensors(tensors, prefix);
    }
    void attach(ModelManager::ResidencyMode mode, ggml_backend_t backend) {
        registered = manager->register_runner_params(get_desc(), *this, mode, backend, backend);
        require(registered, "application parameter registration failed");
    }
    sd::Tensor<float> run(const sd::Tensor<float>& input) {
        auto build = [&] {
            auto* graph = ggml_new_graph_custom(compute_ctx, 128, false);
            auto context = get_context();
            auto* output = projection.forward(&context, make_input(input));
            ggml_build_forward_expand(graph, output);
            return graph;
        };
        return take_or_empty(compute<float>(build, 1));
    }
};
static void check_forward(const sd::Tensor<float>& result, int pass) {
    require(result.numel() == outputs * tokens, "application graph returned wrong shape");
    for (int token = 0; token < tokens; ++token) {
        for (int out = 0; out < outputs; ++out) {
            double expected = bias_value(out);
            for (int in = 0; in < inputs; ++in)
                expected += double(weight_value(in, out)) * input_value(in, token, pass);
            const float actual = result.data()[token * outputs + out];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-5)
                throw std::runtime_error("application Linear forward differs from independent reference");
        }
    }
}
static void check_original_shape_boundary(const std::filesystem::path& file) {
    auto context = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({1024 * 1024, nullptr, false}), &ggml_free);
    auto metadata = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_empty(), &gguf_free);
    auto* tensor = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q4_K, 256, 45);
    ggml_set_name(tensor, "conv_in.weight");
    std::memset(tensor->data, 0, ggml_nbytes(tensor));
    const std::int64_t logical[]{320, 4, 3, 3};
    gguf_set_arr_data(metadata.get(), "comfy.gguf.orig_shape.conv_in.weight", GGUF_TYPE_INT64, logical, 4);
    gguf_add_tensor(metadata.get(), tensor);
    require(gguf_write_to_file(metadata.get(), file.string().c_str(), false), "shape fixture write failed");
    auto generic = easyapi::GgufCpuWeights::load(file.string());
    require(generic.tensor_info(0).shape == std::vector<std::int64_t>({256, 45}), "generic storage shape changed");
    std::vector<TensorStorage> application;
    std::string error;
    require(read_gguf_file(file.string(), application, &error) && application.size() == 1,
            "application GGUF shape reader failed");
    const auto& t = application.front();
    require(t.n_dims == 4 && t.ne[0] == 3 && t.ne[1] == 3 && t.ne[2] == 4 && t.ne[3] == 320,
            "application logical convolution shape was not restored");
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto root = std::filesystem::path(argv[1]) /
        ("application-contract-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(root)) return 3;
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
    } cleanup{root};
    try {
        const auto file = root / "clip-mlp-fc1.gguf";
        auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), &ggml_backend_free);
        require(bool(backend), "CPU backend unavailable");
        for (auto mode : {ModelManager::ResidencyMode::Disk, ModelManager::ResidencyMode::ParamBackend}) {
            write_linear_fixture(file);
            {
                auto generic = easyapi::GgufCpuWeights::load(file.string());
                require(generic.tensor_info(0).name == prefix + ".weight" &&
                        generic.tensor_info(0).shape == std::vector<std::int64_t>({inputs, outputs}),
                        "long-name/storage metadata mismatch");
            } // Application must never borrow this independent resource's handles.
            auto manager = std::make_shared<ModelManager>();
            manager->set_n_threads(1);
            require(manager->loader().init_from_file(file.string()), "application model loader rejected fixture");
            {
                ApplicationRunner runner(backend.get(), manager);
                runner.attach(mode, backend.get());
                require(manager->validate_registered_tensors(), "application rejected valid parameter shapes");
                if (mode == ModelManager::ResidencyMode::ParamBackend) {
                    require(manager->load_all_params_eagerly(), "application eager loading failed");
                    std::filesystem::remove(file);
                }
                for (int pass = 0; pass < 3; ++pass) {
                    sd::Tensor<float> input({inputs, tokens});
                    for (int token = 0; token < tokens; ++token)
                        for (int in = 0; in < inputs; ++in)
                            input.data()[token * inputs + in] = input_value(in, token, pass);
                    check_forward(runner.run(input), pass);
                }
            }
            require(manager->tensor_names().empty(), "runner destroyed before releasing manager bindings");
        }
        write_linear_fixture(file, true);
        auto manager = std::make_shared<ModelManager>();
        require(manager->loader().init_from_file(file.string()), "bad-shape fixture parse failed");
        {
            ApplicationRunner runner(backend.get(), manager);
            runner.attach(ModelManager::ResidencyMode::Disk, backend.get());
            require(!manager->validate_registered_tensors(), "application accepted incompatible architecture shape");
        }
        check_original_shape_boundary(root / "original-shape.gguf");
        std::cout << "Application graph: CLIP-sized Linear forward, disk/eager ownership and GGUF shape boundary passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
