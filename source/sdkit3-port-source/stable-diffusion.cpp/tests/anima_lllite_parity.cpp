#include <fstream>
#include <iostream>
#include "model/adapter/anima_lllite.hpp"
#include "model_manager.h"
#include "stable-diffusion.h"

struct ParityRunner : GGMLRunner {
    AnimaLLLite::Adapter adapter;
    ParityRunner(ggml_backend_t backend, const String2TensorStorage& storage, std::shared_ptr<ModelManager> manager)
        : GGMLRunner(backend, manager) { adapter.init(params_ctx, storage); }
    std::string get_desc() override { return "anima-lllite-parity"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) { adapter.get_param_tensors(tensors); }
    sd::Tensor<float> run(const sd::Tensor<float>& image, const sd::Tensor<float>& x, bool encode_only) {
        auto graph = [&]() {
            auto g = ggml_new_graph_custom(compute_ctx, 16384, false);
            auto ctx = get_context();
            auto condition = adapter.encode(&ctx, make_input(image));
            auto out = encode_only ? condition : adapter.apply(&ctx, "model.diffusion_model.net.blocks.0.self_attn.q_proj", make_input(x), condition, 1.f);
            ggml_build_forward_expand(g, out);
            return g;
        };
        return take_or_empty(compute<float>(graph, 1));
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Expected MODEL FIXTURE_DIRECTORY");
        const std::string directory = argv[2];
        const char* name = std::getenv("SD_TEST_BACKEND");
        auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_init_by_name(name ? name : "CPU", nullptr), ggml_backend_free);
        if (!backend) throw std::runtime_error("Backend unavailable");
        auto manager = std::make_shared<ModelManager>();
        manager->set_n_threads(1);
        if (!manager->loader().init_from_file(argv[1], "lllite.")) throw std::runtime_error("Cannot load weights");
        ParityRunner runner(backend.get(), manager->loader().get_tensor_storage_map(), manager);
        if (!manager->register_runner_params("lllite", runner, ModelManager::ResidencyMode::Disk, backend.get(), backend.get())
            || !manager->validate_registered_tensors()) throw std::runtime_error("Invalid tensor registration");
        sd::Tensor<float> image({64,64,3,1}), x({2048,16,1});
        for (auto entry : {std::make_pair("image.f32", &image), std::make_pair("x.f32", &x)}) {
            std::ifstream file(directory + "/" + entry.first, std::ios::binary);
            if (!file.read(reinterpret_cast<char*>(entry.second->data()), entry.second->numel() * sizeof(float))) throw std::runtime_error("Missing fixture");
        }
        for (bool encode : {true, false}) {
            auto result = runner.run(image, x, encode);
            if (result.empty()) throw std::runtime_error("Computation failed");
            std::ofstream file(directory + (encode ? "/native-condition.f32" : "/native-output.f32"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(result.data()), result.numel() * sizeof(float));
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
