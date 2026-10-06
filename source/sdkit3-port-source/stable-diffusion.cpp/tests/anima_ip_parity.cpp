#include <fstream>
#include <iostream>

#include "model/adapter/anima_ip_adapter.hpp"
#include "model_manager.h"
#include "stable-diffusion.h"

sd::Tensor<float> read_tensor(const std::string& path, std::vector<int64_t> shape) {
    sd::Tensor<float> tensor(std::move(shape));
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char*>(tensor.data()), tensor.numel() * sizeof(float)) ||
        stream.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Invalid input: " + path);
    return tensor;
}

void write_tensor(const std::string& path, const sd::Tensor<float>& tensor) {
    if (tensor.empty()) throw std::runtime_error("Native computation failed");
    std::ofstream stream(path, std::ios::binary);
    if (!stream.write(reinterpret_cast<const char*>(tensor.data()), tensor.numel() * sizeof(float))) {
        throw std::runtime_error("Cannot write: " + path);
    }
}

struct TestRunner : GGMLRunner {
    AnimaIP::Adapter adapter;
    TestRunner(ggml_backend_t backend, const String2TensorStorage& storage,
               std::shared_ptr<ModelManager> manager) : GGMLRunner(backend, manager) {
        adapter.init(params_ctx, storage, "anima_ip");
    }
    std::string get_desc() override { return "anima-ip-parity"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        adapter.get_param_tensors(tensors, "anima_ip");
    }
    sd::Tensor<float> attention(int block, const sd::Tensor<float>& query, const sd::Tensor<float>& tokens,
                                const sd::Tensor<float>& timestep, float strength) {
        auto graph = [&]() {
            auto g = ggml_new_graph(compute_ctx);
            auto ctx = get_context();
            auto q = make_input(query);
            auto output = adapter.image_attention(&ctx, block, q, make_input(tokens), make_input(timestep), strength);
            ggml_build_forward_expand(g, output);
            return g;
        };
        return take_or_empty(compute<float>(graph, 1));
    }
    sd::Tensor<float> lora(int block, const std::string& projection, const sd::Tensor<float>& x) {
        auto graph = [&]() {
            auto g = ggml_new_graph(compute_ctx);
            auto ctx = get_context();
            ggml_build_forward_expand(g, adapter.lora(&ctx, block, projection, make_input(x)));
            return g;
        };
        return take_or_empty(compute<float>(graph, 1));
    }
};

int main(int argc, char** argv) {
    try {
        sd_set_log_callback([](sd_log_level_t, const char* text, void*) { std::cerr << text; }, nullptr);
        if (argc != 6) throw std::invalid_argument("Expected MODEL FIXTURES BLOCK STRENGTH OUTPUT");
        const int block = std::stoi(argv[3]);
        const float strength = std::stof(argv[4]);
        if (block < 0 || block >= 28 || !std::isfinite(strength) || strength < 0) {
            throw std::invalid_argument("Invalid adapter parameters");
        }
        const char* backend_name = std::getenv("SD_TEST_BACKEND");
        auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
            ggml_backend_init_by_name(backend_name ? backend_name : "CPU", nullptr), ggml_backend_free);
        if (!backend) throw std::runtime_error("Missing CPU backend");
        auto manager = std::make_shared<ModelManager>();
        manager->set_n_threads(1);
        auto& loader = manager->loader();
        if (!loader.init_from_file(argv[1], "anima_ip.")) throw std::runtime_error("Cannot read checkpoint");
        AnimaIP::validate_checkpoint(loader.get_tensor_storage_map(), loader.get_metadata());
        TestRunner runner(backend.get(), loader.get_tensor_storage_map(), manager);
        if (!manager->register_runner_params("anima-ip", runner, ModelManager::ResidencyMode::Disk,
                                               backend.get(), backend.get()) || !manager->validate_registered_tensors()) {
            throw std::runtime_error("Weight registration failed");
        }
        const std::string fixtures = argv[2];
        auto query = read_tensor(fixtures + "/query.f32", {2048, 7, 1});
        auto tokens = read_tensor(fixtures + "/tokens.f32", {768, 19, 1});
        auto timestep = read_tensor(fixtures + "/embedding.f32", {2048, 1, 1});
        write_tensor(argv[5], runner.attention(block, query, tokens, timestep, strength));
        for (const auto& name : {"q_proj", "k_proj", "v_proj", "output_proj"}) {
            const bool context = std::string(name) == "k_proj" || std::string(name) == "v_proj";
            sd::Tensor<float> input({context ? 1024 : 2048, 7, 1});
            for (int t = 0; t < 7; ++t) {
                std::copy_n(query.data() + t * 2048, input.shape()[0], input.data() + t * input.shape()[0]);
            }
            write_tensor(fixtures + "/lora-" + std::to_string(block) + "-" + name + ".f32", runner.lora(block, name, input));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
