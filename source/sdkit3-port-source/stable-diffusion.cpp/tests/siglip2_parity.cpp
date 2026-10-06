#include <fstream>
#include <iostream>

#include "model/te/siglip2.hpp"
#include "model_manager.h"
#include "stable-diffusion.h"

template <typename T>
std::vector<T> read_values(const char* path, size_t count) {
    std::ifstream stream(path, std::ios::binary);
    std::vector<T> values(count);
    if (!stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(T)) ||
        stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(std::string("Wrong input length: ") + path);
    }
    return values;
}

void write_values(const char* path, const sd::Tensor<float>& tensor) {
    if (tensor.empty()) throw std::runtime_error("Empty native output");
    std::ofstream stream(path, std::ios::binary);
    if (!stream.write(reinterpret_cast<const char*>(tensor.data()), tensor.numel() * sizeof(float))) {
        throw std::runtime_error(std::string("Cannot write: ") + path);
    }
}

int main(int argc, char** argv) {
    try {
        sd_set_log_callback([](sd_log_level_t, const char* text, void*) { std::cerr << text; }, nullptr);
        if (argc == 6 && std::string(argv[1]) == "preprocess") {
            const int width = std::stoi(argv[2]);
            const int height = std::stoi(argv[3]);
            if (width < 1 || height < 1 || width > 32768 || height > 32768) {
                throw std::invalid_argument("Invalid image extent");
            }
            const auto rgb = read_values<uint8_t>(argv[4], size_t(width) * height * 3);
            write_values(argv[5], SigLIP2::preprocess(rgb.data(), width, height));
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "encode") {
            const char* backend_name = std::getenv("SD_TEST_BACKEND");
            auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
                ggml_backend_init_by_name(backend_name ? backend_name : "CPU", nullptr), ggml_backend_free);
            if (!backend) throw std::runtime_error("CPU backend unavailable");
            auto manager = std::make_shared<ModelManager>();
            manager->set_n_threads(1);
            auto& loader = manager->loader();
            if (!loader.init_from_file(argv[2])) throw std::runtime_error("Model load failed");
            SigLIP2::Runner runner(backend.get(), loader.get_tensor_storage_map(), manager);
            if (!manager->register_runner_params("siglip2", runner, ModelManager::ResidencyMode::Disk,
                                                   backend.get(), backend.get()) || !manager->validate_registered_tensors()) {
                throw std::runtime_error("Weight registration failed");
            }
            auto data = read_values<float>(argv[3], 512 * 512 * 3);
            write_values(argv[4], runner.compute(1, sd::Tensor<float>({512, 512, 3, 1}, std::move(data))));
            runner.free_compute_buffer();
            return 0;
        }
        throw std::invalid_argument("Usage: siglip2-parity preprocess W H RGB OUTPUT | encode MODEL PIXELS OUTPUT");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
