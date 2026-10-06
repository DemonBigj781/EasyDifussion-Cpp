#include <fstream>
#include <iostream>

#include "model/diffusion/anima.hpp"
#include "model_manager.h"
#include "stable-diffusion.h"

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("Expected BASE_MODEL ADAPTER SIGLIP_FEATURES");
        sd_set_log_callback([](sd_log_level_t, const char* text, void*) { std::cerr << text; }, nullptr);
        const char* backend_name = std::getenv("SD_TEST_BACKEND");
        auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
            ggml_backend_init_by_name(backend_name ? backend_name : "CPU", nullptr), ggml_backend_free);
        if (!backend) throw std::runtime_error("Requested test backend unavailable");
        auto manager = std::make_shared<ModelManager>();
        manager->set_n_threads(1);
        manager->set_enable_mmap(true);
        manager->set_writable_mmap(false);
        manager->set_keep_compute_params(true);
        auto& loader = manager->loader();
        if (!loader.init_from_file(argv[1], "model.diffusion_model.") ||
            !loader.init_from_file(argv[2], "anima_ip.")) throw std::runtime_error("Checkpoint load failed");
        AnimaIP::validate_checkpoint(loader.get_tensor_storage_map(), loader.get_metadata());
        loader.convert_tensors_name();
        Anima::AnimaRunner runner(backend.get(), loader.get_tensor_storage_map(), "model.diffusion_model", manager);
        if (!manager->register_runner_params("anima", runner, "model.diffusion_model", ModelManager::ResidencyMode::Disk,
                                               backend.get(), backend.get()) || !manager->validate_registered_tensors()) {
            throw std::runtime_error("Checkpoint registration failed");
        }
        sd::Tensor<float> features({768, 1024});
        std::ifstream stream(argv[3], std::ios::binary);
        if (!stream.read(reinterpret_cast<char*>(features.data()), features.numel() * sizeof(float))) {
            throw std::runtime_error("Cannot read SigLIP2 features");
        }
        sd::Tensor<float> x({16, 16, 16, 1});
        sd::Tensor<float> context({1024, 16, 1});
        sd::Tensor<float> time({1}, {500.f});
        for (int64_t i = 0; i < x.numel(); ++i) x.values()[i] = std::sin(float(i) * 0.13f);
        for (int64_t i = 0; i < context.numel(); ++i) context.values()[i] = std::cos(float(i) * 0.07f);
        auto run = [&](bool use_image, float strength, bool use_lora) {
            DiffusionParams params;
            params.x = &x;
            params.timesteps = &time;
            params.context = &context;
            AnimaDiffusionExtra extra;
            extra.ip_context = use_image ? &features : nullptr;
            extra.ip_strength = strength;
            extra.ip_lora = use_lora;
            params.extra = extra;
            auto result = runner.compute(1, params);
            if (result.shape() != x.shape()) throw std::runtime_error("Invalid denoiser output shape");
            for (float value : result.values()) {
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite denoiser output");
            }
            return result;
        };
        const auto baseline = run(false, 0.f, false);
        const auto conditioned = run(true, 0.35f, true);
        const auto negative = run(false, 0.35f, true);
        const auto restored = run(false, 0.f, false);
        const auto zero = run(true, 0.f, false);
        if (baseline.values() != restored.values() || baseline.values() != zero.values()) {
            throw std::runtime_error("IP-Adapter leaked state into disabled or zero-strength inference");
        }
        if (conditioned.values() == baseline.values() || negative.values() == baseline.values() ||
            conditioned.values() == negative.values()) {
            throw std::runtime_error("Conditioning or cross-attention LoRA had no effect");
        }
        runner.runner_done();
        std::cout << "Anima enabled/negative/disabled/zero lifecycle checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
