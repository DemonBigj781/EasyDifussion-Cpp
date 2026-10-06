#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "stable-diffusion.h"

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::invalid_argument("Expected BASE_MODEL TEXT_ENCODER VAE ADAPTER SIGLIP2 OUTPUT_PREFIX");
        sd_set_log_callback([](sd_log_level_t, const char* text, void*) { std::cerr << text; }, nullptr);
        sd_ctx_params_t config;
        sd_ctx_params_init(&config);
        config.diffusion_model_path = argv[1];
        config.llm_path = argv[2];
        config.vae_path = argv[3];
        config.ip_adapter_path = argv[4];
        config.clip_vision_path = argv[5];
        config.n_threads = 1;
        config.backend = "cuda";
        config.params_backend = "cpu";
        config.enable_mmap = true;
        config.keep_compute_params = false;
        config.max_vram = "5";
        const char* flash = std::getenv("SD_TEST_FLASH_ATTENTION");
        config.flash_attn = flash != nullptr && std::string(flash) == "1";
        config.diffusion_flash_attn = config.flash_attn;
        config.rng_type = STD_DEFAULT_RNG;
        auto context = std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)>(new_sd_ctx(&config), free_sd_ctx);
        if (!context) throw std::runtime_error("Native Anima context initialization failed");
        std::vector<uint8_t> reference(128 * 128 * 3);
        for (int y = 0; y < 128; ++y) {
            for (int x = 0; x < 128; ++x) {
                const bool circle = (x - 64) * (x - 64) + (y - 64) * (y - 64) < 40 * 40;
                const size_t index = (y * 128 + x) * 3;
                reference[index] = circle ? 30 : 240;
                reference[index + 1] = circle ? 110 : 200;
                reference[index + 2] = circle ? 210 : 150;
            }
        }
        struct Cancellation {
            sd_ctx_t* context;
            bool requested = false;
        } cancellation{context.get()};
        auto run = [&](const std::string& name, bool image, float strength, bool cancel = false) {
            sd_img_gen_params_t params;
            sd_img_gen_params_init(&params);
            params.prompt = "a blue ceramic teapot on a wooden table, still life illustration, warm lighting";
            params.negative_prompt = "blurry, text, watermark";
            params.seed = 1234;
            params.width = 256;
            params.height = 256;
            params.batch_count = 1;
            params.sample_params.sample_steps = 8;
            params.sample_params.sample_method = sd_get_default_sample_method(context.get());
            params.sample_params.scheduler = sd_get_default_scheduler(context.get(), params.sample_params.sample_method);
            params.sample_params.guidance.txt_cfg = 4.0f;
            if (image) params.ip_adapter.image = sd_image_t{128, 128, 3, reference.data()};
            params.ip_adapter.strength = strength;
            params.ip_adapter.start_percent = 0;
            params.ip_adapter.end_percent = 100;
            sd_image_t* images = nullptr;
            int count = 0;
            cancellation.requested = false;
            if (cancel) {
                sd_set_sample_progress_callback([](int step, int, float, void* data) {
                    auto* state = static_cast<Cancellation*>(data);
                    if (step >= 1 && !state->requested) {
                        state->requested = true;
                        sd_cancel_generation(state->context, SD_CANCEL_ALL);
                    }
                }, &cancellation);
            }
            const bool success = generate_image(context.get(), &params, &images, &count);
            sd_set_sample_progress_callback(nullptr, nullptr);
            auto cleanup = [&]() {
                for (int i = 0; images != nullptr && i < count; ++i) std::free(images[i].data);
                std::free(images);
            };
            if (cancel) {
                cleanup();
                if (!cancellation.requested || success || count != 0) {
                    throw std::runtime_error("Cancellation did not stop native generation");
                }
                return std::vector<uint8_t>{};
            }
            if (!success || count != 1 || images == nullptr || images[0].data == nullptr) {
                cleanup();
                throw std::runtime_error("Generation failed: " + name);
            }
            std::vector<uint8_t> result(images[0].data, images[0].data + images[0].width * images[0].height * images[0].channel);
            const auto width = images[0].width, height = images[0].height, channels = images[0].channel;
            cleanup();
            if (channels != 3 || width != 256 || height != 256) throw std::runtime_error("Unexpected image geometry");
            const auto range = std::minmax_element(result.begin(), result.end());
            if (*range.second - *range.first < 20) throw std::runtime_error("Degenerate output image");
            std::ofstream out(std::string(argv[6]) + "-" + name + ".ppm", std::ios::binary);
            out << "P6\n" << width << " " << height << "\n255\n";
            if (!out.write(reinterpret_cast<const char*>(result.data()), result.size())) throw std::runtime_error("Cannot write output");
            return result;
        };
        const auto baseline = run("disabled-before", false, 0);
        const auto conditioned = run("enabled", true, 0.35f);
        run("cancelled", true, 0.35f, true);
        const auto restored = run("disabled-after", false, 0);
        const auto zero = run("zero-strength", true, 0);
        if (baseline != restored || baseline != zero) throw std::runtime_error("Image conditioning leaked into disabled generation");
        if (baseline == conditioned) throw std::runtime_error("IP-Adapter did not affect generation");
        std::cout << "PASS: real native Anima generation; enabled image differs; cancellation stops sampling; disabled and zero strength are byte-identical to baseline\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
