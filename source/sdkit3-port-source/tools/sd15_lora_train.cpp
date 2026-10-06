#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <set>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "conditioning/conditioner.hpp"
#include "core/rng.hpp"
#include "model/diffusion/sd15_lora_trainer.hpp"
#include "model/vae/auto_encoder_kl.hpp"
#include "model_manager.h"

namespace fs = std::filesystem;

static std::string timestamp_utc() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

static void log_event_prefix(const char* event) {
    std::cout << "{\"timestamp\":\"" << timestamp_utc()
              << "\",\"event\":\"" << event << "\"";
}

struct Sample {
    std::vector<float> latent;
    std::vector<float> context;
};

static std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream data;
    data << in.rdbuf();
    std::string value = data.str();
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

static std::vector<fs::path> images_in(const fs::path& root) {
    static const std::set<std::string> extensions{ ".png", ".jpg", ".jpeg", ".webp", ".bmp" };
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && extensions.count(entry.path().extension().string())) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

static sd::Tensor<float> load_rgb_square(const fs::path& path, int resolution) {
    int width = 0, height = 0, channels = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
        stbi_load(path.c_str(), &width, &height, &channels, 3), stbi_image_free);
    if (!pixels || width < 1 || height < 1) throw std::runtime_error("could not decode image: " + path.string());
    auto image = sd::zeros<float>({resolution, resolution, 3, 1});
    // Square resize is deterministic and keeps this first native path independent
    // of Python image libraries; training augmentations can be layered later.
    for (int y = 0; y < resolution; ++y) {
        const int sy = std::min(height - 1, static_cast<int>((static_cast<int64_t>(y) * height) / resolution));
        for (int x = 0; x < resolution; ++x) {
            const int sx = std::min(width - 1, static_cast<int>((static_cast<int64_t>(x) * width) / resolution));
            for (int c = 0; c < 3; ++c) image.index(x, y, c, 0) = pixels.get()[(sy * width + sx) * 3 + c] / 255.0f;
        }
    }
    return image;
}

#include "sd15_lora_export.hpp"
#include "sd15_training_schedule.hpp"
#include "sd15_epoch_sampling.hpp"

static ggml_backend_t backend_by_name(const std::string& name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto device = ggml_backend_dev_get(i);
        if (name == ggml_backend_dev_name(device)) return ggml_backend_dev_init(device, nullptr);
    }
    return nullptr;
}

struct BackendCleanup {
    ggml_backend_t backend;
    ~BackendCleanup() { if (backend) ggml_backend_free(backend); }
};

struct RunnerCleanup {
    ModelManager& manager;
    ggml_backend_t backend;
    std::shared_ptr<SD15LoraTrainingRunner>& unet;
    std::shared_ptr<FrozenCLIPEmbedderWithCustomWords>& clip;
    std::shared_ptr<AutoEncoderKL>& vae;
    bool registered_unet = false;
    bool registered_clip = false;
    bool registered_vae = false;

    ~RunnerCleanup() {
        if (unet) unet->runner_done();
        if (clip) clip->runner_done();
        if (vae) vae->runner_done();
        bool released = true;
        if (registered_unet) released = manager.unregister_param_tensors("SD15 trainer UNet") && released;
        if (registered_clip) released = manager.unregister_param_tensors("SD15 trainer CLIP") && released;
        if (registered_vae) released = manager.unregister_param_tensors("SD15 trainer VAE") && released;
        // Unregistration can be deferred when a backend graph exits abnormally.
        // Force the remaining manager-owned buffers out before runner contexts.
        if (!released) manager.release_registered_storage();
        unet.reset();
        clip.reset();
        vae.reset();
    }
};

int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];
    auto need = [&](const char* key) -> std::string {
        auto it = args.find(key);
        if (it == args.end() || it->second.empty()) throw std::runtime_error(std::string("missing argument ") + key);
        return it->second;
    };
    auto integer_option = [&](const char* key, int fallback) {
        auto it = args.find(key);
        if (it == args.end()) return fallback;
        size_t used = 0;
        const int value = std::stoi(it->second, &used);
        if (used != it->second.size()) throw std::invalid_argument(std::string(key) + " must be an integer");
        return value;
    };
    try {
        const fs::path checkpoint = need("--model");
        const fs::path dataset = need("--dataset");
        const fs::path output = need("--output");
        const std::string trigger = args.count("--trigger") ? args["--trigger"] : "";
        const std::string backend_name = args.count("--device") ? args["--device"] : "SYCL1";
        const int resolution = args.count("--resolution") ? std::stoi(args["--resolution"]) : 256;
        const auto image_paths = images_in(dataset);
        const SD15EpochSchedule epoch_schedule(image_paths.size(), integer_option("--dataset-repeats", 1),
                                               integer_option("--epochs", 0), integer_option("--steps", 1000));
        const int steps = epoch_schedule.steps;
        const int save_every = args.count("--save-every") ? std::stoi(args["--save-every"]) : 100;
        const std::string save_step_zero_arg = args.count("--save-step-zero") ? args["--save-step-zero"] : "0";
        if (save_step_zero_arg != "0" && save_step_zero_arg != "1")
            throw std::runtime_error("--save-step-zero must be 0 or 1");
        const bool save_step_zero = save_step_zero_arg == "1";
        const int rank = args.count("--rank") ? std::stoi(args["--rank"]) : 16;
        size_t alpha_used = 0;
        const float network_alpha = args.count("--network-alpha") ? std::stof(args["--network-alpha"], &alpha_used) : static_cast<float>(rank);
        if (args.count("--network-alpha") && alpha_used != args["--network-alpha"].size())
            throw std::runtime_error("LoRA alpha must be a number");
        if (!std::isfinite(network_alpha) || network_alpha <= 0.0f || network_alpha > 128.0f)
            throw std::runtime_error("LoRA alpha must be finite, positive and at most 128");
        const std::string lr_scheduler = args.count("--lr-scheduler") ? args["--lr-scheduler"] : "constant";
        const int warmup_steps = integer_option("--lr-warmup-steps", 0);
        const int cycles = integer_option("--lr-scheduler-num-cycles", 1);
        const SD15TrainingSchedule schedule(lr_scheduler, steps, warmup_steps, cycles);
        const int threads = args.count("--threads") ? std::stoi(args["--threads"]) : 10;
        const float learning_rate = args.count("--learning-rate") ? std::stof(args["--learning-rate"]) : 1e-4f;
        const float text_encoder_learning_rate = args.count("--text-encoder-learning-rate")
            ? std::stof(args["--text-encoder-learning-rate"]) : 0.0f;
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0f || learning_rate > 0.1f ||
            !std::isfinite(text_encoder_learning_rate) || text_encoder_learning_rate < 0.0f || text_encoder_learning_rate > 0.1f)
            throw std::runtime_error("invalid UNet or text encoder learning rate");
        const uint32_t seed = args.count("--seed") ? static_cast<uint32_t>(std::stoul(args["--seed"])) : 42;
        if (resolution < 128 || resolution > 1024 || resolution % 64 || steps < 1 || rank < 1 || rank > 128 || save_every < 1)
            throw std::runtime_error("invalid resolution, steps, rank, or save interval");

        ggml_backend_load_all();
        ggml_backend_t backend = backend_by_name(backend_name);
        if (!backend) throw std::runtime_error("GGML backend not found: " + backend_name);
        BackendCleanup backend_cleanup{backend};
        log_event_prefix("startup");
        std::cout << ",\"build_date\":\"" << __DATE__ << " " << __TIME__
                  << "\",\"checkpoint\":\"" << json_escape(fs::absolute(checkpoint).string())
                  << "\",\"dataset\":\"" << json_escape(fs::absolute(dataset).string())
                  << "\",\"output\":\"" << json_escape(fs::absolute(output).string())
                  << "\",\"device_requested\":\"" << json_escape(backend_name)
                  << "\",\"device_name\":\"" << json_escape(ggml_backend_dev_name(ggml_backend_get_device(backend)))
                  << "\",\"resolution\":" << resolution << ",\"steps\":" << steps
                  << ",\"rank\":" << rank << ",\"learning_rate\":" << learning_rate
                  << ",\"network_alpha\":" << network_alpha
                  << ",\"lr_scheduler\":\"" << lr_scheduler << "\",\"lr_warmup_steps\":" << warmup_steps
                  << ",\"lr_scheduler_num_cycles\":" << cycles
                  << ",\"text_encoder_learning_rate\":" << text_encoder_learning_rate
                  << ",\"threads\":" << threads << ",\"images\":" << image_paths.size()
                  << ",\"steps_per_epoch\":" << epoch_schedule.steps_per_epoch
                  << ",\"trigger\":\"" << json_escape(trigger) << "\"}" << std::endl;
        std::cout << "Trainer build: " << __DATE__ << " " << __TIME__
                  << " | device: " << ggml_backend_dev_name(ggml_backend_get_device(backend))
                  << " | checkpoint: " << fs::absolute(checkpoint).string() << std::endl;
        // Keep runners alive across ModelManager destruction on error, so it
        // can release parameter buffers before their tensor contexts disappear.
        std::shared_ptr<SD15LoraTrainingRunner> unet;
        std::shared_ptr<FrozenCLIPEmbedderWithCustomWords> clip;
        std::shared_ptr<AutoEncoderKL> vae;
        ModelManager manager;
        manager.set_n_threads(threads);
        RunnerCleanup cleanup{manager, backend, unet, clip, vae};
        auto& loader = manager.loader();
        if (!loader.init_from_file_and_convert_name(checkpoint.string(), "", VERSION_SD1))
            throw std::runtime_error("could not read SD 1.5 checkpoint metadata");
        loader.process_model_files(false, false);
        std::shared_ptr<RunnerWeightManager> weight_manager(&manager, [](RunnerWeightManager*) {});
        const auto& tensor_map = loader.get_tensor_storage_map();

        unet = std::make_shared<SD15LoraTrainingRunner>(backend, tensor_map, "model.diffusion_model", rank,
                                                        network_alpha, seed, weight_manager, text_encoder_learning_rate);
        clip = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(backend, tensor_map,
            std::map<std::string, std::string>{}, VERSION_SD1, weight_manager);
        clip->truncate_long_prompts = true;
        vae = std::make_shared<AutoEncoderKL>(backend, tensor_map, "first_stage_model", false, false,
                                              VERSION_SD1, weight_manager);
        cleanup.registered_unet = manager.register_runner_params("SD15 trainer UNet", *unet, "model.diffusion_model", ModelManager::ResidencyMode::ParamBackend, backend, backend);
        if (text_encoder_learning_rate == 0.0f)
            cleanup.registered_clip = manager.register_runner_params("SD15 trainer CLIP", *clip, ModelManager::ResidencyMode::ParamBackend, backend, backend);
        cleanup.registered_vae = manager.register_runner_params("SD15 trainer VAE", *vae, ModelManager::ResidencyMode::ParamBackend, backend, backend);
        if (!cleanup.registered_unet || (text_encoder_learning_rate == 0.0f && !cleanup.registered_clip) || !cleanup.registered_vae ||
            !manager.validate_registered_tensors()) throw std::runtime_error("failed to register checkpoint model tensors");

        const int latent_size = resolution / 8;
        const auto& graph = unet->prepare_training_graph(latent_size, latent_size, 1, 77);
        if (!unet->initialize_optimizer(learning_rate, 0.0f, 1)) throw std::runtime_error("could not initialize LoRA optimizer");
        auto rng = std::make_shared<STDDefaultRNG>();
        rng->manual_seed(seed);
        std::vector<Sample> samples;
        std::string sample_prompt;
        samples.reserve(image_paths.size());
        for (size_t i = 0; i < image_paths.size(); ++i) {
            fs::path caption_path = image_paths[i];
            caption_path.replace_extension(".txt");
            std::string caption = read_text(caption_path);
            const bool caption_file_found = !caption.empty();
            if (!trigger.empty() && caption.find(trigger) == std::string::npos) caption += (caption.empty() ? "" : ", ") + trigger;
            if (caption.empty()) throw std::runtime_error("empty caption: " + caption_path.string());
            if (i == 0) sample_prompt = caption;
            auto image = load_rgb_square(image_paths[i], resolution);
            auto posterior = vae->encode(threads, image, {});
            auto latents = vae->vae_output_to_latents(posterior, rng);
            // SD 1.x diffusion operates on scaled VAE latents (0.18215), not
            // the raw posterior sample. Match the conversion used at inference.
            latents = vae->vae_to_diffusion_latents(latents);
            std::vector<float> context;
            if (text_encoder_learning_rate > 0.0f) {
                // Captions are literal training text, not inference prompt markup.
                // Cache only token IDs: CLIP must run again after every update.
                auto tokens = clip->tokenizer.tokenize(caption, nullptr, true, 77, 77, false);
                context.assign(tokens.begin(), tokens.end());
            } else {
                ConditionerParams cp; cp.text = caption; cp.width = resolution; cp.height = resolution;
                context = clip->get_learned_condition(threads, cp).c_crossattn.values();
            }
            const size_t expected_context = text_encoder_learning_rate > 0.0f ? 77u : 77u * 768u;
            if (latents.empty() || latents.values().size() != static_cast<size_t>(graph.latent_features) ||
                context.size() != expected_context) throw std::runtime_error("VAE/CLIP preprocessing failed for " + image_paths[i].string());
            const auto& latent_values = latents.values();
            const auto [latent_min, latent_max] = std::minmax_element(latent_values.begin(), latent_values.end());
            const double latent_mean = std::accumulate(latent_values.begin(), latent_values.end(), 0.0) /
                                       static_cast<double>(latent_values.size());
            samples.push_back({latent_values, std::move(context)});
            log_event_prefix("dataset_image");
            std::cout << ",\"done\":" << (i + 1) << ",\"total\":" << image_paths.size()
                      << ",\"file\":\"" << json_escape(image_paths[i].filename().string())
                      << "\",\"caption_file_found\":" << (caption_file_found ? "true" : "false")
                      << ",\"latent_min\":" << *latent_min << ",\"latent_max\":" << *latent_max
                      << ",\"latent_mean\":" << latent_mean << "}" << std::endl;
        }

        const auto trainable = unet->trainable_parameter_map();
        size_t trainable_values = 0;
        size_t text_encoder_tensors = 0;
        for (const auto& item : trainable) trainable_values += static_cast<size_t>(ggml_nelements(item.second));
        for (const auto& item : trainable) if (starts_with(item.first, "lora.cond_stage_model.")) ++text_encoder_tensors;
        if (text_encoder_learning_rate > 0.0f && text_encoder_tensors != 96)
            throw std::runtime_error("expected 96 CLIP attention LoRA tensors");
        log_event_prefix("optimizer_ready");
        std::cout << ",\"trainable_tensors\":" << trainable.size()
                  << ",\"trainable_values\":" << trainable_values
                  << ",\"text_encoder_tensors\":" << text_encoder_tensors << "}" << std::endl;

        std::mt19937 random(seed);
        std::vector<size_t> sample_order(samples.size());
        for (size_t i = 0; i < sample_order.size(); ++i) sample_order[i] = i;
        std::uniform_int_distribution<int> choose_t(0, 999);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> packed(static_cast<size_t>(graph.packed_features));
        std::vector<float> target(static_cast<size_t>(graph.latent_features));
        std::array<double, 1000> alpha{};
        double product = 1.0;
        for (int t = 0; t < 1000; ++t) {
            const double f = static_cast<double>(t) / 999.0;
            const double beta = std::pow(std::sqrt(0.00085) + f * (std::sqrt(0.012) - std::sqrt(0.00085)), 2.0);
            product *= 1.0 - beta;
            alpha[static_cast<size_t>(t)] = product;
        }
        log_event_prefix("noise_schedule");
        std::cout << ",\"type\":\"scaled_linear_sd15\",\"beta_start\":0.00085"
                  << ",\"beta_end\":0.012,\"alpha_cumprod_last\":" << alpha[999] << "}" << std::endl;
        std::vector<float> negative_context;
        if (text_encoder_learning_rate > 0.0f) {
            auto tokens = clip->tokenizer.tokenize("", nullptr, true, 77, 77, false);
            negative_context.assign(tokens.begin(), tokens.end());
        } else {
            ConditionerParams cp; cp.text = ""; cp.width = resolution; cp.height = resolution;
            negative_context = clip->get_learned_condition(threads, cp).c_crossattn.values();
        }
        if (negative_context.size() != samples.front().context.size())
            throw std::runtime_error("invalid unconditional sampling context");
        auto sample_epoch = [&](int step) {
            const int epoch = step / epoch_schedule.steps_per_epoch;
            log_event_prefix("sample_start");
            std::cout << ",\"step\":" << step << ",\"epoch\":" << epoch
                      << ",\"prompt\":\"" << json_escape(sample_prompt) << "\",\"seed\":" << seed
                      << ",\"sampler\":\"ddim\",\"sampling_steps\":25,\"guidance\":7.5}" << std::endl;
            auto values = sd15_ddim_sample(static_cast<size_t>(graph.latent_features), seed, 25, 7.5f, alpha,
                [&](const std::vector<float>& latent, int timestep, bool positive) {
                    std::vector<float> inputs(static_cast<size_t>(graph.packed_features));
                    std::copy(latent.begin(), latent.end(), inputs.begin());
                    inputs[latent.size()] = static_cast<float>(timestep);
                    const auto& context = positive ? samples.front().context : negative_context;
                    std::copy(context.begin(), context.end(), inputs.begin() + latent.size() + 1);
                    return unet->predict_noise(inputs);
                });
            auto latent = sd::Tensor<float>::from_vector(values);
            latent.reshape_({latent_size, latent_size, 4, 1});
            auto image = vae->decode(threads, vae->diffusion_to_vae_latents(latent), {});
            if (image.empty() || image.values().size() != static_cast<size_t>(resolution * resolution * 3))
                throw std::runtime_error("epoch sample VAE decode failed");
            std::vector<unsigned char> pixels(static_cast<size_t>(resolution * resolution * 3));
            for (int y = 0; y < resolution; ++y) for (int x = 0; x < resolution; ++x) for (int c = 0; c < 3; ++c) {
                const float value = image.index(x, y, c, 0);
                if (!std::isfinite(value)) throw std::runtime_error("non-finite epoch sample pixel");
                pixels[(y * resolution + x) * 3 + c] = static_cast<unsigned char>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
            }
            const auto directory = output.parent_path() / "samples";
            fs::create_directories(directory);
            const auto path = directory / (output.stem().string() + "-epoch-" + std::to_string(epoch) + ".png");
            const auto temporary = path.string() + ".tmp";
            if (!stbi_write_png(temporary.c_str(), resolution, resolution, 3, pixels.data(), resolution * 3))
                throw std::runtime_error("could not write epoch sample");
            fs::rename(temporary, path);
            std::ofstream caption(path.parent_path() / (path.stem().string() + ".txt"));
            caption << sample_prompt << '\n';
            caption.close();
            if (!caption) throw std::runtime_error("could not write epoch sample caption");
            log_event_prefix("sample_saved");
            std::cout << ",\"step\":" << step << ",\"epoch\":" << epoch
                      << ",\"path\":\"" << json_escape(fs::absolute(path).string()) << "\"}" << std::endl;
        };
        // Optional diagnostic snapshot: no optimizer update has happened yet.
        if (save_step_zero) {
            const fs::path initial_path = output.parent_path() /
                (output.stem().string() + "-0" + output.extension().string());
            if (!save_lora(initial_path, unet->trainable_parameter_map(), trigger, 0, rank, network_alpha))
                throw std::runtime_error("failed to save step-zero LoRA checkpoint");
            log_event_prefix("saved");
            std::cout << ",\"path\":\"" << json_escape(fs::absolute(initial_path).string())
                      << "\",\"step\":0}" << std::endl;
        }
        const auto training_started = std::chrono::steady_clock::now();
        for (int step = 1; step <= steps; ++step) {
            const float lr_factor = schedule.factor(step - 1);
            unet->set_learning_rate(learning_rate * lr_factor);
            const size_t position = static_cast<size_t>(step - 1) % sample_order.size();
            if (position == 0) std::shuffle(sample_order.begin(), sample_order.end(), random);
            const Sample& sample = samples[sample_order[position]];
            const int timestep = choose_t(random);
            const float signal = static_cast<float>(std::sqrt(alpha[static_cast<size_t>(timestep)]));
            const float noise_scale = static_cast<float>(std::sqrt(1.0 - alpha[static_cast<size_t>(timestep)]));
            for (size_t i = 0; i < sample.latent.size(); ++i) {
                target[i] = normal(random);
                packed[i] = signal * sample.latent[i] + noise_scale * target[i];
            }
            packed[sample.latent.size()] = static_cast<float>(timestep);
            std::copy(sample.context.begin(), sample.context.end(), packed.begin() + sample.latent.size() + 1);
            float loss = 0.0f;
            log_event_prefix("step_start");
            std::cout << ",\"step\":" << step << ",\"timestep\":" << timestep << "}" << std::endl;
            if (!unet->train_batch(packed.data(), target.data(), &loss) || !std::isfinite(loss)) throw std::runtime_error("optimizer step failed");
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - training_started).count();
            log_event_prefix("progress");
            std::cout << ",\"step\":" << step << ",\"total\":" << steps << ",\"loss\":" << loss
                      << ",\"learning_rate\":" << learning_rate * lr_factor
                      << ",\"text_encoder_learning_rate\":" << text_encoder_learning_rate * lr_factor
                      << ",\"step_seconds\":" << elapsed / step
                      << ",\"elapsed_seconds\":" << elapsed
                      << ",\"eta_seconds\":" << (elapsed / step) * (steps - step) << "}" << std::endl;
            if (step % save_every == 0 || step == steps || epoch_schedule.epoch_end(step)) {
                fs::path save_path = output;
                if (step != steps) save_path = output.parent_path() / (output.stem().string() + "-" + std::to_string(step) + output.extension().string());
                if (!save_lora(save_path, unet->trainable_parameter_map(), trigger, step, rank, network_alpha)) throw std::runtime_error("failed to save LoRA checkpoint");
                log_event_prefix("saved");
                std::cout << ",\"path\":\"" << json_escape(fs::absolute(save_path).string())
                          << "\",\"step\":" << step << "}" << std::endl;
            }
            if (epoch_schedule.epoch_end(step)) {
                log_event_prefix("epoch_end");
                std::cout << ",\"step\":" << step << ",\"epoch\":" << step / epoch_schedule.steps_per_epoch << "}" << std::endl;
                sample_epoch(step);
            }
        }
        log_event_prefix("completed");
        std::cout << ",\"steps\":" << steps << "}" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "{\"timestamp\":\"" << timestamp_utc() << "\",\"event\":\"failed\",\"error\":\"" << json_escape(e.what()) << "\"}" << std::endl;
        return 1;
    }
}
