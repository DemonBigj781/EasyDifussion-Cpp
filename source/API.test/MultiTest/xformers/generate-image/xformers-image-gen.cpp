#include "xformers-image-gen.hpp"

#include "stable-diffusion.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace api_test::xformers_image {
namespace {

struct Options {
    std::string prompt;
    std::string negative_prompt;
    std::string model_path;
    std::string vae_path;
    std::string device;
    std::string output_path;
    int steps = 20;
    float cfg = 7.0f;
};

struct ContextDeleter {
    void operator()(sd_ctx_t* context) const noexcept {
        if (context != nullptr) {
            free_sd_ctx(context);
        }
    }
};

using Context = std::unique_ptr<sd_ctx_t, ContextDeleter>;

void print_usage(const char* executable, const BackendConfiguration& backend) {
    std::cerr
        << "usage: " << executable << " -m MODEL -p PROMPT [options]\n\n"
        << "Generate one 512x512 PPM image with xFormers-compatible attention.\n"
        << "Sampling is fixed to DDIM trailing with the Simple scheduler.\n\n"
        << "  -p TEXT   positive prompt (required)\n"
        << "  -n TEXT   negative prompt (default: empty)\n"
        << "  -s N      DDIM steps (default: 20)\n"
        << "  -m PATH   checkpoint/model path (required)\n"
        << "  -v PATH   optional external VAE path\n"
        << "  -d VALUE  device id, UUID, or name (default: "
        << backend.default_device << ")\n"
        << "  -c VALUE  CFG scale (default: 7.0)\n"
        << "  -o PATH   output PPM path (default: "
        << backend.default_output << ")\n"
        << "  -h        show this help\n";
}

bool parse_integer(std::string_view text, int& value) {
    int parsed = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end) {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_float(std::string_view text, float& value) {
    try {
        std::size_t consumed = 0;
        const float parsed = std::stof(std::string(text), &consumed);
        if (consumed != text.size() || !std::isfinite(parsed)) {
            return false;
        }
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_options(
        int argc,
        char** argv,
        const BackendConfiguration& backend,
        Options& options,
        bool& help_requested) {
    options.device = backend.default_device;
    options.output_path = backend.default_output;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0], backend);
            help_requested = true;
            return true;
        }
        if (argument != "-p" && argument != "-n" && argument != "-s" &&
            argument != "-m" && argument != "-v" && argument != "-d" &&
            argument != "-c" && argument != "-o") {
            std::cerr << "unknown argument: " << argument << '\n';
            print_usage(argv[0], backend);
            return false;
        }
        if (++index >= argc) {
            std::cerr << argument << " requires a value\n";
            return false;
        }

        const std::string value(argv[index]);
        if (argument == "-p") options.prompt = value;
        if (argument == "-n") options.negative_prompt = value;
        if (argument == "-m") options.model_path = value;
        if (argument == "-v") options.vae_path = value;
        if (argument == "-d") options.device = value;
        if (argument == "-o") options.output_path = value;
        if (argument == "-s" && !parse_integer(value, options.steps)) {
            std::cerr << "invalid step count: " << value << '\n';
            return false;
        }
        if (argument == "-c" && !parse_float(value, options.cfg)) {
            std::cerr << "invalid CFG value: " << value << '\n';
            return false;
        }
    }

    if (options.prompt.empty()) {
        std::cerr << "-p requires a non-empty positive prompt\n";
        return false;
    }
    if (options.model_path.empty()) {
        std::cerr << "-m is required\n";
        return false;
    }
    if (options.steps <= 0 || options.steps > 1000) {
        std::cerr << "-s must be between 1 and 1000\n";
        return false;
    }
    if (options.cfg < 0.0f) {
        std::cerr << "-c must be non-negative\n";
        return false;
    }
    if (!std::filesystem::is_regular_file(options.model_path)) {
        std::cerr << "model is not a regular file: " << options.model_path << '\n';
        return false;
    }
    if (!options.vae_path.empty() &&
        !std::filesystem::is_regular_file(options.vae_path)) {
        std::cerr << "VAE is not a regular file: " << options.vae_path << '\n';
        return false;
    }
    return true;
}

bool enable_cuda_xformers() {
#ifdef _WIN32
    return _putenv_s("SD_CUDA_XFORMERS", "1") == 0;
#else
    return setenv("SD_CUDA_XFORMERS", "1", 1) == 0;
#endif
}

bool write_ppm(const std::string& path, const sd_image_t& image) {
    if (image.data == nullptr || image.width == 0 || image.height == 0 ||
        image.channel == 0) {
        return false;
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }
    output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    const std::size_t pixels =
        static_cast<std::size_t>(image.width) * image.height;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        const std::size_t base = pixel * image.channel;
        const char red = static_cast<char>(image.data[base]);
        const char green = static_cast<char>(
            image.channel > 1 ? image.data[base + 1] : image.data[base]);
        const char blue = static_cast<char>(
            image.channel > 2 ? image.data[base + 2] : image.data[base]);
        output.write(&red, 1);
        output.write(&green, 1);
        output.write(&blue, 1);
    }
    return output.good();
}

} // namespace

int run(int argc, char** argv, const BackendConfiguration& backend) {
    if (backend.label == nullptr || backend.default_device == nullptr ||
        backend.default_output == nullptr || backend.resolve_device == nullptr ||
        (backend.enable_cuda_xformers && backend.xformers_launch_count == nullptr)) {
        std::cerr << "invalid image-generator backend configuration\n";
        return EXIT_FAILURE;
    }

    Options options;
    bool help_requested = false;
    if (!parse_options(argc, argv, backend, options, help_requested)) {
        return EXIT_FAILURE;
    }
    if (help_requested) return EXIT_SUCCESS;

    std::string backend_name;
    std::string device_description;
    std::string device_error;
    if (!backend.resolve_device(
            options.device, backend_name, device_description, device_error)) {
        std::cerr << "device selection failed: " << device_error << '\n';
        return EXIT_FAILURE;
    }
    if (backend.enable_cuda_xformers && !enable_cuda_xformers()) {
        std::cerr << "failed to enable SD_CUDA_XFORMERS\n";
        return EXIT_FAILURE;
    }

    sd_ctx_params_t context_parameters;
    sd_ctx_params_init(&context_parameters);
    context_parameters.model_path = options.model_path.c_str();
    if (!options.vae_path.empty()) {
        context_parameters.vae_path = options.vae_path.c_str();
    }
    context_parameters.n_threads = static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency()));
    context_parameters.enable_mmap = true;
    context_parameters.flash_attn = true;
    context_parameters.diffusion_flash_attn = true;
    context_parameters.backend = backend_name.c_str();
    context_parameters.params_backend = backend_name.c_str();

    std::cout << "backend: " << backend.label << " / "
              << device_description << '\n'
              << "model: " << options.model_path << '\n'
              << "sampler: ddim_trailing\n"
              << "scheduler: simple\n"
              << "steps: " << options.steps << '\n'
              << "CFG: " << options.cfg << '\n';
    if (backend.enable_cuda_xformers) {
        std::cout << "CUDA xFormers selection: requested by SD_CUDA_XFORMERS=1; "
                     "a native launch is required for this test to pass\n";
    } else {
        std::cout << "CPU memory-efficient attention requested\n";
    }

    const auto load_start = std::chrono::steady_clock::now();
    Context context(new_sd_ctx(&context_parameters));
    const auto load_stop = std::chrono::steady_clock::now();
    if (!context) {
        std::cerr << "failed to load the Stable Diffusion context\n";
        return EXIT_FAILURE;
    }
    if (!sd_ctx_supports_image_generation(context.get())) {
        std::cerr << "the selected model does not support image generation\n";
        return EXIT_FAILURE;
    }

    sd_img_gen_params_t generation;
    sd_img_gen_params_init(&generation);
    generation.prompt = options.prompt.c_str();
    generation.negative_prompt = options.negative_prompt.c_str();
    generation.width = 512;
    generation.height = 512;
    generation.batch_count = 1;
    generation.seed = 42;
    generation.sample_params.sample_method = DDIM_TRAILING_SAMPLE_METHOD;
    generation.sample_params.scheduler = SIMPLE_SCHEDULER;
    generation.sample_params.sample_steps = options.steps;
    generation.sample_params.guidance.txt_cfg = options.cfg;

    const std::uint64_t launches_before = backend.xformers_launch_count == nullptr
        ? 0
        : backend.xformers_launch_count();
    const auto generation_start = std::chrono::steady_clock::now();
    sd_image_t* images = generate_image(context.get(), &generation);
    const auto generation_stop = std::chrono::steady_clock::now();
    if (images == nullptr) {
        std::cerr << "DDIM image generation failed\n";
        return EXIT_FAILURE;
    }

    const std::uint64_t launches_after = backend.xformers_launch_count == nullptr
        ? 0
        : backend.xformers_launch_count();
    if (backend.enable_cuda_xformers && launches_after <= launches_before) {
        free_sd_images(images, 1);
        std::cerr << "CUDA generation completed without launching the native "
                     "xFormers kernel\n";
        return EXIT_FAILURE;
    }

    const bool saved = write_ppm(options.output_path, images[0]);
    free_sd_images(images, 1);
    if (!saved) {
        std::cerr << "failed to save output image: "
                  << options.output_path << '\n';
        return EXIT_FAILURE;
    }

    const double load_seconds =
        std::chrono::duration<double>(load_stop - load_start).count();
    const double generation_seconds =
        std::chrono::duration<double>(generation_stop - generation_start).count();
    std::cout << "image: " << options.output_path << '\n'
              << "model load: " << load_seconds << " s\n"
              << "generation: " << generation_seconds << " s\n";
    if (backend.enable_cuda_xformers) {
        std::cout << "native CUDA xFormers launches: "
                  << (launches_after - launches_before) << '\n';
    }
    return EXIT_SUCCESS;
}

} // namespace api_test::xformers_image
