#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#else
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "device_cli.h"
#include "logging.h"
#include "model_manager.h"
#include "server.h"
#include "stable-diffusion.h"

std::unique_ptr<Server> g_server;
std::atomic<bool> g_should_exit(false);
std::unique_ptr<std::thread> g_watchdog_thread;

void parent_watchdog(int parent_pid) {
    LOG_INFO("Starting parent process watchdog for PID %d", parent_pid);

#ifdef _WIN32
    // Windows: Use timed wait so we can check for exit signal
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, parent_pid);
    if (process == NULL) {
        LOG_ERROR("Failed to open parent process (PID %d). Cannot monitor.", parent_pid);
        return;
    }

    // Wait with timeout so we can check g_should_exit periodically
    while (!g_should_exit.load()) {
        DWORD result = WaitForSingleObject(process, 1000);  // 1 second timeout
        if (result == WAIT_OBJECT_0) {
            // Parent process exited
            LOG_WARNING("Parent process (PID %d) is no longer running. Shutting down...", parent_pid);
            if (g_server) {
                g_server->stop();
            }
            g_should_exit.store(true);
            break;
        }
        // WAIT_TIMEOUT means parent is still alive, continue loop
    }
    CloseHandle(process);
#else
    // Unix/Linux/Mac: Poll with kill(pid, 0) check
    while (!g_should_exit.load()) {
        if (kill(parent_pid, 0) != 0) {
            LOG_WARNING("Parent process (PID %d) is no longer running. Shutting down...", parent_pid);
            if (g_server) {
                g_server->stop();
            }
            g_should_exit.store(true);
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
#endif

    LOG_INFO("Parent process watchdog stopped");
}

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_should_exit.store(true);
    if (g_server) {
        g_server->stop();
    }
}

struct CommandLineArgs {
    int port = 8188;
    std::string log_level = "info";
    std::string compute_backend = "auto";
    std::string pcie_devices;
    bool pcie_devices_set = false;
    bool list_devices = false;
    std::string list_devices_backend;
    int parent_pid = 0;
    std::string ckpt_dir;
    std::string vae_dir;
    std::string hypernetwork_dir;
    std::string gfpgan_models_path;
    std::string realesrgan_models_path;
    std::string lora_dir;
    std::string codeformer_models_path;
    std::string embeddings_dir;
    std::string controlnet_dir;
    std::string control_net_sd1_path;
    std::string control_net_sdxl_path;
    std::string text_encoder_dir;
    bool image_vae_on_cpu = false;
    bool no_half = false;
    bool no_half_vae = false;
    bool vae_tiling = false;
    std::string vae_tile_size;
    int vae_tiles = 32;
    int vae_tiled_overlap = 16;
    bool offload_to_cpu = false;
    bool mmap_weights = false;
    bool mmap_explicitly_disabled = false;
    bool keep_model_loaded = false;
    bool flash_attention = false;
    bool diffusion_fa = false;
    bool sage_attention = false;
    bool stream_layers = false;
    bool cuda_malloc = false;
    bool cuda_unified_memory = false;
    bool xformers_compat = false;
    bool split_attention = false;
    std::string attention_mode;
    bool control_net_cpu = false;
    bool image_clip_on_cpu = false;
    bool image_clip_vision_on_cpu = false;
    bool image_ip_adapter_on_cpu = false;
    bool video_clip_on_cpu = false;
    bool video_vae_on_cpu = false;
    bool video_offload_to_cpu = false;
    bool video_stream_layers = false;
    bool chroma_disable_dit_mask = false;
    std::string convert_model;
    std::string convert_output;
    std::string convert_type = "f16";
};

static std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static const char* device_type_name(sd_backend_device_type_t type) {
    switch (type) {
        case SD_BACKEND_DEVICE_TYPE_CPU:
            return "cpu";
        case SD_BACKEND_DEVICE_TYPE_GPU:
            return "gpu";
        case SD_BACKEND_DEVICE_TYPE_IGPU:
            return "integrated-gpu";
        case SD_BACKEND_DEVICE_TYPE_ACCELERATOR:
            return "accelerator";
        case SD_BACKEND_DEVICE_TYPE_META:
            return "meta";
    }
    return "unknown";
}

static std::vector<sd_backend_device_info_t> get_backend_devices() {
    std::vector<sd_backend_device_info_t> result;
    const size_t count = sd_get_backend_device_count();
    for (size_t i = 0; i < count; ++i) {
        sd_backend_device_info_t info{};
        if (sd_get_backend_device_info(i, &info)) {
            result.push_back(info);
        }
    }
    return result;
}

static std::vector<sdkit::PhysicalDevice> physical_inventory(
        const std::vector<sd_backend_device_info_t>& devices) {
    std::vector<sdkit::PhysicalDevice> result;
    for (const auto& info : devices) {
        result.push_back({lowercase(info.backend), info.name, info.device_id});
    }
    return result;
}

static bool print_backend_devices(const std::string& backend) {
    bool any = false;
    const auto devices = get_backend_devices();
    const auto inventory = physical_inventory(devices);
    std::cout << "driver\tselector\tPCIe address\ttype\tfree MiB\ttotal MiB\tvendor\tcapabilities\tdescription"
              << std::endl;
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto& info = devices[i];
        const auto& device = inventory[i];
        if (!backend.empty() && device.driver != backend) continue;
        any = true;
        const auto address = sdkit::selectable_pcie_address(device, inventory);
        const char* pcie = address.empty() ? (device.metadata_id.empty() ? "not reported" : "not selectable")
                                          : address.c_str();
        std::ostringstream capabilities;
        capabilities << "async=" << (info.async ? "yes" : "no")
                     << ",host-buffer=" << (info.host_buffer ? "yes" : "no")
                     << ",host-pointer=" << (info.buffer_from_host_ptr ? "yes" : "no")
                     << ",events=" << (info.events ? "yes" : "no");
        std::cout << (device.driver == "sycl" ? "oneapi" : device.driver) << '\t'
                  << info.name << '\t' << pcie << '\t'
                  << device_type_name(info.type) << '\t'
                  << (info.memory_free / (1024 * 1024)) << '\t'
                  << (info.memory_total / (1024 * 1024)) << '\t'
                  << info.vendor << '\t' << capabilities.str() << '\t' << info.description << std::endl;
    }
    return any || backend.empty();
}

static bool build_compute_backend_spec(const CommandLineArgs& args, std::string& spec, std::string& error) {
    const auto& backend = args.compute_backend;
    spec.clear();
    if (args.pcie_devices_set) {
        error = sdkit::resolve_pcie_devices(backend, args.pcie_devices,
                                           physical_inventory(get_backend_devices()), spec);
        return error.empty();
    }
    if (backend == "auto") return true;

    const auto devices = get_backend_devices();
    if (std::none_of(devices.begin(), devices.end(), [&](const auto& info) {
        return lowercase(info.backend) == backend || lowercase(info.name) == backend;
    })) {
        error = "Requested driver '" + backend + "' is not available in this build";
        return false;
    }
    spec = backend;
    return true;
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --port <port>                      Server port (default: 8188)" << std::endl;
    std::cerr << "  --log-level <level>                Log level: verbose, debug, info, warning, error (default: info)"
              << std::endl;
    std::cerr << "  --driver <name>                    auto (default), cpu, cuda, vulkan, oneapi, openvino*" << std::endl;
    std::cerr << "                                      *OpenVINO is not wired into the native runtime yet" << std::endl;
    std::cerr << "  --device <PCIE[,PCIE...]>           Physical PCIe address(es), e.g. 0000:01:00.0" << std::endl;
    std::cerr << "  --devices list [driver]            List selectors, memory, and capabilities, then exit" << std::endl;
    std::cerr << "  --parent-pid <pid>                 Parent process PID" << std::endl;
    std::cerr << "  --dir-stable-diffusion <dir>      Checkpoint models directory" << std::endl;
    std::cerr << "  --dir-vae <dir>                   VAE models directory" << std::endl;
    std::cerr << "  --dir-hypernetwork <dir>          Hypernetwork models directory" << std::endl;
    std::cerr << "  --dir-gfpgan <dir>                GFPGAN models directory" << std::endl;
    std::cerr << "  --dir-realesrgan <dir>            RealESRGAN models directory" << std::endl;
    std::cerr << "  --dir-lora <dir>                  LoRA models directory" << std::endl;
    std::cerr << "  --dir-codeformer <dir>            Codeformer models directory" << std::endl;
    std::cerr << "  --dir-embeddings <dir>            Embeddings directory" << std::endl;
    std::cerr << "  --dir-controlnet <dir>            ControlNet models directory" << std::endl;
    std::cerr << "  --control-net-sd1-path <path>      Uni-ControlNet weights for automatic SD1.x routing" << std::endl;
    std::cerr << "  --control-net-sdxl-path <path>     ControlNet Union weights for automatic SDXL routing" << std::endl;
    std::cerr << "  --dir-text-encoder <dir>          Text encoder models directory" << std::endl;
    std::cerr << "  --no-half                          Force all model weights to F32 (high memory use)" << std::endl;
    std::cerr << "  --no-half-vae                      Force VAE weights to F32" << std::endl;
    std::cerr << "  --vae-tiling                       Enable VAE tiling (default: false)" << std::endl;
    std::cerr << "  --vae-tile-size <size>             VAE tile size (in pixels), format [X]x[Y] (default: 256x256)"
              << std::endl;
    std::cerr << "  --vae-tiles <size>                 VAE latent tile size; 32 means 32x32 (default: 32)"
              << std::endl;
    std::cerr << "  --vae-tiled-overlap <pixels>       VAE latent tile overlap in pixels (default: 16)"
              << std::endl;
    std::cerr << "  --offload-to-cpu                   Offload parameters to CPU (default: false)" << std::endl;
    std::cerr << "  --mmap                             Memory-map weights instead of committing a RAM copy" << std::endl;
    std::cerr << "  --no-mmap                          Disable automatic mmap with --offload-to-cpu" << std::endl;
    std::cerr << "  --keep-model-loaded                Keep staged compute weights resident until model switch (default: false)"
              << std::endl;
    std::cerr << "  --attention <mode>                 Select one attention pipeline: split, flash," << std::endl;
    std::cerr << "                                     xformers, or sage (default: split)" << std::endl;
    std::cerr << "  --opt-split-attention              Legacy alias for --attention split" << std::endl;
    std::cerr << "  --diffusion-fa                     Legacy alias for --attention flash" << std::endl;
    std::cerr << "  --flash-attention                  Legacy alias for --attention flash" << std::endl;
    std::cerr << "  --xformers                         Legacy alias for --attention xformers" << std::endl;
    std::cerr << "  --sage-attention                   Legacy alias for --attention sage" << std::endl;
    std::cerr << "  --stream-layers                   Stream denoiser layers using reported VRAM (requires --offload-to-cpu)" << std::endl;
    std::cerr << "  --cuda-malloc                     Use the flushable legacy cudaMalloc pool instead of CUDA VMM" << std::endl;
    std::cerr << "  --cuda-unified-memory             Let CUDA spill allocations into system RAM (slow)" << std::endl;
    std::cerr << "  --video-offload-to-cpu             Offload native-video parameters to CPU (default: false)"
              << std::endl;
    std::cerr << "  --video-stream-layers              Stream video denoiser layers using reported VRAM (requires CPU offload)"
              << std::endl;
    std::cerr << "  --chroma-disable-dit-mask          Disable DiT mask for Chroma models (default: false)"
              << std::endl;
    std::cerr << "  --convert-model <path>             Convert a checkpoint, safetensors, or Diffusers model and exit"
              << std::endl;
    std::cerr << "  --convert-output <path>            GGUF output path used with --convert-model" << std::endl;
    std::cerr << "  --convert-type <type>              Conversion type (default: f16; e.g. f32, bf16, q8_0, q5_0, q4_0)"
              << std::endl;
}

// Legacy model-directory names are diagnosed, never accepted aliases.
static const char* retired_model_directory_flag(const std::string& arg) {
    const auto option = arg.substr(0, arg.find('='));
    if (option == "--ckpt-dir") return "--dir-stable-diffusion";
    if (option == "--vae-dir") return "--dir-vae";
    if (option == "--hypernetwork-dir") return "--dir-hypernetwork";
    if (option == "--gfpgan-models-path") return "--dir-gfpgan";
    if (option == "--realesrgan-models-path") return "--dir-realesrgan";
    if (option == "--lora-dir") return "--dir-lora";
    if (option == "--codeformer-models-path") return "--dir-codeformer";
    if (option == "--embeddings-dir") return "--dir-embeddings";
    if (option == "--controlnet-dir") return "--dir-controlnet";
    if (option == "--text-encoder-dir") return "--dir-text-encoder";
    return nullptr;
}

static const char* read_model_directory(const std::string& flag, int argc, char* argv[], int& index) {
    if (index + 1 >= argc || argv[index + 1][0] == '\0' ||
        std::string_view(argv[index + 1]).rfind("--", 0) == 0 ||
        std::string_view(argv[index + 1]) == "-h") {
        std::cerr << flag << " requires a directory argument" << std::endl;
        exit(1);
    }
    return argv[++index];
}

CommandLineArgs parse_args(int argc, char* argv[]) {
    CommandLineArgs args;
    bool discovery_selected = false;
    bool driver_selected = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        const auto option = arg.substr(0, arg.find('='));
        if (option == "--max-vram" || option == "--video-max-vram") {
            std::cerr << option << " was removed; native denoiser streaming plans from reported device memory"
                      << std::endl;
            exit(1);
        }
        if (const auto replacement = sdkit::retired_device_option_replacement(arg); !replacement.empty()) {
            std::cerr << arg << " was removed; use " << replacement << std::endl;
            exit(1);
        }
        if (arg == "--port" && i + 1 < argc) {
            try {
                args.port = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid port number: " << argv[i] << std::endl;
                print_usage(argv[0]);
                exit(1);
            }
        } else if (arg == "--log-level" && i + 1 < argc) {
            args.log_level = argv[++i];
        } else if (arg == "--driver") {
            if (driver_selected || i + 1 >= argc || std::string_view(argv[i + 1]).starts_with("-")) {
                std::cerr << "Specify --driver <name> once: auto, cpu, cuda, vulkan, oneapi, or openvino" << std::endl;
                exit(1);
            }
            if (const auto error = sdkit::parse_driver_name(argv[++i], false, args.compute_backend);
                !error.empty()) {
                std::cerr << error << std::endl;
                exit(1);
            }
            driver_selected = true;
        } else if (arg == "--device") {
            if (args.pcie_devices_set || i + 1 >= argc || std::string_view(argv[i + 1]).starts_with("-")) {
                std::cerr << "Specify --device once with physical PCIe address(es), e.g. 0000:01:00.0" << std::endl;
                exit(1);
            }
            args.pcie_devices = argv[++i];
            args.pcie_devices_set = true;
        } else if (arg == "--devices") {
            if (discovery_selected || i + 1 >= argc || std::string_view(argv[i + 1]) != "list") {
                std::cerr << "Use --devices list [driver] for discovery or --device <PCIe address> for selection"
                          << std::endl;
                exit(1);
            }
            ++i;
            discovery_selected = true;
            args.list_devices = true;
            if (i + 1 < argc && !std::string_view(argv[i + 1]).starts_with("-")) {
                if (const auto error = sdkit::parse_driver_name(argv[++i], true, args.list_devices_backend);
                    !error.empty()) {
                    std::cerr << error << std::endl;
                    exit(1);
                }
            }
        } else if (arg == "--parent-pid" && i + 1 < argc) {
            try {
                args.parent_pid = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid parent PID: " << argv[i] << std::endl;
                print_usage(argv[0]);
                exit(1);
            }
        } else if (arg == "--dir-stable-diffusion") {
            args.ckpt_dir = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--dir-vae") {
            args.vae_dir = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--dir-hypernetwork") {
            args.hypernetwork_dir = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--dir-gfpgan") {
            args.gfpgan_models_path = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--dir-realesrgan") {
            args.realesrgan_models_path = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--dir-lora") {
            args.lora_dir = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--dir-codeformer") {
            args.codeformer_models_path = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--dir-embeddings") {
            args.embeddings_dir = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--dir-controlnet") {
            args.controlnet_dir = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--control-net-sd1-path" && i + 1 < argc) {
            args.control_net_sd1_path = argv[++i];
        } else if (arg == "--control-net-sdxl-path" && i + 1 < argc) {
            args.control_net_sdxl_path = argv[++i];
        } else if (arg == "--dir-text-encoder") {
            args.text_encoder_dir = read_model_directory(arg, argc, argv, i);
        } else if (arg == "--image-vae-on-cpu") {
            args.image_vae_on_cpu = true;
        } else if (arg == "--no-half") {
            args.no_half = true;
        } else if (arg == "--no-half-vae") {
            args.no_half_vae = true;
        } else if (arg == "--vae-tiling") {
            args.vae_tiling = true;
        } else if (arg == "--vae-tile-size" && i + 1 < argc) {
            args.vae_tile_size = argv[++i];
        } else if ((arg == "--vae-tiles" || arg == "--vae-tiled-overlap") && i + 1 < argc) {
            try {
                const int value = std::stoi(argv[++i]);
                if (arg == "--vae-tiles") {
                    args.vae_tiles = value;
                } else {
                    args.vae_tiled_overlap = value;
                }
            } catch (const std::exception& e) {
                std::cerr << "Invalid " << arg << ": expected an integer" << std::endl;
                exit(1);
            }
        } else if (arg == "--offload-to-cpu") {
            args.offload_to_cpu = true;
        } else if (arg == "--mmap") {
            args.mmap_weights = true;
            args.mmap_explicitly_disabled = false;
        } else if (arg == "--no-mmap") {
            args.mmap_weights = false;
            args.mmap_explicitly_disabled = true;
        } else if (arg == "--keep-model-loaded") {
            args.keep_model_loaded = true;
        } else if (arg == "--attention" && i + 1 < argc) {
            args.attention_mode = argv[++i];
        } else if (arg == "--opt-split-attention") {
            args.split_attention = true;
        } else if (arg == "--diffusion-fa") {
            args.diffusion_fa = true;
        } else if (arg == "--flash-attention") {
            args.flash_attention = true;
        } else if (arg == "--xformers") {
            args.xformers_compat = true;
        } else if (arg == "--sage-attention") {
            args.sage_attention = true;
        } else if (arg == "--stream-layers") {
            args.stream_layers = true;
        } else if (arg == "--cuda-malloc") {
            args.cuda_malloc = true;
        } else if (arg == "--cuda-unified-memory") {
            args.cuda_unified_memory = true;
        } else if (arg == "--control-net-cpu") {
            args.control_net_cpu = true;
        } else if (arg == "--image-clip-on-cpu") {
            args.image_clip_on_cpu = true;
        } else if (arg == "--image-clip-vision-on-cpu") {
            args.image_clip_vision_on_cpu = true;
        } else if (arg == "--image-ip-adapter-on-cpu") {
            args.image_ip_adapter_on_cpu = true;
        } else if (arg == "--video-clip-on-cpu") {
            args.video_clip_on_cpu = true;
        } else if (arg == "--video-vae-on-cpu") {
            args.video_vae_on_cpu = true;
        } else if (arg == "--video-offload-to-cpu") {
            args.video_offload_to_cpu = true;
        } else if (arg == "--video-stream-layers") {
            args.video_stream_layers = true;
        } else if (arg == "--chroma-disable-dit-mask") {
            args.chroma_disable_dit_mask = true;
        } else if (arg == "--convert-model" && i + 1 < argc) {
            args.convert_model = argv[++i];
        } else if (arg == "--convert-output" && i + 1 < argc) {
            args.convert_output = argv[++i];
        } else if (arg == "--convert-type" && i + 1 < argc) {
            args.convert_type = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else if (const char* replacement = retired_model_directory_flag(arg)) {
            std::cerr << "Retired directory argument: " << arg.substr(0, arg.find('='))
                      << ". Use " << replacement << " <dir>." << std::endl;
            exit(1);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    if (discovery_selected && (driver_selected || args.pcie_devices_set)) {
        std::cerr << "Use --devices list [driver] on its own, without --driver or --device" << std::endl;
        exit(1);
    }

    if (args.vae_tiles < 4) {
        std::cerr << "--vae-tiles must be at least 4" << std::endl;
        exit(1);
    }
    if (args.vae_tiled_overlap < 0 || args.vae_tiled_overlap * 2 > args.vae_tiles) {
        std::cerr << "--vae-tiled-overlap must be between 0 and half of --vae-tiles" << std::endl;
        exit(1);
    }

    // Attention implementations are alternative graph pipelines, not additive
    // toggles. Preserve legacy flags, but resolve conflicts compatibility-first
    // rather than allowing a newer, more demanding implementation to mask an
    // older request. The explicit selector takes precedence over legacy flags.
    std::string selected_attention = args.attention_mode;
    if (selected_attention.empty()) {
        const bool legacy_flash_requested = args.diffusion_fa || args.flash_attention;
        const int legacy_attention_count = static_cast<int>(args.split_attention) +
                                           static_cast<int>(legacy_flash_requested) +
                                           static_cast<int>(args.xformers_compat) +
                                           static_cast<int>(args.sage_attention);
        if (args.split_attention) {
            selected_attention = "split";
        } else if (legacy_flash_requested) {
            selected_attention = "flash";
        } else if (args.xformers_compat) {
            selected_attention = "xformers";
        } else if (args.sage_attention) {
            selected_attention = "sage";
        } else {
            selected_attention = "split";
        }
        if (legacy_attention_count > 1) {
            std::cerr << "Multiple legacy attention flags supplied; selected compatibility-first mode: "
                      << selected_attention << std::endl;
        }
    }
    if (selected_attention == "standard") {
        // Accept the short-lived development spelling as a compatibility alias.
        selected_attention = "split";
    }
    if (selected_attention != "split" && selected_attention != "flash" &&
        selected_attention != "xformers" &&
        selected_attention != "sage") {
        std::cerr << "Invalid --attention mode: " << selected_attention << std::endl;
        std::cerr << "Expected split, flash, xformers, or sage" << std::endl;
        exit(1);
    }

    args.flash_attention = false;
    args.diffusion_fa = false;
    args.xformers_compat = false;
    args.sage_attention = false;
    args.split_attention = selected_attention == "split";
    args.attention_mode = selected_attention;
    if (selected_attention == "flash") {
        // The historical flags controlled the scope of the same Flash
        // Attention implementation. The unified mode applies it consistently.
        args.flash_attention = true;
        args.diffusion_fa = true;
    } else if (selected_attention == "xformers") {
        // xFormers is the selected FLASH_ATTN_EXT executor. These graph flags
        // expose eligible operations to it; they do not select extra backends.
        args.flash_attention = true;
        args.diffusion_fa = true;
        args.xformers_compat = true;
    } else if (selected_attention == "sage") {
        args.diffusion_fa = true;
        args.sage_attention = true;
    }
    if (args.control_net_sd1_path.empty() != args.control_net_sdxl_path.empty()) {
        std::cerr << "--control-net-sd1-path and --control-net-sdxl-path must be set together" << std::endl;
        exit(1);
    }
    if (!args.convert_model.empty() && args.convert_output.empty()) {
        std::cerr << "--convert-output is required with --convert-model" << std::endl;
        exit(1);
    }

    if (args.stream_layers && !args.offload_to_cpu) {
        std::cerr << "--stream-layers requires --offload-to-cpu" << std::endl;
        exit(1);
    }
    if (args.video_stream_layers && !args.offload_to_cpu && !args.video_offload_to_cpu) {
        std::cerr << "--video-stream-layers requires --video-offload-to-cpu or --offload-to-cpu" << std::endl;
        exit(1);
    }

    return args;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Parse command line arguments
    CommandLineArgs args = parse_args(argc, argv);

    if (args.cuda_malloc) {
#ifdef _WIN32
        _putenv_s("GGML_CUDA_FORCE_MALLOC", "1");
#else
        setenv("GGML_CUDA_FORCE_MALLOC", "1", 1);
#endif
    }
    if (args.cuda_unified_memory) {
#ifdef _WIN32
        _putenv_s("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1");
#else
        setenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1", 1);
#endif
    }

    // Set log level from command line argument
    set_log_level(args.log_level);

    if (args.list_devices) {
        if (!print_backend_devices(args.list_devices_backend)) {
            std::cerr << "No devices were reported for backend '" << args.list_devices_backend << "'." << std::endl;
            return 1;
        }
        return 0;
    }

    std::string compute_backend_spec;
    std::string backend_error;
    if (!build_compute_backend_spec(args, compute_backend_spec, backend_error)) {
        LOG_ERROR("%s", backend_error.c_str());
        std::cerr << "Run with --devices list to see the selectors available in this build." << std::endl;
        return 1;
    }
    const std::string selected_backend = lowercase(args.compute_backend);
    if (selected_backend != "auto" && selected_backend != "cuda" &&
        (args.cuda_malloc || args.cuda_unified_memory || args.xformers_compat || args.sage_attention)) {
        LOG_ERROR("CUDA-only memory and attention options require --backend cuda (or auto)");
        return 1;
    }

    if (!args.convert_model.empty()) {
        const sd_type_t output_type = str_to_sd_type(args.convert_type.c_str());
        if (output_type == SD_TYPE_COUNT) {
            LOG_ERROR("Unsupported conversion type: %s", args.convert_type.c_str());
            return 1;
        }
        LOG_INFO("Converting model '%s' to '%s' as %s",
                 args.convert_model.c_str(),
                 args.convert_output.c_str(),
                 args.convert_type.c_str());
        const bool success = convert(args.convert_model.c_str(),
                                     "",
                                     args.convert_output.c_str(),
                                     output_type,
                                     "",
                                     false);
        if (!success) {
            LOG_ERROR("Native model conversion failed");
            return 1;
        }
        LOG_INFO("Native model conversion completed");
        return 0;
    }
    if (args.cuda_malloc) {
        LOG_INFO("CUDA legacy malloc pool enabled (VMM scratch pool disabled)");
    }
    if (args.cuda_unified_memory) {
        LOG_WARNING("CUDA unified memory enabled: allocations may spill into system RAM and run much slower");
    }
    LOG_INFO("Attention pipeline: %s", args.attention_mode.c_str());
    if (args.xformers_compat) {
#ifdef _WIN32
        _putenv_s("SD_CUDA_XFORMERS", "1");
#else
        setenv("SD_CUDA_XFORMERS", "1", 1);
#endif
        LOG_INFO("--xformers enabled native fused C++/CUDA memory-efficient attention");
    }
    if (args.sage_attention) {
        LOG_INFO("Native SageAttention SM80 preference enabled; unsupported operations use ggml flash attention");
    }
    if (compute_backend_spec.empty()) {
        LOG_INFO("Compute backend: %s", selected_backend.c_str());
    } else {
        LOG_INFO("Compute backend: %s (%s)", selected_backend.c_str(), compute_backend_spec.c_str());
    }

    // Create and configure model manager
    auto model_manager = std::make_shared<ModelManager>();

    if (!args.ckpt_dir.empty()) {
        model_manager->setCheckpointDir(args.ckpt_dir);
    }
    if (!args.vae_dir.empty()) {
        model_manager->setVaeDir(args.vae_dir);
    }
    if (!args.hypernetwork_dir.empty()) {
        model_manager->setHypernetworkDir(args.hypernetwork_dir);
    }
    if (!args.gfpgan_models_path.empty()) {
        model_manager->setGfpganModelsPath(args.gfpgan_models_path);
    }
    if (!args.realesrgan_models_path.empty()) {
        model_manager->setRealesrganModelsPath(args.realesrgan_models_path);
    }
    if (!args.lora_dir.empty()) {
        model_manager->setLoraDir(args.lora_dir);
    }
    if (!args.codeformer_models_path.empty()) {
        model_manager->setCodeformerModelsPath(args.codeformer_models_path);
    }
    if (!args.embeddings_dir.empty()) {
        model_manager->setEmbeddingsDir(args.embeddings_dir);
    }
    if (!args.controlnet_dir.empty()) {
        model_manager->setControlnetDir(args.controlnet_dir);
    }
    if (!args.text_encoder_dir.empty()) {
        model_manager->setTextEncoderDir(args.text_encoder_dir);
    }

    // Scan all model directories
    std::cout << "Scanning model directories..." << std::endl;
    model_manager->scanAllDirectories();
    std::cout << "Model scanning complete." << std::endl;
    std::cout << std::endl;

    // Start parent process watchdog if parent PID is specified
    if (args.parent_pid > 0) {
        g_watchdog_thread = std::make_unique<std::thread>(parent_watchdog, args.parent_pid);
    }

    try {
        // Create server parameters
        ServerParams server_params;
        server_params.port = args.port;
        server_params.model_manager = model_manager;
        server_params.compute_backend = compute_backend_spec;
        server_params.image_vae_on_cpu = args.image_vae_on_cpu;
        server_params.no_half = args.no_half;
        server_params.no_half_vae = args.no_half_vae;
        server_params.vae_tiling = args.vae_tiling;
        server_params.vae_tile_size = args.vae_tile_size;
        server_params.vae_tiles = args.vae_tiles;
        server_params.vae_tiled_overlap = args.vae_tiled_overlap;
        server_params.offload_to_cpu = args.offload_to_cpu;
        server_params.mmap_weights = args.mmap_weights ||
                                     (args.offload_to_cpu && !args.mmap_explicitly_disabled);
        server_params.keep_model_loaded = args.keep_model_loaded;
        server_params.flash_attention = args.flash_attention;
        server_params.diffusion_fa = args.diffusion_fa;
        server_params.sage_attention = args.sage_attention;
        server_params.stream_layers = args.stream_layers;
        server_params.control_net_cpu = args.control_net_cpu;
        server_params.control_net_sd1_path = args.control_net_sd1_path;
        server_params.control_net_sdxl_path = args.control_net_sdxl_path;
        server_params.image_clip_on_cpu = args.image_clip_on_cpu;
        server_params.image_clip_vision_on_cpu = args.image_clip_vision_on_cpu;
        server_params.image_ip_adapter_on_cpu = args.image_ip_adapter_on_cpu;
        server_params.video_clip_on_cpu = args.video_clip_on_cpu;
        server_params.video_vae_on_cpu = args.video_vae_on_cpu;
        server_params.video_offload_to_cpu = args.video_offload_to_cpu;
        server_params.video_mmap_weights = args.video_offload_to_cpu && !args.mmap_explicitly_disabled;
        server_params.video_stream_layers = args.video_stream_layers;
        server_params.chroma_disable_dit_mask = args.chroma_disable_dit_mask;

        // Create and start the server
        g_server = std::make_unique<Server>(server_params);
        g_server->run();
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        g_should_exit.store(true);
        if (g_watchdog_thread && g_watchdog_thread->joinable()) {
            g_watchdog_thread->join();
        }
        // The server owns CUDA-backed model contexts. Destroy it before main
        // returns so CUDA is still initialized when those buffers are freed.
        g_server.reset();
        return 1;
    }

    // Clean up watchdog thread
    g_should_exit.store(true);
    if (g_watchdog_thread && g_watchdog_thread->joinable()) {
        g_watchdog_thread->join();
    }

    // g_server has static storage duration, while CUDA's internal globals live
    // in shared libraries with an unspecified destruction order. Releasing the
    // model here prevents cudaFree calls after the driver has begun shutdown.
    g_server.reset();

    std::cout << "Server stopped." << std::endl;
    return 0;
}
