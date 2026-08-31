#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#else
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#endif

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
    std::string text_encoder_dir;
    bool image_vae_on_cpu = false;
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
    std::string max_vram;
    bool stream_layers = false;
    bool cuda_malloc = false;
    bool xformers_compat = false;
    bool control_net_cpu = false;
    bool image_clip_on_cpu = false;
    bool video_clip_on_cpu = false;
    bool video_vae_on_cpu = false;
    bool video_offload_to_cpu = false;
    std::string video_max_vram;
    bool video_stream_layers = false;
    bool chroma_disable_dit_mask = false;
    std::string convert_model;
    std::string convert_output;
    std::string convert_type = "f16";
};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --port <port>                      Server port (default: 8188)" << std::endl;
    std::cerr << "  --log-level <level>                Log level: verbose, debug, info, warning, error (default: info)"
              << std::endl;
    std::cerr << "  --parent-pid <pid>                 Parent process PID" << std::endl;
    std::cerr << "  --ckpt-dir <path>                  Checkpoint models directory" << std::endl;
    std::cerr << "  --vae-dir <path>                   VAE models directory" << std::endl;
    std::cerr << "  --hypernetwork-dir <path>          Hypernetwork models directory" << std::endl;
    std::cerr << "  --gfpgan-models-path <path>        GFPGAN models directory" << std::endl;
    std::cerr << "  --realesrgan-models-path <path>    RealESRGAN models directory" << std::endl;
    std::cerr << "  --lora-dir <path>                  LoRA models directory" << std::endl;
    std::cerr << "  --codeformer-models-path <path>    Codeformer models directory" << std::endl;
    std::cerr << "  --embeddings-dir <path>            Embeddings directory" << std::endl;
    std::cerr << "  --controlnet-dir <path>            ControlNet models directory" << std::endl;
    std::cerr << "  --text-encoder-dir <path>          Text encoder models directory" << std::endl;
    std::cerr << "  --image-vae-on-cpu                 Keep image-generation VAE on CPU (default: false)" << std::endl;
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
    std::cerr << "  --diffusion-fa                     Enable diffusion flash attention (default: false)" << std::endl;
    std::cerr << "  --flash-attention                  Enable native memory-efficient attention for all modules" << std::endl;
    std::cerr << "  --xformers                         C++/CUDA xFormers-equivalent fused attention" << std::endl;
    std::cerr << "  --sage-attention                  Prefer native SageAttention SM80 INT8-QK kernels" << std::endl;
    std::cerr << "  --max-vram <GiB|assignments>       Graph VRAM budget, e.g. 6 or cuda=6,cpu=0" << std::endl;
    std::cerr << "  --stream-layers                   Stream model layers within the --max-vram budget" << std::endl;
    std::cerr << "  --cuda-malloc                     Use the flushable legacy cudaMalloc pool instead of CUDA VMM" << std::endl;
    std::cerr << "  --control-net-cpu                  Keep ControlNet on CPU (default: false)" << std::endl;
    std::cerr << "  --image-clip-on-cpu                Keep image-generation text encoders on CPU (default: false)"
              << std::endl;
    std::cerr << "  --video-clip-on-cpu                Keep native-video text encoders on CPU (default: false)"
              << std::endl;
    std::cerr << "  --video-vae-on-cpu                 Keep native-video VAE on CPU (default: false)" << std::endl;
    std::cerr << "  --video-offload-to-cpu             Offload native-video parameters to CPU (default: false)"
              << std::endl;
    std::cerr << "  --video-max-vram <GiB>             Native-video graph VRAM budget" << std::endl;
    std::cerr << "  --video-stream-layers              Stream native-video diffusion layers within its VRAM budget"
              << std::endl;
    std::cerr << "  --chroma-disable-dit-mask          Disable DiT mask for Chroma models (default: false)"
              << std::endl;
    std::cerr << "  --convert-model <path>             Convert a checkpoint, safetensors, or Diffusers model and exit"
              << std::endl;
    std::cerr << "  --convert-output <path>            GGUF output path used with --convert-model" << std::endl;
    std::cerr << "  --convert-type <type>              Conversion type (default: f16; e.g. f32, bf16, q8_0, q5_0, q4_0)"
              << std::endl;
}

CommandLineArgs parse_args(int argc, char* argv[]) {
    CommandLineArgs args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

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
        } else if (arg == "--parent-pid" && i + 1 < argc) {
            try {
                args.parent_pid = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid parent PID: " << argv[i] << std::endl;
                print_usage(argv[0]);
                exit(1);
            }
        } else if (arg == "--ckpt-dir" && i + 1 < argc) {
            args.ckpt_dir = argv[++i];
        } else if (arg == "--vae-dir" && i + 1 < argc) {
            args.vae_dir = argv[++i];
        } else if (arg == "--hypernetwork-dir" && i + 1 < argc) {
            args.hypernetwork_dir = argv[++i];
        } else if (arg == "--gfpgan-models-path" && i + 1 < argc) {
            args.gfpgan_models_path = argv[++i];
        } else if (arg == "--realesrgan-models-path" && i + 1 < argc) {
            args.realesrgan_models_path = argv[++i];
        } else if (arg == "--lora-dir" && i + 1 < argc) {
            args.lora_dir = argv[++i];
        } else if (arg == "--codeformer-models-path" && i + 1 < argc) {
            args.codeformer_models_path = argv[++i];
        } else if (arg == "--embeddings-dir" && i + 1 < argc) {
            args.embeddings_dir = argv[++i];
        } else if (arg == "--controlnet-dir" && i + 1 < argc) {
            args.controlnet_dir = argv[++i];
        } else if (arg == "--text-encoder-dir" && i + 1 < argc) {
            args.text_encoder_dir = argv[++i];
        } else if (arg == "--image-vae-on-cpu") {
            args.image_vae_on_cpu = true;
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
        } else if (arg == "--diffusion-fa") {
            args.diffusion_fa = true;
        } else if (arg == "--flash-attention") {
            args.flash_attention = true;
            args.diffusion_fa = true;
        } else if (arg == "--xformers") {
            args.xformers_compat = true;
            args.flash_attention = true;
            args.diffusion_fa = true;
        } else if (arg == "--sage-attention") {
            args.sage_attention = true;
            args.diffusion_fa = true;
        } else if (arg == "--max-vram" && i + 1 < argc) {
            args.max_vram = argv[++i];
        } else if (arg == "--stream-layers") {
            args.stream_layers = true;
        } else if (arg == "--cuda-malloc") {
            args.cuda_malloc = true;
        } else if (arg == "--control-net-cpu") {
            args.control_net_cpu = true;
        } else if (arg == "--image-clip-on-cpu") {
            args.image_clip_on_cpu = true;
        } else if (arg == "--video-clip-on-cpu") {
            args.video_clip_on_cpu = true;
        } else if (arg == "--video-vae-on-cpu") {
            args.video_vae_on_cpu = true;
        } else if (arg == "--video-offload-to-cpu") {
            args.video_offload_to_cpu = true;
        } else if (arg == "--video-max-vram" && i + 1 < argc) {
            args.video_max_vram = argv[++i];
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
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    if (args.vae_tiles < 4) {
        std::cerr << "--vae-tiles must be at least 4" << std::endl;
        exit(1);
    }
    if (args.vae_tiled_overlap < 0 || args.vae_tiled_overlap * 2 > args.vae_tiles) {
        std::cerr << "--vae-tiled-overlap must be between 0 and half of --vae-tiles" << std::endl;
        exit(1);
    }
    if (!args.convert_model.empty() && args.convert_output.empty()) {
        std::cerr << "--convert-output is required with --convert-model" << std::endl;
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

    // Set log level from command line argument
    set_log_level(args.log_level);

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
        server_params.image_vae_on_cpu = args.image_vae_on_cpu;
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
        server_params.max_vram = args.max_vram;
        server_params.stream_layers = args.stream_layers;
        server_params.control_net_cpu = args.control_net_cpu;
        server_params.image_clip_on_cpu = args.image_clip_on_cpu;
        server_params.video_clip_on_cpu = args.video_clip_on_cpu;
        server_params.video_vae_on_cpu = args.video_vae_on_cpu;
        server_params.video_offload_to_cpu = args.video_offload_to_cpu;
        server_params.video_mmap_weights = args.video_offload_to_cpu && !args.mmap_explicitly_disabled;
        server_params.video_max_vram = args.video_max_vram;
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
