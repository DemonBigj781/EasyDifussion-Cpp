#include "logging.h"

#include <stdexcept>
#include <thread>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    const std::string generic = "Generation failed: Image generation failed";
    const char* gpu_errors[] = {
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 7052.27 MiB on device 0: cudaMalloc failed: out of memory",
        "CUDA error: out of memory",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "hipMalloc failed: out of memory",
        "hipErrorOutOfMemory",
        "Vulkan: VK_ERROR_OUT_OF_DEVICE_MEMORY",
    };
    for (const char* message : gpu_errors) {
        reset_sd_generation_error();
        sd_log_cb(SD_LOG_ERROR, message, nullptr);
        sd_log_cb(SD_LOG_ERROR, "vae encode compute failed", nullptr);
        require(sd_generation_error_message(generic).find("Out of VRAM:") == 0,
                "GPU allocation failure was hidden by a later generic error");
    }
    // A worker request must not inherit another request thread's memory error.
    bool isolated = false;
    std::thread other([&]() { isolated = sd_generation_error_message(generic) == generic; });
    other.join();
    require(isolated, "GPU error leaked to another request thread");

    require(sd_generation_error_message("Generation failed: vk::Device::waitForFences: ErrorDeviceLost")
                .find("Vulkan GPU device lost") == 0,
            "Vulkan device loss was mislabeled as VRAM exhaustion");
    reset_sd_generation_error();
    require(sd_generation_error_message(generic) == generic, "GPU error leaked to the next request");
    for (const char* message : {"CPU malloc: out of memory", "CUDA host allocation: out of memory",
                                "VK_ERROR_OUT_OF_HOST_MEMORY", "Model file not found",
                                "CUDA error: invalid device ordinal"}) {
        sd_log_cb(SD_LOG_ERROR, message, nullptr);
        require(sd_generation_error_message(generic) == generic, "Non-VRAM failure was mislabeled");
    }
    sd_log_cb(SD_LOG_INFO, "CUDA out of memory fallback is available", nullptr);
    require(sd_generation_error_message(generic) == generic, "Informational log was treated as an error");
    return 0;
}
