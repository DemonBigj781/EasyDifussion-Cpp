#ifndef __IMAGE_GENERATOR_H__
#define __IMAGE_GENERATOR_H__

#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "image_filters.h"
#include "model_manager.h"
#include "options_manager.h"
#include "stable-diffusion.h"
#include "task_state.h"

// Forward declaration
struct ServerParams;

struct ImageGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    std::vector<std::string> lora_paths;
    std::vector<float> lora_alphas;
    int width = 512;
    int height = 512;
    int steps = 20;
    float cfg_scale = 7.0f;
    int64_t seed = -1;
    int batch_count = 1;
    int batch_size = 1;

    // Sampler settings
    sample_method_t sampler = EULER_A_SAMPLE_METHOD;
    scheduler_t scheduler = DISCRETE_SCHEDULER;

    // img2img specific
    std::string init_image_base64;
    std::string mask_base64;
    float strength = 0.75f;

    // ControlNet specific
    std::string control_image_base64;
    float control_strength = 1.0f;
    std::string controlnet_model;
    control_net_type_t control_net_type = CONTROL_NET_TYPE_CANNY;

    // ControlNet-LLLite specific
    std::string control_net_lllite_image_base64;
    std::string control_net_lllite_model_path;
    float control_net_lllite_strength = 1.0f;
    float control_net_lllite_start_percent = 0.0f;
    float control_net_lllite_end_percent = 0.0f;

    // Native IP-Adapter (base SD1.5/SDXL checkpoints)
    std::string ip_adapter_image_base64;
    std::string ip_adapter_model_path;
    std::string ip_adapter_clip_vision_path;
    float ip_adapter_strength = 1.0f;
    float ip_adapter_start_percent = 0.0f;
    float ip_adapter_end_percent = 100.0f;

    // Staged v1/xl/v3/fx latent sources and city96 v4.0 conversion
    std::string latent_interposer_model_path;
    std::string furception_vae_path;
    std::string latent_interposer_source_image_base64;
    bool latent_interposer_enabled = false;
    sd_latent_source_t latent_interposer_source = SD_LATENT_SOURCE_NONE;
    int latent_interposer_phase_x = 0;
    int latent_interposer_phase_y = 0;

    // Optional VAE/model latent bridges. Encode is VAE -> diffusion model;
    // decode is diffusion model -> VAE and can be enabled independently.
    std::string latent_interposer_encode_model_path;
    std::string latent_interposer_decode_model_path;
    sd_vae_format_t latent_interposer_vae_format = SD_VAE_FORMAT_AUTO;

    // Reference images for vision-based models (Qwen, etc.)
    std::vector<std::string> ref_images_base64;
    bool auto_resize_ref_image = true;

    // Other options
    int clip_skip = -1;
};

struct VideoGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    std::vector<std::string> lora_paths;
    std::vector<float> lora_alphas;
    std::string init_image_base64;
    std::string end_image_base64;
    int width = 512;
    int height = 512;
    int steps = 20;
    int frames = 25;
    int fps = 8;
    int motion_bucket_id = 127;
    float augmentation_level = 0.0f;
    float cfg_scale = 5.0f;
    float strength = 0.75f;
    float flow_shift = INFINITY;
    float moe_boundary = 0.875f;
    int64_t seed = -1;
    int clip_skip = -1;
    sample_method_t sampler = EULER_SAMPLE_METHOD;
    scheduler_t scheduler = DISCRETE_SCHEDULER;
    sd_cache_mode_t cache_mode = SD_CACHE_DISABLED;
    float cache_threshold = INFINITY;
    float cache_start_percent = 0.15f;
    float cache_end_percent = 0.95f;
};

class ImageGenerator {
   public:
    ImageGenerator(std::shared_ptr<TaskStateManager> task_state_manager,
                   std::shared_ptr<OptionsManager> options_manager, std::shared_ptr<ModelManager> model_manager,
                   std::shared_ptr<ImageFilters> image_filters, const ServerParams& server_params);
    ~ImageGenerator();

    // Check if initialized
    bool isInitialized() const;

    // Get currently loaded model path
    std::string getCurrentModelPath() const;

    // Generate images
    std::vector<std::string> generateTxt2Img(const ImageGenerationParams& params, const std::string& task_id = "");

    std::vector<std::string> generateImg2Img(const ImageGenerationParams& params, const std::string& task_id = "");

    // Native stable-diffusion.cpp SVD/Wan/LTX video path. Frames are returned
    // as PNG base64 strings so the HTTP/UI layer can stream or encode them.
    std::vector<std::string> generateVideo(const VideoGenerationParams& params, const std::string& task_id = "");

    // Interrupt current generation
    void interrupt();

   private:
    // Create init image from base64 string
    sd_image_t createInitImage(const ImageGenerationParams& params);

    // Create mask image from base64 string or default
    sd_image_t createMaskImage(const ImageGenerationParams& params);

    // Create control image from base64 string and apply canny preprocessing
    sd_image_t createControlImage(const ImageGenerationParams& params);

    // Decode and resize an LLLite condition image without preprocessing
    sd_image_t createControlNetLLLiteImage(const ImageGenerationParams& params);

    // Decode the NumPy-PCG source image used by the original Furception node.
    sd_image_t createLatentInterposerSourceImage(const ImageGenerationParams& params);

    // Create reference images from base64 strings (for vision-based models)
    std::vector<sd_image_t> createRefImages(const ImageGenerationParams& params);

    // Free sd_image_t data
    void freeImage(sd_image_t& image);

    // Generate image internally
    std::vector<std::string> generateInternal(const ImageGenerationParams& params, bool is_img2img,
                                              const std::string& task_id, bool allow_ram_fallback = true);

    // Check if model needs to be reloaded based on options
    bool needsModelReload(const std::string& model_path) const;

    // Ensure model is loaded based on current options and controlnet
    bool ensureModelLoaded(const std::string& controlnet_model = "",
                           const std::string& control_net_lllite_model_path = "",
                           const std::string& ip_adapter_model_path = "",
                           const std::string& ip_adapter_clip_vision_path = "",
                           const std::string& latent_interposer_model_path = "",
                           const std::string& furception_vae_path = "",
                           const std::string& latent_interposer_encode_model_path = "",
                           const std::string& latent_interposer_decode_model_path = "",
                           sd_vae_format_t latent_interposer_vae_format = SD_VAE_FORMAT_AUTO);

    sd_ctx_t* sd_ctx_;
    std::shared_ptr<TaskStateManager> task_state_manager_;
    std::shared_ptr<OptionsManager> options_manager_;
    std::shared_ptr<ModelManager> model_manager_;
    std::shared_ptr<ImageFilters> image_filters_;
    std::mutex mutex_;
    // Kept separate from mutex_: generation holds mutex_ for the complete
    // request, while cancellation must remain callable from another HTTP
    // worker.  The short-lived guard also keeps the active context alive until
    // sd_cancel_generation() has published its lock-free cancellation flag.
    std::mutex interrupt_mutex_;
    sd_ctx_t* active_generation_ctx_;
    bool cancel_requested_;
    bool initialized_;
    std::string current_task_id_;

    // Track currently loaded model paths for change detection
    std::string current_model_path_;
    std::string current_vae_path_;
    std::string current_clip_l_path_;
    std::string current_clip_g_path_;
    std::string current_clip_vision_path_;
    std::string current_t5xxl_path_;
    std::string current_llm_path_;
    std::string current_taesd_path_;
    std::string current_lora_model_dir_;
    std::string current_embeddings_dir_;
    std::string current_controlnet_path_;
    std::string current_control_net_lllite_path_;
    std::string current_ip_adapter_path_;
    std::string current_latent_interposer_path_;
    std::string current_latent_interposer_encode_path_;
    std::string current_latent_interposer_decode_path_;
    sd_vae_format_t current_latent_interposer_vae_format_ = SD_VAE_FORMAT_AUTO;
    std::string current_furception_vae_path_;
    // If a CUDA generation for this checkpoint fails, rebuild its context with
    // VAE compute on CPU. Diffusion layers remain on CUDA and their parameters
    // continue streaming from the CPU backend.
    std::string cpu_vae_fallback_model_path_;
    bool current_vae_uses_cpu_ = false;

    // SD context parameters from CLI
    bool vae_on_cpu_;
    bool vae_tiling_;
    std::string vae_tile_size_;
    bool offload_to_cpu_;
    bool mmap_weights_;
    bool keep_model_loaded_;
    bool flash_attention_;
    bool diffusion_fa_;
    bool sage_attention_;
    std::string max_vram_;
    bool stream_layers_;
    bool control_net_cpu_;
    bool clip_on_cpu_;
    bool chroma_disable_dit_mask_;
};

#endif  // __IMAGE_GENERATOR_H__
