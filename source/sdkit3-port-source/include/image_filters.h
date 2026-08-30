#ifndef __IMAGE_FILTERS_H__
#define __IMAGE_FILTERS_H__

#include <memory>
#include <string>
#include <vector>

#include "model_manager.h"
#include "stable-diffusion.h"

// Image filter manager class that handles various image processing operations
class ImageFilters {
   public:
    ImageFilters(std::shared_ptr<ModelManager> model_manager);
    ~ImageFilters();

    // Upscaling
    std::vector<std::string> upscaleBatch(const std::vector<std::string>& base64_images,
                                          const std::string& upscaler_name = "", int upscale_factor = 4);
    sd_image_t upscaleImage(const sd_image_t& input_image, int upscale_factor = 4);

    // ControlNet preprocessing
    std::vector<std::string> applyControlNetFilterBatch(const std::vector<std::string>& base64_images,
                                                        const std::string& module = "canny");
    sd_image_t applyControlNetFilter(const sd_image_t& input_image, const std::string& module = "canny");

    // Future: Add GFPGAN and other filters
    // std::vector<std::string> restoreFacesBatch(const std::vector<std::string>& base64_images);
    // sd_image_t restoreFaces(const sd_image_t& input_image);

   private:
    std::shared_ptr<ModelManager> model_manager_;
    // Ensure upscaler is loaded
    bool ensureUpscalerLoaded(const std::string& upscaler_name = "");

    upscaler_ctx_t* upscaler_ctx_;
    std::string current_upscaler_path_;
};

#endif  // __IMAGE_FILTERS_H__
