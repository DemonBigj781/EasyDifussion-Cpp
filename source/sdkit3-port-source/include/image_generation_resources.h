#ifndef SDKIT_IMAGE_GENERATION_RESOURCES_H
#define SDKIT_IMAGE_GENERATION_RESOURCES_H

#include "image_utils.h"
#include <vector>
#include <functional>
#include <utility>

namespace sdkit {

// Run while the request mutex is still held. Explicit reset before recursive
// fallback prevents an older scope from clearing the retry's callback state.
class ImageGenerationStateGuard {
    std::function<void()> cleanup_;
public:
    explicit ImageGenerationStateGuard(std::function<void()> cleanup) : cleanup_(std::move(cleanup)) {}
    ImageGenerationStateGuard(const ImageGenerationStateGuard&) = delete;
    ImageGenerationStateGuard& operator=(const ImageGenerationStateGuard&) = delete;
    ~ImageGenerationStateGuard() { reset(); }
    void reset() {
        auto action = std::move(cleanup_);
        cleanup_ = {};
        if (action) action();
    }
};

// These guards own only request/result image allocations, never model tensors.
// Native generation receives borrowed descriptors until the guards are reset.
class ImageBufferGuard {
    sd_image_t& image_;
public:
    explicit ImageBufferGuard(sd_image_t& image) : image_(image) {}
    ImageBufferGuard(const ImageBufferGuard&) = delete;
    ImageBufferGuard& operator=(const ImageBufferGuard&) = delete;
    ~ImageBufferGuard() { ::freeImage(image_); }
};

class ImageListGuard {
    std::vector<sd_image_t>& images_;
    bool active_ = true;
public:
    explicit ImageListGuard(std::vector<sd_image_t>& images) : images_(images) {}
    ImageListGuard(const ImageListGuard&) = delete;
    ImageListGuard& operator=(const ImageListGuard&) = delete;
    ~ImageListGuard() {
        if (active_) for (auto& image : images_) ::freeImage(image);
    }
    void dismiss() noexcept { active_ = false; }
};

class ImageInputResources {
    sd_img_gen_params_t& params_;
public:
    std::vector<sd_image_t> references;
    explicit ImageInputResources(sd_img_gen_params_t& params) : params_(params) {}
    ImageInputResources(const ImageInputResources&) = delete;
    ImageInputResources& operator=(const ImageInputResources&) = delete;
    ~ImageInputResources() { reset(); }
    void reset() noexcept {
        ::freeImage(params_.init_image);
        ::freeImage(params_.mask_image);
        ::freeImage(params_.control_image);
        ::freeImage(params_.control_net_lllite_image);
        ::freeImage(params_.ip_adapter.image);
        ::freeImage(params_.latent_interposer_source_image);
        for (auto& image : references) ::freeImage(image);
        references.clear();
        params_.ref_images = nullptr;
        params_.ref_images_count = 0;
    }
};

class ImageResultResources {
public:
    sd_image_t* images = nullptr;
    int count = 0;
    ImageResultResources() = default;
    ImageResultResources(const ImageResultResources&) = delete;
    ImageResultResources& operator=(const ImageResultResources&) = delete;
    ~ImageResultResources() { reset(); }
    void reset() noexcept {
        free_sd_images(images, count);
        images = nullptr;
        count = 0;
    }
};

} // namespace sdkit
#endif
