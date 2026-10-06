#include "image_bridge.h"

#include <cstdio>
#include <exception>

extern "C" sd_ctx_t *cosmo_image_create_context(const sd_ctx_params_t *params,
                                                char *error, size_t error_size) {
    if (error_size) error[0] = '\0';
    try {
        return new_sd_ctx(params);
    } catch (const std::exception &exception) {
        if (error_size) std::snprintf(error, error_size, "%s", exception.what());
    } catch (...) {
        if (error_size) std::snprintf(error, error_size, "unknown native exception during model loading");
    }
    return nullptr;
}

extern "C" bool cosmo_image_generate_pixels(sd_ctx_t *context, const sd_img_gen_params_t *params,
                                            sd_image_t **images, int *image_count,
                                            char *error, size_t error_size) {
    *images = nullptr;
    *image_count = 0;
    if (error_size) error[0] = '\0';
    try {
        return generate_image(context, params, images, image_count);
    } catch (const std::exception &exception) {
        if (error_size) std::snprintf(error, error_size, "%s", exception.what());
    } catch (...) {
        if (error_size) std::snprintf(error, error_size, "unknown native exception during image generation");
    }
    return false;
}
