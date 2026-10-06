#ifndef EASY_DIFFUSION_COSMO_IMAGE_BRIDGE_H
#define EASY_DIFFUSION_COSMO_IMAGE_BRIDGE_H

#include "stable-diffusion.h"

#ifdef __cplusplus
extern "C" {
#endif

sd_ctx_t *cosmo_image_create_context(const sd_ctx_params_t *params,
                                     char *error, size_t error_size);
bool cosmo_image_generate_pixels(sd_ctx_t *context, const sd_img_gen_params_t *params,
                                 sd_image_t **images, int *image_count,
                                 char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
