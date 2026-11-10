#ifndef __IMAGE_UTILS_H__
#define __IMAGE_UTILS_H__

#include <string>

#include "stable-diffusion.h"

// Image format conversion utilities
std::string imageToBase64(const sd_image_t& image);
sd_image_t base64ToImage(const std::string& base64_data, int desired_channels = 3);

// Allocate a filled image; returns an empty descriptor on invalid size or allocation failure.
sd_image_t createSolidImage(int width, int height, int channels, uint8_t value);

// Image memory management
void freeImage(sd_image_t& image);

// Image manipulation
bool resizeImage(sd_image_t& image, int target_width, int target_height, bool clamp_to_8 = false);

#endif  // __IMAGE_UTILS_H__
