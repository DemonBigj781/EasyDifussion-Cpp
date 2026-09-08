#include "features/vae/common/vae.hpp"

#include <cmath>

namespace edcpp::inference::vae {
namespace {

bool valid_config(const Config& config, std::string& message) {
    if (config.image_channels == 0 || config.latent_channels == 0 || config.spatial_downscale == 0) {
        message = "VAE channel counts and spatial downscale must be non-zero";
        return false;
    }
    if (!std::isfinite(config.latent_scale) || config.latent_scale == 0.0f ||
        !std::isfinite(config.latent_shift)) {
        message = "VAE latent scale and shift must be finite, and scale must be non-zero";
        return false;
    }
    if (config.tiling.enabled) {
        if (config.tiling.tile_width == 0 || config.tiling.tile_height == 0) {
            message = "enabled VAE tiling requires non-zero tile dimensions";
            return false;
        }
        if (config.tiling.overlap >= config.tiling.tile_width ||
            config.tiling.overlap >= config.tiling.tile_height) {
            message = "VAE tile overlap must be smaller than both tile dimensions";
            return false;
        }
    }
    return true;
}

bool valid_shape(const Shape& shape) {
    return shape.batch != 0 && shape.channels != 0 && shape.height != 0 && shape.width != 0;
}

} // namespace

Result make_encode_plan(const Shape& image, const Config& config) {
    Result result;
    if (!valid_config(config, result.message)) return result;
    if (!valid_shape(image) || image.channels != config.image_channels) {
        result.message = "VAE encode requires a valid image shape with the configured channel count";
        return result;
    }
    if ((image.height % config.spatial_downscale) != 0 ||
        (image.width % config.spatial_downscale) != 0) {
        result.message = "VAE encode dimensions must be divisible by the spatial downscale";
        return result;
    }
    result.ok = true;
    result.plan = {Operation::encode, image,
                   {image.batch, config.latent_channels,
                    image.height / config.spatial_downscale,
                    image.width / config.spatial_downscale},
                   config};
    return result;
}

Result make_decode_plan(const Shape& latent, const Config& config) {
    Result result;
    if (!valid_config(config, result.message)) return result;
    if (!valid_shape(latent) || latent.channels != config.latent_channels) {
        result.message = "VAE decode requires a valid latent shape with the configured channel count";
        return result;
    }
    result.ok = true;
    result.plan = {Operation::decode, latent,
                   {latent.batch, config.image_channels,
                    latent.height * config.spatial_downscale,
                    latent.width * config.spatial_downscale},
                   config};
    return result;
}

} // namespace edcpp::inference::vae
