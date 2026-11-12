#ifndef __SD_RUNTIME_VAE_DECODE_RETRY_H__
#define __SD_RUNTIME_VAE_DECODE_RETRY_H__

#include "core/backend_fit.h"
#include "model/vae/vae.hpp"

namespace sd {

template <typename CancelFn>
inline Tensor<float> decode_vae_with_retries(VAE& vae,
                                             int n_threads,
                                             const Tensor<float>& latents,
                                             sd_tiling_params_t& tiling_params,
                                             bool auto_fit_enabled,
                                             bool decode_video,
                                             bool circular_x,
                                             bool circular_y,
                                             CancelFn&& cancelled) {
    auto decoded = vae.decode(n_threads, latents, tiling_params, decode_video, circular_x, circular_y);
    if (decoded.empty() && cancelled()) {
        LOG_INFO("VAE decode cancelled; skipping tiled fallback");
        return {};
    }
    const bool prefer_temporal_tiling = decode_video && vae.can_temporal_tile_decode();
    while (decoded.empty() &&
           !cancelled() &&
           vae.can_retry_decode() &&
           auto_fit_enabled &&
           sd::backend_fit::prepare_vae_decode_retry_tiling(tiling_params, prefer_temporal_tiling)) {
        vae.free_compute_buffer();
        decoded = vae.decode(n_threads, latents, tiling_params, decode_video, circular_x, circular_y);
    }
    if (decoded.empty() && !cancelled() &&
        vae.can_retry_decode() && !tiling_params.enabled && !decode_video) {
        sd_tiling_params_t fallback = tiling_params;
        fallback.enabled            = true;
        const int fallback_tile_x   = fallback.tile_size_x >= 4 ? fallback.tile_size_x : 32;
        const int fallback_tile_y   = fallback.tile_size_y >= 4 ? fallback.tile_size_y : 32;
        fallback.tile_size_x        = fallback_tile_x;
        fallback.tile_size_y        = fallback_tile_y;
        const int fallback_overlap  = static_cast<int>(std::round(
            std::min(fallback_tile_x, fallback_tile_y) * fallback.target_overlap));
        LOG_WARN("untiled VAE decode failed; retrying once with automatic %dx%d latent tiling and %d-pixel overlap",
                 fallback_tile_x,
                 fallback_tile_y,
                 fallback_overlap);
        vae.free_compute_buffer();
        decoded = vae.decode(n_threads, latents, fallback, false, circular_x, circular_y);
    }
    return decoded;
}

}  // namespace sd

#endif  // __SD_RUNTIME_VAE_DECODE_RETRY_H__
