#ifndef __SD_RUNTIME_UPSCALE_ALPHA_H__
#define __SD_RUNTIME_UPSCALE_ALPHA_H__

#include "core/tensor.hpp"
#include "core/util.h"

namespace sd {

template <typename UpscaleFn>
inline Tensor<float> upscale_rgb_preserving_alpha(const Tensor<float>& input, UpscaleFn&& upscale_rgb) {
    if (input.empty() || input.dim() != 4 || (input.shape()[2] != 3 && input.shape()[2] != 4)) {
        LOG_ERROR("esrgan expects a 4D RGB or RGBA image tensor");
        return {};
    }
    const bool has_alpha = input.shape()[2] == 4;
    Tensor<float> rgb;
    if (has_alpha) {
        rgb = ops::slice(input, 2, 0, 3);
    }
    auto upscaled = upscale_rgb(has_alpha ? rgb : input);
    if (upscaled.empty() || !has_alpha) {
        return upscaled;
    }
    auto alpha = ops::slice(input, 2, 3, 4);
    auto shape = alpha.shape();
    shape[0] = upscaled.shape()[0];
    shape[1] = upscaled.shape()[1];
    alpha = ops::interpolate(alpha, shape, ops::InterpolateMode::Bilinear);
    return ops::concat(upscaled, alpha, 2);
}

}  // namespace sd

#endif  // __SD_RUNTIME_UPSCALE_ALPHA_H__
