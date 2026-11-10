#ifndef SDKIT_GENERATION_OPTIONS_H
#define SDKIT_GENERATION_OPTIONS_H

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include "crow/json.h"
#include "stable-diffusion.h"
#include "image_utils.h"

inline std::string validate_generation_options(const crow::json::rvalue& body, bool masked_image) {
    if (body.has("distilled_cfg_scale")) {
        const auto& value = body["distilled_cfg_scale"];
        if (value.t() != crow::json::type::Number || !std::isfinite(value.d()) ||
            std::abs(value.d()) > std::numeric_limits<float>::max()) {
            return "distilled_cfg_scale must be a finite float";
        }
    }
    if (masked_image && body.has("mask_blur")) {
        const auto& value = body["mask_blur"];
        if (value.t() != crow::json::type::Number || !std::isfinite(value.d()) ||
            value.d() < 0 || value.d() > kMaxMaskBlur) {
            return "mask_blur must be a number between 0 and 64";
        }
    }
    return {};
}

inline std::optional<float> parse_distilled_guidance(const crow::json::rvalue& body) {
    if (!body.has("distilled_cfg_scale")) return std::nullopt;
    return static_cast<float>(body["distilled_cfg_scale"].d());
}

inline void apply_distilled_guidance(sd_sample_params_t& sampling, std::optional<float> requested) {
    if (requested) sampling.guidance.distilled_guidance = *requested;
}

#endif
