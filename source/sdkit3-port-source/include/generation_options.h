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
    if (masked_image) {
        if (body.has("inpaint_color_guidance") && body["inpaint_color_guidance"].t() != crow::json::type::Null)
            return "inpaint_color_guidance is not supported; use the independent inpaint_color_guide RGBA input";
        if (body.has("inpaint_color_guide") && body["inpaint_color_guide"].t() != crow::json::type::Null) {
            const auto& guide = body["inpaint_color_guide"];
            if (guide.t() != crow::json::type::String || std::string(guide.s()).empty())
                return "inpaint_color_guide must be a nonempty encoded RGBA image";
            if (!body.has("init_images") || body["init_images"].t() != crow::json::type::List ||
                body["init_images"].size() != 1 || body["init_images"][0].t() != crow::json::type::String ||
                std::string(body["init_images"][0].s()).empty())
                return "inpaint_color_guide requires one original init_images image";
            if (body.has("denoising_strength")) {
                const auto& strength = body["denoising_strength"];
                if (strength.t() != crow::json::type::Number || !std::isfinite(strength.d()) ||
                    strength.d() <= 0 || strength.d() >= 1)
                    return "Advanced Inpaint requires denoising_strength strictly between 0 and 1";
            }
        }
    }
    if (masked_image && body.has("mask_blur")) {
        const auto& value = body["mask_blur"];
        if (value.t() != crow::json::type::Number || !std::isfinite(value.d()) ||
            value.d() < 0 || value.d() > kMaxMaskBlur) {
            return "mask_blur must be a number between 0 and 64";
        }
    }
    if (masked_image && body.has("inpainting_mask_invert")) {
        const auto& value = body["inpainting_mask_invert"];
        const bool boolean = value.t() == crow::json::type::True || value.t() == crow::json::type::False;
        const bool integer = value.t() == crow::json::type::Number && std::isfinite(value.d()) &&
                             std::trunc(value.d()) == value.d();
        if (!boolean && !integer) return "inpainting_mask_invert must be a boolean or integer";
    }
    return {};
}

inline bool parse_mask_invert(const crow::json::rvalue& body) {
    if (!body.has("inpainting_mask_invert")) return false;
    const auto& value = body["inpainting_mask_invert"];
    if (value.t() == crow::json::type::True) return true;
    if (value.t() == crow::json::type::False) return false;
    return value.d() != 0;
}

inline std::optional<float> parse_distilled_guidance(const crow::json::rvalue& body) {
    if (!body.has("distilled_cfg_scale")) return std::nullopt;
    return static_cast<float>(body["distilled_cfg_scale"].d());
}

inline void apply_distilled_guidance(sd_sample_params_t& sampling, std::optional<float> requested) {
    if (requested) sampling.guidance.distilled_guidance = *requested;
}

#endif
