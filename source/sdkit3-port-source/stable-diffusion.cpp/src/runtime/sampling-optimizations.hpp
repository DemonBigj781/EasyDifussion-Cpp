#ifndef SD_RUNTIME_SAMPLING_OPTIMIZATIONS_HPP
#define SD_RUNTIME_SAMPLING_OPTIMIZATIONS_HPP
#include <cmath>
#include <optional>
#include "core/util.h"

struct SamplingOptimizations {
    float tome_ratio = 0.f;
    float negative_guidance_min_sigma = 0.f;

    bool cache_compatible(sd_cache_mode_t mode) const {
        return negative_guidance_min_sigma == 0.f || mode == SD_CACHE_DISABLED;
    }

    // Conservative NGMS policy: alternate eligible evaluations, matching the
    // usual minimum-sigma behavior rather than dropping all low-sigma CFG.
    bool skip_negative(float sigma, int step) const {
        return negative_guidance_min_sigma > 0.f && std::isfinite(sigma) && sigma >= 0.f &&
               sigma < negative_guidance_min_sigma && step > 0 && step % 2 == 0;
    }

    static std::optional<SamplingOptimizations> parse(const char* args) {
        SamplingOptimizations result;
        for (const auto& [key, value] : parse_key_value_args(args, "sampling optimization")) {
            if (key != "tome_ratio" && key != "negative_guidance_min_sigma") continue;
            float number;
            if (!parse_strict_float(value, number) || !std::isfinite(number) || number < 0.f ||
                (key == "tome_ratio" && number > .5f)) {
                LOG_ERROR("Invalid %s: expected a finite non-negative value%s", key.c_str(),
                          key == "tome_ratio" ? " no greater than 0.5" : "");
                return std::nullopt;
            }
            if (key == "tome_ratio") result.tome_ratio = number;
            else result.negative_guidance_min_sigma = number;
        }
        return result;
    }
};
#endif
