#ifndef __SD_RUNTIME_CUSTOM_SIGMA_STEPS_H__
#define __SD_RUNTIME_CUSTOM_SIGMA_STEPS_H__

#include "core/util.h"

static inline void resolve_custom_sigma_steps(int sigma_count, int& sample_steps,
                                             int& high_noise_sample_steps, int& total_steps) {
    if (total_steps != sigma_count - 1) {
        total_steps = sigma_count - 1;
        LOG_WARN("total_steps != custom_sigmas_count - 1, set total_steps to %d", total_steps);
        if (sample_steps > total_steps || (high_noise_sample_steps <= 0 && sample_steps != total_steps)) {
            sample_steps = total_steps;
            LOG_WARN("total_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
        }
        if (high_noise_sample_steps > 0) {
            high_noise_sample_steps = total_steps - sample_steps;
            LOG_WARN("total_steps != custom_sigmas_count - 1, set high_noise_sample_steps to %d", high_noise_sample_steps);
        }
    }
}

#endif  // __SD_RUNTIME_CUSTOM_SIGMA_STEPS_H__
