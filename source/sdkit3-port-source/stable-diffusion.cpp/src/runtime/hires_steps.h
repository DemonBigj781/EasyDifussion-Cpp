#ifndef __SD_RUNTIME_HIRES_STEPS_H__
#define __SD_RUNTIME_HIRES_STEPS_H__

#include <cstddef>

static inline int hires_sigma_start(int scheduler_steps, int effective_steps) {
    size_t t_enc = static_cast<size_t>(effective_steps - 1);
    if (t_enc >= static_cast<size_t>(scheduler_steps)) {
        t_enc = static_cast<size_t>(scheduler_steps) - 1;
    }
    return scheduler_steps - static_cast<int>(t_enc) - 1;
}

#endif  // __SD_RUNTIME_HIRES_STEPS_H__
