#ifndef SDKIT_NATIVE_PROMPT_VALIDATION_H
#define SDKIT_NATIVE_PROMPT_VALIDATION_H

#include "stable-diffusion.h"

// Narrow application adapter: parser and C++ exceptions stay inside the runtime.
// No tensor handles, resource ownership or C++ containers cross this boundary.
enum sdkit_prompt_validation_status {
    SDKIT_PROMPT_VALID = 0,
    SDKIT_PROMPT_INVALID_WEIGHT,
    SDKIT_PROMPT_WEIGHT_OUT_OF_RANGE,
    SDKIT_PROMPT_NONFINITE_WEIGHT,
    SDKIT_PROMPT_VALIDATION_FAILED
};
#ifdef __cplusplus
extern "C" {
#endif
SD_API enum sdkit_prompt_validation_status sdkit_validate_prompt_attention(const char* prompt, size_t length);
#ifdef __cplusplus
}
#endif
#endif
