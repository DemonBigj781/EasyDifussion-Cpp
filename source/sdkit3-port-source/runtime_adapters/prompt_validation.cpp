#include "native_prompt_validation.h"
#include "core/util.h"
#include <cmath>
#include <stdexcept>

extern "C" enum sdkit_prompt_validation_status sdkit_validate_prompt_attention(const char* prompt, size_t length) {
    if (!prompt) return SDKIT_PROMPT_VALIDATION_FAILED;
    try {
        for (const auto& segment : parse_prompt_attention(std::string(prompt, length))) {
            if (!std::isfinite(segment.second)) return SDKIT_PROMPT_NONFINITE_WEIGHT;
        }
        return SDKIT_PROMPT_VALID;
    } catch (const std::invalid_argument&) {
        return SDKIT_PROMPT_INVALID_WEIGHT;
    } catch (const std::out_of_range&) {
        return SDKIT_PROMPT_WEIGHT_OUT_OF_RANGE;
    } catch (...) {
        return SDKIT_PROMPT_VALIDATION_FAILED;
    }
}
