#include "prompt_validation.h"
#include "native_prompt_validation.h"
#include <stdexcept>

// Apply Theory's finite-weight safety rule through the active application's
// parser. Do not substitute Theory shorthand/tag semantics or rewrite prompts.
std::string validate_generation_prompts(const crow::json::rvalue& request) {
    if (request.t() != crow::json::type::Object) {
        return "Generation request must be a JSON object";
    }
    for (const char* field : {"prompt", "negative_prompt"}) {
        if (!request.has(field)) continue;
        if (request[field].t() != crow::json::type::String) {
            return std::string(field) + " must be a string";
        }
        const std::string prompt = std::string(request[field].s());
        switch (sdkit_validate_prompt_attention(prompt.data(), prompt.size())) {
            case SDKIT_PROMPT_VALID:
                break;
            case SDKIT_PROMPT_INVALID_WEIGHT:
                return std::string(field) + " contains an invalid numeric attention weight";
            case SDKIT_PROMPT_WEIGHT_OUT_OF_RANGE:
                return std::string(field) + " contains an attention weight outside the supported numeric range";
            case SDKIT_PROMPT_NONFINITE_WEIGHT:
                return std::string(field) + " contains a non-finite combined attention weight";
            default:
                throw std::runtime_error("Prompt validation could not complete");
        }
    }
    return {};
}
