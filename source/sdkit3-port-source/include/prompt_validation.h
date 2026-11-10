#ifndef SDKIT_PROMPT_VALIDATION_H
#define SDKIT_PROMPT_VALIDATION_H

#include "crow/json.h"
#include <string>

// Empty on success; no request mutation or model/resource access.
std::string validate_generation_prompts(const crow::json::rvalue& request);

#endif
