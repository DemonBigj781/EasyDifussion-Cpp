#pragma once

#include "inference/types.hpp"

#include <cstdint>
#include <vector>

namespace edcpp::inference {

enum class Stage : std::uint8_t {
    tokenize,
    condition,
    prepare_input,
    initialize_latent,
    denoise,
    decode,
    assemble_result,
};

struct ExecutionPlan {
    GenerationRequest request{};
    std::vector<Stage> stages{};
};

ValidationResult validate(const GenerationRequest& request);
ExecutionPlan make_plan(const GenerationRequest& request);

} // namespace edcpp::inference
