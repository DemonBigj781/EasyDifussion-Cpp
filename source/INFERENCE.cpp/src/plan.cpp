#include "inference/plan.hpp"

#include <cmath>
#include <stdexcept>

namespace edcpp::inference {

ValidationResult validate(const GenerationRequest& request) {
    if (request.model == nullptr || !request.model->loaded())
        return {false, "a loaded DiffUser Common model resource is required"};
    if (request.prompt.empty()) return {false, "a prompt is required"};
    if (request.width == 0 || request.height == 0) return {false, "dimensions must be non-zero"};
    if ((request.width % 8) != 0 || (request.height % 8) != 0)
        return {false, "dimensions must be divisible by eight"};
    if (request.steps == 0) return {false, "at least one denoising step is required"};
    if (request.batch == 0) return {false, "batch must be non-zero"};
    if (request.media == MediaKind::image && request.frames != 1)
        return {false, "image generation requires exactly one frame"};
    if (request.media == MediaKind::video && request.frames < 2)
        return {false, "video generation requires at least two frames"};
    if (!std::isfinite(request.guidance) || request.guidance < 0.0f)
        return {false, "guidance must be finite and non-negative"};
    if (!std::isfinite(request.strength) || request.strength < 0.0f || request.strength > 1.0f)
        return {false, "strength must be finite and within zero and one"};
    return {true, {}};
}

ExecutionPlan make_plan(const GenerationRequest& request) {
    const auto checked = validate(request);
    if (!checked.ok) throw std::invalid_argument(checked.message);

    ExecutionPlan plan;
    plan.request = request;
    plan.stages = {Stage::tokenize, Stage::condition};
    plan.stages.push_back(Stage::initialize_latent);
    plan.stages.push_back(Stage::denoise);
    plan.stages.push_back(Stage::decode);
    plan.stages.push_back(Stage::assemble_result);
    return plan;
}

} // namespace edcpp::inference
