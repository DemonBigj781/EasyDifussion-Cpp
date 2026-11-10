#include "translate.hpp"

namespace edcpp::api::attention::xformers::cpu::translation {

bool apply_mask_and_bias(
    const AttentionRequest& request, ScoreBuffer& scores) {
    native::NativeScoreBuffer native_scores = translate_scores(scores);
    if (!native::apply_mask_and_bias(
            translate_request(request), native_scores)) {
        return false;
    }
    translate_scores(std::move(native_scores), scores);
    return true;
}

} // namespace edcpp::api::attention::xformers::cpu::translation
