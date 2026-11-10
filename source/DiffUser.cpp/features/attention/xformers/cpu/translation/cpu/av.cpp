#include "translate.hpp"

namespace edcpp::api::attention::xformers::cpu::translation {

bool av(
    const AttentionRequest& request,
    const ScoreBuffer& probabilities) {
    return native::av(
        translate_request(request), translate_scores(probabilities));
}

} // namespace edcpp::api::attention::xformers::cpu::translation
