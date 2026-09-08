#include "xformers.hpp"

namespace edcpp::api::attention::xformers {

bool apply_mask_and_bias(Backend backend, const AttentionRequest& request, ScoreBuffer& scores) {
    const Translation* translation = translation_for(backend);
    return translation != nullptr && translation->mask != nullptr && translation->mask(request, scores);
}

} // namespace edcpp::api::attention::xformers
