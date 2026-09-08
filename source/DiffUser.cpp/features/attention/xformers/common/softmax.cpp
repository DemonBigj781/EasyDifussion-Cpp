#include "xformers.hpp"

namespace edcpp::api::attention::xformers {

bool softmax(Backend backend, const AttentionRequest& request, ScoreBuffer& scores) {
    const Translation* translation = translation_for(backend);
    return translation != nullptr && translation->softmax != nullptr && translation->softmax(request, scores);
}

} // namespace edcpp::api::attention::xformers
