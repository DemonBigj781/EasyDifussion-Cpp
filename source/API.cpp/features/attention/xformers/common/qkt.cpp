#include "xformers.hpp"

namespace edcpp::api::attention::xformers {

bool qkt(Backend backend, const AttentionRequest& request, ScoreBuffer& scores) {
    const Translation* translation = translation_for(backend);
    return translation != nullptr && translation->qkt != nullptr && translation->qkt(request, scores);
}

} // namespace edcpp::api::attention::xformers
