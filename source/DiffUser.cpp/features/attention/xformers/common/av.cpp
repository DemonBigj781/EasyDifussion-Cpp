#include "xformers.hpp"

namespace edcpp::api::attention::xformers {

bool av(Backend backend, const AttentionRequest& request, const ScoreBuffer& probabilities) {
    const Translation* translation = translation_for(backend);
    return translation != nullptr && translation->av != nullptr && translation->av(request, probabilities);
}

} // namespace edcpp::api::attention::xformers
