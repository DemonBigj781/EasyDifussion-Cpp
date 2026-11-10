#pragma once

#include "features/token/common/t5_unigram.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace edcpp::inference {

struct T5VocabularyLoadResult {
    bool ok = false;
    std::string message{};
    std::vector<UnigramPiece> pieces{};
    T5UnigramConfig config{};
};

T5VocabularyLoadResult load_t5_unigram_json(std::string_view serialized);

} // namespace edcpp::inference
