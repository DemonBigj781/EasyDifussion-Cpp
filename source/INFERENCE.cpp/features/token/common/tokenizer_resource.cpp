#include "features/token/common/tokenizer_resource.hpp"

#include "features/token/common/t5_unigram.hpp"
#include "features/token/common/t5_vocabulary.hpp"

#include <stdexcept>
#include <utility>

namespace edcpp::inference {
namespace {

std::string resource_error(const TokenizerResourceView& resource,
                           const std::string& message) {
    if (resource.identifier.empty()) return message;
    return "tokenizer resource '" + std::string(resource.identifier) + "': " + message;
}

} // namespace

TokenizerResourceLoadResult load_tokenizer_resource(const TokenizerResourceView& resource) {
    TokenizerResourceLoadResult result;
    if (resource.serialized.empty()) {
        result.message = resource_error(resource, "serialized content is empty");
        return result;
    }

    try {
        switch (resource.format) {
        case TokenizerResourceFormat::clip_merges: {
            auto tokenizer = std::make_unique<ClipTokenizer>(
                std::string(resource.serialized), resource.maximum_tokens,
                resource.padding_token);
            result.special_tokens.unknown = tokenizer->end_token();
            result.special_tokens.beginning = tokenizer->beginning_token();
            result.special_tokens.end = tokenizer->end_token();
            result.special_tokens.padding = tokenizer->padding_token();
            result.special_tokens.adds_beginning = true;
            result.special_tokens.adds_end = true;
            result.tokenizer = std::move(tokenizer);
            break;
        }
        case TokenizerResourceFormat::hugging_face_unigram_json: {
            auto loaded = load_t5_unigram_json(resource.serialized);
            if (!loaded.ok) {
                result.message = resource_error(resource, loaded.message);
                return result;
            }
            loaded.config.maximum_tokens = resource.maximum_tokens;
            if (resource.padding_token >= 0)
                loaded.config.padding_token = resource.padding_token;

            result.special_tokens.unknown = loaded.config.unknown_token;
            result.special_tokens.beginning = loaded.config.beginning_token;
            result.special_tokens.end = loaded.config.end_token;
            result.special_tokens.padding = loaded.config.padding_token;
            result.special_tokens.adds_beginning = loaded.config.add_beginning;
            result.special_tokens.adds_end = loaded.config.add_end;
            result.tokenizer = std::make_unique<T5UnigramTokenizer>(
                std::move(loaded.pieces), std::move(loaded.config));
            break;
        }
        default:
            result.message = resource_error(resource, "format is not supported");
            return result;
        }
    } catch (const std::invalid_argument& error) {
        result.message = resource_error(resource, error.what());
        return result;
    }

    result.ok = true;
    return result;
}

} // namespace edcpp::inference
