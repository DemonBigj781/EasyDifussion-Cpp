// Native Sprite-GPT text-to-image inference sidecar.

#include <ATen/Parallel.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/xpu.h>

#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/utf16.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "json.hpp"
#include "safetensors.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace sprite_gpt {

static constexpr int kContextLength = 32;
static constexpr int kImageSize     = 64;

struct Options {
    std::string unet_model;
    std::string clip_model;
    std::string tokenizer_dir;
    std::string prompt;
    std::string negative_prompt;
    std::string output;
    std::string dataset;
    std::string trigger;
    std::string device = "cuda";
    int steps          = 50;
    int batch_size     = 1;
    int warmup         = 100;
    int log_every      = 10;
    int save_every     = 500;
    float learning_rate = 2.0e-4f;
    float condition_drop = 0.1f;
    float guidance     = 5.0f;
    int64_t seed       = 42;
    int threads        = 0;
    bool tokenize_only = false;
    bool progress = false;
    bool train_mode = false;
};

struct Tokenized {
    std::array<int64_t, kContextLength> ids{};
    std::array<int64_t, kContextLength> mask{};
};

enum class TokenClass {
    LETTER,
    DIGIT,
    OTHER,
};

class CLIPTokenizer {
   private:
    std::unordered_map<std::string, int64_t> vocabulary_;
    std::unordered_map<std::string, int> merge_ranks_;
    std::unordered_map<std::string, std::vector<int64_t>> bpe_cache_;
    std::array<std::string, 256> byte_encoder_;
    int64_t bos_id_ = -1;
    int64_t eos_id_ = -1;

    static std::string pair_key(const std::string& first, const std::string& second) {
        return first + "\n" + second;
    }

    static std::string utf8_codepoint(int codepoint) {
        std::string value;
        if (codepoint <= 0x7f) {
            value.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            value.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            value.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            value.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            value.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            value.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            value.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            value.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            value.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            value.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
        return value;
    }

    static std::vector<std::string> utf8_symbols(const std::string& value) {
        std::vector<std::string> symbols;
        for (size_t i = 0; i < value.size();) {
            const unsigned char lead = static_cast<unsigned char>(value[i]);
            size_t length            = 1;
            if ((lead & 0xe0) == 0xc0) length = 2;
            else if ((lead & 0xf0) == 0xe0)
                length = 3;
            else if ((lead & 0xf8) == 0xf0)
                length = 4;
            length = std::min(length, value.size() - i);
            symbols.emplace_back(value.substr(i, length));
            i += length;
        }
        return symbols;
    }

    static size_t utf8_symbol_length(const std::string& value, size_t offset) {
        const unsigned char lead = static_cast<unsigned char>(value[offset]);
        if ((lead & 0xe0) == 0xc0) return std::min<size_t>(2, value.size() - offset);
        if ((lead & 0xf0) == 0xe0) return std::min<size_t>(3, value.size() - offset);
        if ((lead & 0xf8) == 0xf0) return std::min<size_t>(4, value.size() - offset);
        return 1;
    }

    static char32_t decode_codepoint(const std::string& value, size_t offset) {
        const unsigned char lead = static_cast<unsigned char>(value[offset]);
        if (lead < 0x80) return lead;
        const size_t length = utf8_symbol_length(value, offset);
        if (length == 2) {
            return ((lead & 0x1f) << 6) |
                   (static_cast<unsigned char>(value[offset + 1]) & 0x3f);
        }
        if (length == 3) {
            return ((lead & 0x0f) << 12) |
                   ((static_cast<unsigned char>(value[offset + 1]) & 0x3f) << 6) |
                   (static_cast<unsigned char>(value[offset + 2]) & 0x3f);
        }
        if (length == 4) {
            return ((lead & 0x07) << 18) |
                   ((static_cast<unsigned char>(value[offset + 1]) & 0x3f) << 12) |
                   ((static_cast<unsigned char>(value[offset + 2]) & 0x3f) << 6) |
                   (static_cast<unsigned char>(value[offset + 3]) & 0x3f);
        }
        return lead;
    }

    static TokenClass classify(const std::string& value, size_t offset) {
        const char32_t codepoint = decode_codepoint(value, offset);
        if (u_isalpha(static_cast<UChar32>(codepoint))) return TokenClass::LETTER;
        const int8_t category = u_charType(static_cast<UChar32>(codepoint));
        if (category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
            category == U_OTHER_NUMBER) {
            return TokenClass::DIGIT;
        }
        return TokenClass::OTHER;
    }

    static void check_icu(UErrorCode status, const char* operation) {
        if (U_FAILURE(status)) {
            throw std::runtime_error(std::string(operation) + ": " + u_errorName(status));
        }
    }

    static std::string normalize(const std::string& text) {
        if (text.empty()) return {};
        if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("Prompt is too long for ICU normalization");
        }

        UErrorCode status = U_ZERO_ERROR;
        int32_t utf16_length;
        u_strFromUTF8(nullptr, 0, &utf16_length, text.data(), static_cast<int32_t>(text.size()),
                      &status);
        if (status != U_BUFFER_OVERFLOW_ERROR) check_icu(status, "Could not decode prompt UTF-8");
        status = U_ZERO_ERROR;
        std::vector<UChar> utf16(utf16_length);
        u_strFromUTF8(utf16.data(), utf16_length, &utf16_length, text.data(),
                      static_cast<int32_t>(text.size()), &status);
        check_icu(status, "Could not decode prompt UTF-8");

        status = U_ZERO_ERROR;
        const UNormalizer2* nfc = unorm2_getNFCInstance(&status);
        check_icu(status, "Could not initialize ICU NFC normalization");
        int32_t normalized_length =
            unorm2_normalize(nfc, utf16.data(), utf16_length, nullptr, 0, &status);
        if (status != U_BUFFER_OVERFLOW_ERROR) check_icu(status, "Could not normalize prompt");
        status = U_ZERO_ERROR;
        std::vector<UChar> normalized(normalized_length);
        unorm2_normalize(nfc, utf16.data(), utf16_length, normalized.data(), normalized_length,
                         &status);
        check_icu(status, "Could not normalize prompt");

        std::vector<UChar> lower;
        lower.reserve(normalized.size());
        for (int32_t index = 0; index < normalized_length;) {
            const int32_t character_length =
                U16_IS_LEAD(normalized[index]) && index + 1 < normalized_length &&
                        U16_IS_TRAIL(normalized[index + 1])
                    ? 2
                    : 1;
            status = U_ZERO_ERROR;
            const int32_t item_length =
                u_strToLower(nullptr, 0, normalized.data() + index, character_length, "root", &status);
            if (status != U_BUFFER_OVERFLOW_ERROR) check_icu(status, "Could not lowercase prompt");
            status = U_ZERO_ERROR;
            std::vector<UChar> item(item_length);
            u_strToLower(item.data(), item_length, normalized.data() + index, character_length,
                         "root", &status);
            check_icu(status, "Could not lowercase prompt");
            lower.insert(lower.end(), item.begin(), item.end());
            index += character_length;
        }

        status = U_ZERO_ERROR;
        int32_t utf8_length;
        u_strToUTF8(nullptr, 0, &utf8_length, lower.data(), static_cast<int32_t>(lower.size()), &status);
        if (status != U_BUFFER_OVERFLOW_ERROR) check_icu(status, "Could not encode normalized prompt");
        status = U_ZERO_ERROR;
        std::string normalized_utf8(utf8_length, '\0');
        u_strToUTF8(normalized_utf8.data(), utf8_length, &utf8_length, lower.data(),
                    static_cast<int32_t>(lower.size()), &status);
        check_icu(status, "Could not encode normalized prompt");

        std::string value;
        value.reserve(normalized_utf8.size());
        bool pending_space = false;
        for (size_t offset = 0; offset < normalized_utf8.size();) {
            const size_t length = utf8_symbol_length(normalized_utf8, offset);
            const char32_t codepoint = decode_codepoint(normalized_utf8, offset);
            if (u_isUWhiteSpace(static_cast<UChar32>(codepoint))) {
                pending_space = !value.empty();
                offset += length;
                continue;
            }
            if (pending_space) {
                value.push_back(' ');
                pending_space = false;
            }
            value.append(normalized_utf8, offset, length);
            offset += length;
        }
        return value;
    }

    static std::vector<std::string> pretokenize(const std::string& text) {
        const std::string value = normalize(text);
        std::vector<std::string> tokens;
        for (size_t i = 0; i < value.size();) {
            if (value[i] == ' ') {
                ++i;
                continue;
            }

            size_t end = i;
            if (value[i] == '\'') {
                static const std::array<std::string, 7> contractions = {
                    "'re", "'ve", "'ll", "'s", "'t", "'m", "'d",
                };
                for (const auto& contraction : contractions) {
                    if (value.compare(i, contraction.size(), contraction) == 0) {
                        end = i + contraction.size();
                        break;
                    }
                }
            }
            if (end == i) {
                const TokenClass token_class = classify(value, i);
                end = i + utf8_symbol_length(value, i);
                while (token_class != TokenClass::DIGIT && end < value.size() && value[end] != ' ' &&
                       classify(value, end) == token_class) {
                    if (token_class == TokenClass::LETTER && value[end] == '\'') break;
                    end += utf8_symbol_length(value, end);
                }
            }

            tokens.push_back(value.substr(i, end - i));
            i = end;
        }
        return tokens;
    }

    std::vector<int64_t> bpe(const std::string& raw_token) {
        std::string encoded;
        for (unsigned char byte : raw_token) encoded += byte_encoder_[byte];

        auto cached = bpe_cache_.find(encoded);
        if (cached != bpe_cache_.end()) return cached->second;

        std::vector<std::string> symbols = utf8_symbols(encoded);
        if (symbols.empty()) return {};
        symbols.back() += "</w>";

        while (symbols.size() > 1) {
            int best_rank   = std::numeric_limits<int>::max();
            size_t best_pos = symbols.size();
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                auto rank = merge_ranks_.find(pair_key(symbols[i], symbols[i + 1]));
                if (rank != merge_ranks_.end() && rank->second < best_rank) {
                    best_rank = rank->second;
                    best_pos  = i;
                }
            }
            if (best_pos == symbols.size()) break;

            const std::string first  = symbols[best_pos];
            const std::string second = symbols[best_pos + 1];
            std::vector<std::string> merged;
            merged.reserve(symbols.size() - 1);
            for (size_t i = 0; i < symbols.size();) {
                if (i + 1 < symbols.size() && symbols[i] == first && symbols[i + 1] == second) {
                    merged.push_back(first + second);
                    i += 2;
                } else {
                    merged.push_back(symbols[i]);
                    ++i;
                }
            }
            symbols = std::move(merged);
        }

        std::vector<int64_t> ids;
        ids.reserve(symbols.size());
        for (const auto& symbol : symbols) {
            auto item = vocabulary_.find(symbol);
            if (item == vocabulary_.end()) {
                throw std::runtime_error("CLIP vocabulary is missing a BPE symbol");
            }
            ids.push_back(item->second);
        }
        bpe_cache_.emplace(encoded, ids);
        return ids;
    }

   public:
    explicit CLIPTokenizer(const std::string& directory) {
        std::ifstream vocab_stream(std::filesystem::path(directory) / "vocab.json");
        if (!vocab_stream) throw std::runtime_error("Could not open CLIP vocab.json");
        nlohmann::json vocab_json;
        vocab_stream >> vocab_json;
        for (auto item = vocab_json.begin(); item != vocab_json.end(); ++item) {
            vocabulary_.emplace(item.key(), item.value().get<int64_t>());
        }

        auto bos = vocabulary_.find("<|startoftext|>");
        auto eos = vocabulary_.find("<|endoftext|>");
        if (bos == vocabulary_.end() || eos == vocabulary_.end()) {
            throw std::runtime_error("CLIP special tokens are missing from vocab.json");
        }
        bos_id_ = bos->second;
        eos_id_ = eos->second;

        std::ifstream merges_stream(std::filesystem::path(directory) / "merges.txt");
        if (!merges_stream) throw std::runtime_error("Could not open CLIP merges.txt");
        std::string line;
        int rank = 0;
        while (std::getline(merges_stream, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream parts(line);
            std::string first;
            std::string second;
            if (parts >> first >> second) merge_ranks_.emplace(pair_key(first, second), rank++);
        }

        std::vector<int> bytes;
        for (int value = 33; value <= 126; ++value) bytes.push_back(value);
        for (int value = 161; value <= 172; ++value) bytes.push_back(value);
        for (int value = 174; value <= 255; ++value) bytes.push_back(value);
        std::vector<int> codepoints = bytes;
        std::array<bool, 256> included{};
        for (int value : bytes) included[value] = true;
        int extra = 0;
        for (int value = 0; value < 256; ++value) {
            if (!included[value]) {
                bytes.push_back(value);
                codepoints.push_back(256 + extra++);
            }
        }
        for (size_t i = 0; i < bytes.size(); ++i) {
            byte_encoder_[bytes[i]] = utf8_codepoint(codepoints[i]);
        }
    }

    Tokenized tokenize(const std::string& prompt) {
        Tokenized result;
        result.ids.fill(eos_id_);
        result.mask.fill(0);
        size_t position          = 0;
        result.ids[position]     = bos_id_;
        result.mask[position++]  = 1;

        for (const auto& token : pretokenize(prompt)) {
            for (int64_t id : bpe(token)) {
                if (position >= kContextLength - 1) break;
                result.ids[position]  = id;
                result.mask[position] = 1;
                ++position;
            }
            if (position >= kContextLength - 1) break;
        }
        result.ids[position]  = eos_id_;
        result.mask[position] = 1;
        return result;
    }
};

[[noreturn]] static void usage(const char* argv0, const std::string& error = {}) {
    if (!error.empty()) std::cerr << error << "\n\n";
    std::cerr
        << "Usage: " << argv0
        << " --unet sprite-gpt-unet.ts --clip clip-text.ts --tokenizer tokenizer-dir"
           " --prompt TEXT --output image.png [--steps 50] [--guidance 5]"
           " [--seed 42] [--device cuda|xpu|cpu] [--threads N] [--negative-prompt TEXT] [--progress]\n"
           "       "
        << argv0 << " --train --unet sprite-gpt-unet.ts --clip clip-text.ts"
           " --tokenizer tokenizer-dir --dataset images/ --output trained-unet.ts"
           " [--steps N] [--batch-size N] [--device xpu|cuda|cpu] [--trigger TEXT]\n"
           "       "
        << argv0 << " --tokenizer tokenizer-dir --prompt TEXT --tokenize-only\n";
    std::exit(error.empty() ? 0 : 2);
}

static Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&](const char* name) {
            if (i + 1 >= argc) usage(argv[0], std::string("Missing value for ") + name);
            return std::string(argv[++i]);
        };
        if (argument == "--unet") options.unet_model = value("--unet");
        else if (argument == "--clip")
            options.clip_model = value("--clip");
        else if (argument == "--tokenizer")
            options.tokenizer_dir = value("--tokenizer");
        else if (argument == "--train") options.train_mode = true;
        else if (argument == "--dataset") options.dataset = value("--dataset");
        else if (argument == "--trigger") options.trigger = value("--trigger");
        else if (argument == "--prompt")
            options.prompt = value("--prompt");
        else if (argument == "--negative-prompt")
            options.negative_prompt = value("--negative-prompt");
        else if (argument == "--progress")
            options.progress = true;
        else if (argument == "--output")
            options.output = value("--output");
        else if (argument == "--steps")
            options.steps = std::stoi(value("--steps"));
        else if (argument == "--batch-size")
            options.batch_size = std::stoi(value("--batch-size"));
        else if (argument == "--warmup")
            options.warmup = std::stoi(value("--warmup"));
        else if (argument == "--log-every")
            options.log_every = std::stoi(value("--log-every"));
        else if (argument == "--save-every")
            options.save_every = std::stoi(value("--save-every"));
        else if (argument == "--learning-rate")
            options.learning_rate = std::stof(value("--learning-rate"));
        else if (argument == "--condition-drop")
            options.condition_drop = std::stof(value("--condition-drop"));
        else if (argument == "--guidance")
            options.guidance = std::stof(value("--guidance"));
        else if (argument == "--seed")
            options.seed = std::stoll(value("--seed"));
        else if (argument == "--device")
            options.device = value("--device");
        else if (argument == "--threads")
            options.threads = std::stoi(value("--threads"));
        else if (argument == "--tokenize-only")
            options.tokenize_only = true;
        else if (argument == "--help" || argument == "-h")
            usage(argv[0]);
        else
            usage(argv[0], "Unknown argument: " + argument);
    }

    if (options.tokenizer_dir.empty()) usage(argv[0], "--tokenizer is required");
    if (options.train_mode) {
        if (options.unet_model.empty() || options.clip_model.empty() || options.dataset.empty() ||
            options.output.empty()) {
            usage(argv[0], "training requires --unet, --clip, --dataset, and --output");
        }
        if (options.steps < 1 || options.batch_size < 1 || options.warmup < 0 ||
            options.log_every < 1 || options.save_every < 1 ||
            !std::isfinite(options.learning_rate) || options.learning_rate <= 0.f ||
            !std::isfinite(options.condition_drop) || options.condition_drop < 0.f ||
            options.condition_drop > 1.f) {
            usage(argv[0], "invalid training parameters");
        }
    } else if (options.prompt.empty()) {
        usage(argv[0], "--prompt is required");
    }
    if (!options.train_mode && !options.tokenize_only &&
        (options.unet_model.empty() || options.clip_model.empty() || options.output.empty())) {
        usage(argv[0], "--unet, --clip, and --output are required for generation");
    }
    if (!options.train_mode && (options.steps < 1 || options.steps > 1000))
        usage(argv[0], "--steps must be in [1,1000]");
    if (!std::isfinite(options.guidance) || options.guidance < 0.f || options.guidance > 100.f) {
        usage(argv[0], "--guidance must be finite and in [0,100]");
    }
    if (options.threads < 0) usage(argv[0], "--threads cannot be negative");
    return options;
}

static std::string json_escape(const std::string& value) {
    std::string output;
    output.reserve(value.size() + 8);
    for (unsigned char byte : value) {
        switch (byte) {
            case '\\': output += "\\\\"; break;
            case '"': output += "\\\""; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (byte < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    output += "\\u00";
                    output += hex[byte >> 4];
                    output += hex[byte & 0xf];
                } else {
                    output.push_back(static_cast<char>(byte));
                }
        }
    }
    return output;
}

static void print_tokens(const Tokenized& tokens) {
    std::cout << "{\"ids\":[";
    for (size_t i = 0; i < tokens.ids.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << tokens.ids[i];
    }
    std::cout << "],\"mask\":[";
    for (size_t i = 0; i < tokens.mask.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << tokens.mask[i];
    }
    std::cout << "]}\n";
}

static torch::Tensor run_module(torch::jit::script::Module& module,
                                std::vector<torch::jit::IValue> inputs) {
    const torch::jit::IValue output = module.forward(std::move(inputs));
    if (!output.isTensor()) throw std::runtime_error("TorchScript module did not return a tensor");
    return output.toTensor();
}

struct TrainingImage {
    std::filesystem::path path;
    std::string caption;
};

static std::string trim(std::string value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

static std::vector<TrainingImage> load_training_images(const Options& options) {
    const std::filesystem::path root = std::filesystem::weakly_canonical(options.dataset);
    if (!std::filesystem::is_directory(root)) {
        throw std::runtime_error("Dataset directory does not exist: " + root.string());
    }
    std::vector<TrainingImage> images;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (extension != ".png" && extension != ".jpg" && extension != ".jpeg" &&
            extension != ".bmp" && extension != ".tga") continue;
        std::filesystem::path caption_path = entry.path();
        caption_path.replace_extension(".txt");
        std::ifstream caption_file(caption_path);
        if (!caption_file) continue;
        std::ostringstream text;
        text << caption_file.rdbuf();
        std::string caption = trim(text.str());
        if (!options.trigger.empty()) caption = options.trigger + (caption.empty() ? "" : ", " + caption);
        if (caption.empty()) continue;
        int width = 0, height = 0, channels = 0;
        if (!stbi_info(entry.path().string().c_str(), &width, &height, &channels)) continue;
        images.push_back({entry.path(), std::move(caption)});
    }
    std::sort(images.begin(), images.end(), [](const TrainingImage& a, const TrainingImage& b) {
        return a.path.generic_string() < b.path.generic_string();
    });
    if (images.empty()) {
        throw std::runtime_error("Dataset has no supported images with same-basename .txt captions");
    }
    return images;
}

struct EncodedCaptionBatch {
    torch::Tensor context;
    torch::Tensor mask;
};

static EncodedCaptionBatch encode_captions(torch::jit::script::Module& clip,
                                          CLIPTokenizer& tokenizer,
                                          const std::vector<TrainingImage>& images,
                                          const std::vector<size_t>& indices,
                                          torch::Device device) {
    std::vector<int64_t> ids(indices.size() * kContextLength);
    std::vector<int64_t> masks(indices.size() * kContextLength);
    for (size_t batch = 0; batch < indices.size(); ++batch) {
        const Tokenized tokens = tokenizer.tokenize(images[indices[batch]].caption);
        std::copy(tokens.ids.begin(), tokens.ids.end(), ids.begin() + batch * kContextLength);
        std::copy(tokens.mask.begin(), tokens.mask.end(), masks.begin() + batch * kContextLength);
    }
    auto id_tensor = torch::from_blob(ids.data(),
        {static_cast<int64_t>(indices.size()), kContextLength},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    auto mask_tensor = torch::from_blob(masks.data(),
        {static_cast<int64_t>(indices.size()), kContextLength},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    torch::NoGradGuard no_grad;
    auto context = run_module(clip, {id_tensor, mask_tensor}).to(device);
    return {context, mask_tensor.to(device, torch::kBool)};
}

static torch::Tensor load_image_batch(const std::vector<TrainingImage>& images,
                                      const std::vector<size_t>& indices,
                                      std::mt19937& random,
                                      torch::Device device) {
    const size_t plane = static_cast<size_t>(kImageSize) * kImageSize;
    std::vector<float> batch(indices.size() * 3 * plane);
    std::uniform_int_distribution<int> flip_distribution(0, 1);
    std::vector<uint8_t> resized(plane * 3);
    for (size_t batch_index = 0; batch_index < indices.size(); ++batch_index) {
        const std::string path = images[indices[batch_index]].path.string();
        int width = 0, height = 0, channels = 0;
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
            stbi_load(path.c_str(), &width, &height, &channels, 3), stbi_image_free);
        if (!decoded) throw std::runtime_error("Could not decode image: " + path);
        if (!stbir_resize_uint8(decoded.get(), width, height, 0, resized.data(),
                                kImageSize, kImageSize, 0, 3)) {
            throw std::runtime_error("Could not resize image: " + path);
        }
        const bool flip = flip_distribution(random) != 0;
        for (int y = 0; y < kImageSize; ++y) {
            for (int x = 0; x < kImageSize; ++x) {
                const int sx = flip ? kImageSize - 1 - x : x;
                const size_t pixel = (static_cast<size_t>(y) * kImageSize + sx) * 3;
                const size_t out_pixel = static_cast<size_t>(y) * kImageSize + x;
                for (int channel = 0; channel < 3; ++channel) {
                    batch[(batch_index * 3 + channel) * plane + out_pixel] =
                        static_cast<float>(resized[pixel + channel]) / 127.5f - 1.f;
                }
            }
        }
    }
    auto tensor = torch::from_blob(batch.data(),
        {static_cast<int64_t>(indices.size()), 3, kImageSize, kImageSize},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return tensor.to(device);
}

static void save_training_model(torch::jit::script::Module& unet,
                                const std::filesystem::path& output,
                                int step, int total_steps) {
    const std::filesystem::path parent = output.parent_path().empty() ? "." : output.parent_path();
    std::filesystem::create_directories(parent);
    const std::filesystem::path module_path = step == total_steps
        ? output
        : parent / (output.stem().string() + "_step_" + std::to_string(step) + output.extension().string());
    unet.save(module_path.string());
    std::unordered_map<std::string, torch::Tensor> tensors;
    for (const auto& item : unet.named_parameters(/*recurse=*/true)) {
        tensors.emplace(item.name, item.value.detach().to(torch::kCPU).contiguous());
    }
    std::unordered_map<std::string, std::string> metadata = {
        {"format", "sprite-gpt-cpp-training-v1"},
        {"step", std::to_string(step)},
        {"image_size", std::to_string(kImageSize)},
    };
    auto weights_path = module_path;
    weights_path.replace_extension(".safetensors");
    safetensors::save_safetensors(tensors, weights_path.string(), metadata);
}

static float learning_rate_at(int step, const Options& options) {
    if (options.warmup > 0 && step < options.warmup) {
        return options.learning_rate * static_cast<float>(step + 1) / options.warmup;
    }
    const float progress = static_cast<float>(step) / std::max(1, options.steps);
    return options.learning_rate * (0.1f + 0.9f * 0.5f *
        (1.f + std::cos(static_cast<float>(M_PI) * progress)));
}

static int train(const Options& options, CLIPTokenizer& tokenizer) {
    const torch::Device device(options.device);
    if (device.is_cuda() && !torch::cuda::is_available()) {
        throw std::runtime_error("CUDA requested, but LibTorch cannot access a CUDA device");
    }
    if (device.type() == torch::kXPU && !torch::xpu::is_available()) {
        throw std::runtime_error("XPU requested, but this LibTorch build cannot access an Intel GPU");
    }
    const auto images = load_training_images(options);
    auto clip = torch::jit::load(options.clip_model, device);
    auto unet = torch::jit::load(options.unet_model, device);
    clip.eval();
    unet.train();
    std::vector<torch::Tensor> parameters;
    for (const auto& parameter : unet.parameters()) parameters.push_back(parameter);
    if (parameters.empty()) throw std::runtime_error("SpriteGPT UNet has no trainable parameters");
    for (auto& parameter : parameters) parameter.set_requires_grad(true);

    auto adam_options = torch::optim::AdamWOptions(options.learning_rate);
    adam_options.betas(std::make_tuple(0.9, 0.99)).weight_decay(0.0);
    torch::optim::AdamW optimizer(parameters, adam_options);
    std::mt19937 random(static_cast<uint32_t>(options.seed >= 0 ? options.seed : std::random_device{}()));
    std::uniform_int_distribution<size_t> choose_image(0, images.size() - 1);
    const auto null_tokens = tokenizer.tokenize("");
    auto null_ids = torch::from_blob(const_cast<int64_t*>(null_tokens.ids.data()),
        {1, kContextLength}, torch::TensorOptions().dtype(torch::kInt64)).clone();
    auto null_mask_ids = torch::from_blob(const_cast<int64_t*>(null_tokens.mask.data()),
        {1, kContextLength}, torch::TensorOptions().dtype(torch::kInt64)).clone();
    torch::Tensor null_context;
    {
        torch::NoGradGuard no_grad;
        null_context = run_module(clip, {null_ids, null_mask_ids}).to(device);
    }
    const auto null_mask = null_mask_ids.to(device, torch::kBool);

    std::cout << "training: " << images.size() << " captioned images, batch " << options.batch_size
              << ", steps " << options.steps << ", device " << options.device << '\n' << std::flush;
    auto started = std::chrono::steady_clock::now();
    double accumulated_loss = 0.0;
    for (int step = 0; step < options.steps; ++step) {
        const float lr = learning_rate_at(step, options);
        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
        }
        std::vector<size_t> indices(static_cast<size_t>(options.batch_size));
        for (auto& index : indices) index = choose_image(random);
        const auto x0 = load_image_batch(images, indices, random, device);
        auto captions = encode_captions(clip, tokenizer, images, indices, device);
        auto drop = torch::rand({options.batch_size}, torch::TensorOptions().device(device)) < options.condition_drop;
        auto context = torch::where(drop.view({options.batch_size, 1, 1}),
            null_context.expand({options.batch_size, kContextLength, 512}), captions.context);
        auto mask = torch::where(drop.view({options.batch_size, 1}),
            null_mask.expand({options.batch_size, kContextLength}), captions.mask);

        const auto timestep = torch::sigmoid(torch::randn(
            {options.batch_size}, torch::TensorOptions().dtype(torch::kFloat32).device(device)));
        const auto noise = torch::randn_like(x0);
        const auto tb = timestep.view({options.batch_size, 1, 1, 1});
        const auto xt = (1.0 - tb) * x0 + tb * noise;
        const auto velocity = run_module(unet, {xt, timestep, context, mask});
        const auto loss = torch::nn::functional::mse_loss(velocity, noise - x0);
        optimizer.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(parameters, 1.0);
        optimizer.step();
        accumulated_loss += loss.item<double>();

        if ((step + 1) % options.log_every == 0 || step + 1 == options.steps) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - started).count();
            std::cout << "step " << (step + 1) << "/" << options.steps << " loss "
                      << accumulated_loss / std::min(options.log_every, step + 1)
                      << " lr " << lr << " elapsed " << elapsed << "s\n" << std::flush;
            accumulated_loss = 0.0;
            started = now;
        }
        if ((step + 1) % options.save_every == 0 || step + 1 == options.steps) {
            save_training_model(unet, options.output, step + 1, options.steps);
        }
    }
    return 0;
}

static void save_png(const torch::Tensor& image, const std::string& path) {
    torch::Tensor pixels = ((image.clamp(-1, 1) + 1) * 127.5)
                               .round()
                               .to(torch::kUInt8)
                               .squeeze(0)
                               .permute({1, 2, 0})
                               .contiguous()
                               .cpu();
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    if (!stbi_write_png(path.c_str(), kImageSize, kImageSize, 3, pixels.data_ptr(), kImageSize * 3)) {
        throw std::runtime_error("Could not write output PNG: " + path);
    }
}

static int run(const Options& options) {
    CLIPTokenizer tokenizer(options.tokenizer_dir);
    if (options.train_mode) return train(options, tokenizer);
    const Tokenized conditioned = tokenizer.tokenize(options.prompt);
    if (options.tokenize_only) {
        print_tokens(conditioned);
        return 0;
    }
    const Tokenized unconditioned = tokenizer.tokenize(options.negative_prompt);

    const unsigned hardware_threads = std::max(1u, std::thread::hardware_concurrency());
    at::set_num_threads(options.threads > 0 ? options.threads
                                            : static_cast<int>(std::min(4u, hardware_threads)));
    at::set_num_interop_threads(1);

    const torch::Device device(options.device);
    if (device.is_cuda() && !torch::cuda::is_available()) {
        throw std::runtime_error("CUDA was requested but LibTorch cannot access a CUDA device");
    }
    if (device.type() == torch::kXPU && !torch::xpu::is_available()) {
        throw std::runtime_error("XPU was requested but this LibTorch build cannot access an Intel GPU");
    }

    std::vector<int64_t> ids;
    std::vector<int64_t> masks;
    ids.reserve(kContextLength * 2);
    masks.reserve(kContextLength * 2);
    ids.insert(ids.end(), conditioned.ids.begin(), conditioned.ids.end());
    ids.insert(ids.end(), unconditioned.ids.begin(), unconditioned.ids.end());
    masks.insert(masks.end(), conditioned.mask.begin(), conditioned.mask.end());
    masks.insert(masks.end(), unconditioned.mask.begin(), unconditioned.mask.end());

    torch::Tensor input_ids =
        torch::from_blob(ids.data(), {2, kContextLength}, torch::TensorOptions().dtype(torch::kInt64))
            .clone();
    torch::Tensor attention_mask =
        torch::from_blob(masks.data(), {2, kContextLength}, torch::TensorOptions().dtype(torch::kInt64))
            .clone();

    torch::jit::script::Module clip = torch::jit::load(options.clip_model, torch::Device(torch::kCPU));
    torch::jit::script::Module unet = torch::jit::load(options.unet_model, device);
    clip.eval();
    unet.eval();

    torch::InferenceMode inference_guard;
    const auto start = std::chrono::steady_clock::now();
    torch::Tensor contexts = run_module(clip, {input_ids, attention_mask}).to(device);
    torch::Tensor conditioned_context = contexts.narrow(0, 0, 1);
    torch::Tensor null_context        = contexts.narrow(0, 1, 1);
    torch::Tensor conditioned_mask = attention_mask.narrow(0, 0, 1).to(device, torch::kBool);
    torch::Tensor null_mask        = attention_mask.narrow(0, 1, 1).to(device, torch::kBool);

    torch::manual_seed(options.seed);
    torch::Tensor image =
        torch::randn({1, 3, kImageSize, kImageSize},
                     torch::TensorOptions().device(device).dtype(torch::kFloat32));
    const float delta = 1.0f / static_cast<float>(options.steps);
    for (int step = 0; step < options.steps; ++step) {
        const float time_value = 1.0f - static_cast<float>(step) * delta;
        torch::Tensor timestep =
            torch::full({1}, time_value, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        torch::Tensor velocity =
            run_module(unet, {image, timestep, conditioned_context, conditioned_mask});
        if (options.guidance > 0.f) {
            torch::Tensor null_velocity =
                run_module(unet, {image, timestep, null_context, null_mask});
            velocity = null_velocity + options.guidance * (velocity - null_velocity);
        }
        image = image - delta * velocity;
        if (options.progress) {
            std::cout << "{\"type\":\"progress\",\"step\":" << (step + 1)
                      << ",\"total_steps\":" << options.steps << "}\n" << std::flush;
        }
    }
    image = image.clamp(-1, 1);
    save_png(image, options.output);
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "{\"output\":\"" << json_escape(options.output) << "\","
              << "\"prompt\":\"" << json_escape(options.prompt) << "\","
              << "\"steps\":" << options.steps << ",\"guidance\":" << options.guidance
              << ",\"seed\":" << options.seed << ",\"device\":\""
              << json_escape(options.device) << "\",\"elapsed_ms\":" << elapsed_ms << "}\n";
    return 0;
}

}  // namespace sprite_gpt

int main(int argc, char** argv) {
    try {
        return sprite_gpt::run(sprite_gpt::parse_options(argc, argv));
    } catch (const c10::Error& error) {
        std::cerr << "LibTorch error: " << error.what_without_backtrace() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Sprite-GPT error: " << error.what() << '\n';
    }
    return 1;
}
