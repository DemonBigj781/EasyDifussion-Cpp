// Native WD14 ONNX tagger sidecar. Reads one JSON request and writes one JSON
// response per line, keeping model sessions alive between gallery images.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "crow.h"
#include "base64.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../stable-diffusion.cpp/thirdparty/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stable-diffusion.cpp/thirdparty/stb_image_resize.h"

namespace {
constexpr size_t kMaxImageBytes = 64 * 1024 * 1024;
Ort::Env environment(ORT_LOGGING_LEVEL_ERROR, "wd14-tagger");

struct TagLabel { std::string name; int category; };
struct Model {
    Ort::Session session{nullptr};
    std::string input_name;
    std::string output_name;
    std::string provider = "CPUExecutionProvider";
    int size = 0;
    std::vector<TagLabel> labels;
    Model(const std::string& model_path, const std::string& csv_path) {
        Ort::SessionOptions options;
        options.SetLogSeverityLevel(3);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const auto providers = Ort::GetAvailableProviders();
        if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end()) {
            try {
                OrtCUDAProviderOptions cuda_options{};
                options.AppendExecutionProvider_CUDA(cuda_options);
                provider = "CUDAExecutionProvider";
            } catch (const Ort::Exception&) {
                // Fall back to CPU when CUDA support is listed but not loadable.
            }
        }
        try {
            session = Ort::Session(environment, model_path.c_str(), options);
        } catch (const Ort::Exception&) {
            if (provider != "CUDAExecutionProvider") throw;
            provider = "CPUExecutionProvider";
            Ort::SessionOptions cpu_options;
            cpu_options.SetLogSeverityLevel(3);
            cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session = Ort::Session(environment, model_path.c_str(), cpu_options);
        }
        Ort::AllocatorWithDefaultOptions allocator;
        auto input = session.GetInputNameAllocated(0, allocator);
        auto output = session.GetOutputNameAllocated(0, allocator);
        input_name = input.get();
        output_name = output.get();
        auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[1] <= 0 || shape[1] != shape[2] || shape[3] != 3) {
            throw std::runtime_error("WD14 model must use an NHWC square input with three channels");
        }
        size = static_cast<int>(shape[2]);
        std::ifstream csv(csv_path);
        if (!csv) throw std::runtime_error("WD14 tag CSV not found: " + csv_path);
        std::string line;
        if (!std::getline(csv, line)) throw std::runtime_error("WD14 tag CSV is empty");
        auto header = parse_csv(line);
        auto name_col = std::find(header.begin(), header.end(), "name");
        auto category_col = std::find(header.begin(), header.end(), "category");
        if (name_col == header.end() || category_col == header.end())
            throw std::runtime_error("WD14 CSV requires name and category columns");
        const auto name_index = static_cast<size_t>(name_col - header.begin());
        const auto category_index = static_cast<size_t>(category_col - header.begin());
        while (std::getline(csv, line)) {
            auto row = parse_csv(line);
            if (row.size() <= std::max(name_index, category_index))
                throw std::runtime_error("Malformed row in WD14 tag CSV");
            labels.push_back({row[name_index], std::stoi(row[category_index])});
        }
        std::cerr << "WD14_DIAG model_loaded provider=" << provider
                  << " labels=" << labels.size() << std::endl;
    }
    static std::vector<std::string> parse_csv(const std::string& line) {
        std::vector<std::string> fields;
        std::string field;
        bool quoted = false;
        for (size_t i = 0; i < line.size(); ++i) {
            char c = line[i];
            if (quoted && c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"'; ++i;
            } else if (c == '"') quoted = !quoted;
            else if (c == ',' && !quoted) { fields.push_back(field); field.clear(); }
            else field += c;
        }
        fields.push_back(field);
        return fields;
    }
};

std::unordered_map<std::string, std::unique_ptr<Model>> models;

void log_diagnostic(const std::string& event) {
    std::cerr << "WD14_DIAG " << event << std::endl;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: if (c < 0x20) out << ' '; else out << static_cast<char>(c);
        }
    }
    return out.str();
}

std::string string_value(const crow::json::rvalue& request, const char* key,
                         const std::string& fallback = "") {
    const auto& value = request[key];
    return value ? value.s() : fallback;
}
double number_value(const crow::json::rvalue& request, const char* key, double fallback) {
    const auto& value = request[key];
    return value ? value.d() : fallback;
}

std::vector<unsigned char> decode_image(std::string encoded) {
    if (encoded.rfind("data:", 0) == 0) {
        const auto comma = encoded.find(',');
        if (comma == std::string::npos || encoded.substr(0, comma).find(";base64") == std::string::npos)
            throw std::runtime_error("Invalid image data URL");
        encoded.erase(0, comma + 1);
    }
    if (encoded.size() > (kMaxImageBytes * 4 / 3) + 16)
        throw std::runtime_error("Image payload is too large");
    if (encoded.empty() || encoded.size() % 4 != 0)
        throw std::runtime_error("Image is not valid base64");
    bool padding = false;
    int padding_count = 0;
    for (char c : encoded) {
        if (c == '=') { padding = true; ++padding_count; }
        else if (padding || base64_chars.find(c) == std::string::npos)
            throw std::runtime_error("Image is not valid base64");
    }
    if (padding_count > 2) throw std::runtime_error("Image is not valid base64");
    auto bytes = base64_decode(encoded);
    if (bytes.empty() || bytes.size() > kMaxImageBytes)
        throw std::runtime_error("Invalid or oversized image payload");
    return bytes;
}

std::string format_tag(std::string tag, bool replace_underscore) {
    if (replace_underscore) std::replace(tag.begin(), tag.end(), '_', ' ');
    std::string formatted;
    for (char c : tag) { if (c == '(' || c == ')') formatted += '\\'; formatted += c; }
    return formatted;
}

std::string run(const crow::json::rvalue& request) {
    const auto model_path = string_value(request, "model_path");
    const auto csv_path = string_value(request, "csv_path");
    const auto model_name = string_value(request, "model", "wd-v1-4-moat-tagger-v2");
    if (model_path.empty() || csv_path.empty()) throw std::runtime_error("WD14 model and CSV paths are required");
    const std::string key = model_path + "\n" + csv_path;
    const bool model_loaded = !models.contains(key);
    if (!models.contains(key)) models.emplace(key, std::make_unique<Model>(model_path, csv_path));
    Model& model = *models.at(key);

    auto bytes = decode_image(string_value(request, "image"));
    int width = 0, height = 0, channels = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image(
        stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &width, &height, &channels, 3), stbi_image_free);
    if (!image || width < 1 || height < 1) throw std::runtime_error("Could not decode image");
    const double ratio = static_cast<double>(model.size) / std::max(width, height);
    const int resized_width = std::max(1, static_cast<int>(width * ratio));
    const int resized_height = std::max(1, static_cast<int>(height * ratio));
    std::vector<unsigned char> resized(static_cast<size_t>(resized_width) * resized_height * 3);
    if (!stbir_resize_uint8(image.get(), width, height, 0, resized.data(), resized_width, resized_height, 0, 3))
        throw std::runtime_error("Image resize failed");
    const int pad_x = (model.size - resized_width) / 2;
    const int pad_y = (model.size - resized_height) / 2;
    std::vector<float> input(static_cast<size_t>(model.size) * model.size * 3, 255.0f);
    for (int y = 0; y < resized_height; ++y) for (int x = 0; x < resized_width; ++x) {
        const size_t source = (static_cast<size_t>(y) * resized_width + x) * 3;
        const size_t dest = (static_cast<size_t>(y + pad_y) * model.size + x + pad_x) * 3;
        input[dest] = resized[source + 2]; input[dest + 1] = resized[source + 1]; input[dest + 2] = resized[source];
    }
    const std::array<int64_t, 4> shape{1, model.size, model.size, 3};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
    const char* input_names[] = {model.input_name.c_str()};
    const char* output_names[] = {model.output_name.c_str()};
    auto output = model.session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
    auto scores = output[0].GetTensorData<float>();
    auto score_count = output[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (score_count != model.labels.size()) throw std::runtime_error("WD14 model/CSV label count mismatch");

    const auto threshold = number_value(request, "threshold", .35);
    const auto character_threshold = number_value(request, "character_threshold", .85);
    const auto& replace_setting = request["replace_underscore"];
    const auto& trailing_setting = request["trailing_comma"];
    const bool replace_underscore = replace_setting && replace_setting.b();
    const bool trailing_comma = trailing_setting && trailing_setting.b();
    std::unordered_set<std::string> excluded;
    std::string exclusions = string_value(request, "exclude_tags");
    std::stringstream exclusion_stream(exclusions); std::string exclusion;
    while (std::getline(exclusion_stream, exclusion, ',')) {
        auto first = exclusion.find_first_not_of(" \t\r\n");
        auto last = exclusion.find_last_not_of(" \t\r\n");
        if (first != std::string::npos) {
            exclusion = exclusion.substr(first, last - first + 1);
            std::transform(exclusion.begin(), exclusion.end(), exclusion.begin(), [](unsigned char c) { return std::tolower(c); });
            excluded.insert(exclusion);
        }
    }
    struct Match { std::string tag, raw; float score; };
    std::vector<Match> matches, ratings;
    for (size_t i = 0; i < model.labels.size(); ++i) {
        const auto& label = model.labels[i];
        std::string lower = label.name;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
        auto rendered = format_tag(label.name, replace_underscore);
        std::string rendered_lower = rendered;
        std::transform(rendered_lower.begin(), rendered_lower.end(), rendered_lower.begin(), [](unsigned char c) { return std::tolower(c); });
        Match item{rendered, label.name, scores[i]};
        if (label.category == 9) ratings.push_back(item);
        else if ((label.category == 4 && scores[i] > character_threshold) ||
                 (label.category == 0 && scores[i] > threshold)) {
            if (!excluded.contains(lower) && !excluded.contains(rendered_lower)) matches.push_back(item);
        }
    }
    std::sort(ratings.begin(), ratings.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    std::ostringstream tags;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (i) tags << ", ";
        tags << matches[i].tag;
    }
    if (trailing_comma && !matches.empty()) tags << ", ";
    std::ostringstream json;
    json << "{\"model\":\"" << json_escape(model_name) << "\",\"model_loaded\":"
         << (model_loaded ? "true" : "false") << ",\"tags\":\"" << json_escape(tags.str())
         << "\",\"providers\":[\"" << model.provider << "\"],\"matches\":[";
    for (size_t i = 0; i < matches.size(); ++i) {
        if (i) json << ',';
        json << "{\"tag\":\"" << json_escape(matches[i].tag) << "\",\"raw_tag\":\"" << json_escape(matches[i].raw)
             << "\",\"score\":" << matches[i].score << '}';
    }
    json << "],\"ratings\":[";
    for (size_t i = 0; i < ratings.size(); ++i) {
        if (i) json << ',';
        json << "{\"tag\":\"" << json_escape(ratings[i].tag) << "\",\"raw_tag\":\"" << json_escape(ratings[i].raw)
             << "\",\"score\":" << ratings[i].score << '}';
    }
    json << "]}";
    log_diagnostic("result matches=" + std::to_string(matches.size()) +
                   " ratings=" + std::to_string(ratings.size()));
    return json.str();
}
}  // namespace

int main() {
    log_diagnostic("activated backend=cpp");
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            auto request = crow::json::load(line);
            if (!request) throw std::runtime_error("Invalid JSON request");
            std::cout << run(request) << std::endl;
        } catch (const std::exception& error) {
            log_diagnostic("request_failed");
            std::cout << "{\"error\":\"" << json_escape(error.what()) << "\"}" << std::endl;
        }
    }
    return 0;
}
