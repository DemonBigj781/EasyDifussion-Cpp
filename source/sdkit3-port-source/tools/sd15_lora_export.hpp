#ifndef SDKIT_SD15_LORA_EXPORT_HPP
#define SDKIT_SD15_LORA_EXPORT_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"

namespace fs = std::filesystem;

static std::string json_escape(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (static_cast<unsigned char>(c) >= 0x20) out.push_back(c);
    }
    return out;
}

static std::string exported_lora_prefix(const std::string& name) {
    const std::string unet = "lora.model.diffusion_model.";
    const std::string clip = "lora.cond_stage_model.transformer.";
    const std::string weight = ".weight";
    std::string prefix;
    std::string module;
    if (name.compare(0, unet.size(), unet) == 0) {
        prefix = "lora_unet_";
        module = name.substr(unet.size());
    } else if (name.compare(0, clip.size(), clip) == 0) {
        prefix = "lora_te_";
        module = name.substr(clip.size());
    } else {
        throw std::runtime_error("unsupported SD1.5 LoRA module: " + name);
    }
    if (module.size() <= weight.size() || module.compare(module.size() - weight.size(), weight.size(), weight) != 0)
        throw std::runtime_error("LoRA module must end in .weight: " + name);
    module.resize(module.size() - weight.size());
    std::replace(module.begin(), module.end(), '.', '_');
    return prefix + module;
}

static bool save_lora(const fs::path& path, const std::map<std::string, ggml_tensor*>& params,
                      const std::string& trigger, int step, int rank, float alpha) {
    if (rank < 1 || !std::isfinite(alpha) || alpha <= 0.0f)
        throw std::invalid_argument("invalid LoRA export rank or alpha");
    struct Entry { std::string name; std::vector<int64_t> shape; std::vector<float> data; size_t begin, end; };
    std::vector<Entry> entries;
    size_t offset = 0;
    for (const auto& [name, tensor] : params) {
        // ggml_n_dims() removes trailing singleton dimensions. A valid
        // rank-1 down matrix is [input_dim, 1, 1, 1] and reports one dimension.
        // Preserve the logical two-dimensional matrix; never silently drop it.
        if (!tensor || tensor->type != GGML_TYPE_F32 ||
            tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("invalid trainable LoRA matrix for export: " + name);
        }
        Entry entry;
        const bool is_down = name.size() >= 10 && name.compare(name.size() - 10, 10, ".lora_down") == 0;
        const bool is_up = name.size() >= 8 && name.compare(name.size() - 8, 8, ".lora_up") == 0;
        if (!is_down && !is_up) throw std::runtime_error("unsupported LoRA factor: " + name);
        const std::string prefix = name.substr(0, name.size() - (is_down ? 10 : 8));
        const std::string exported = exported_lora_prefix(prefix);
        entry.name = exported + (is_down ? ".lora_down.weight" : ".lora_up.weight");
        entry.shape = {tensor->ne[1], tensor->ne[0]}; // GGML uses fastest dimension first.
        entry.begin = offset;
        entry.data.resize(static_cast<size_t>(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, entry.data.data(), 0, entry.data.size() * sizeof(float));
        // A finite loss does not guarantee that every updated factor is finite.
        // Refuse the export before creating or replacing any output file.
        if (!std::all_of(entry.data.begin(), entry.data.end(),
                         [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("non-finite trainable LoRA matrix: " + name);
        }
        offset += entry.data.size() * sizeof(float);
        entry.end = offset;
        entries.push_back(std::move(entry));
        if (is_down) {
            const auto up = params.find(prefix + ".lora_up");
            if (tensor->ne[1] != rank || up == params.end() || !up->second || up->second->ne[0] != rank)
                throw std::runtime_error("mismatched LoRA export rank or missing up factor: " + name);
            // The inference adapter reads a scalar alpha for each down/up pair.
            entries.push_back({exported + ".alpha", {}, {alpha}, offset, offset + sizeof(float)});
            offset += sizeof(float);
        } else if (!params.count(prefix + ".lora_down")) {
            throw std::runtime_error("missing LoRA down factor: " + name);
        }
    }
    if (entries.empty()) return false;
    std::ostringstream header;
    header << "{\"__metadata__\":{\"format\":\"pt\",\"ss_network_dim\":\"" << rank
           << "\",\"ss_network_alpha\":\"" << alpha << "\",\"ss_training_step\":\"" << step
           << "\",\"ss_tag_frequency\":\"{\\\"" << json_escape(trigger)
           << "\\\":1}\",\"ss_trigger\":\"" << json_escape(trigger) << "\"}";
    for (const auto& entry : entries) {
        header << ",\"" << json_escape(entry.name) << "\":{\"dtype\":\"F32\",\"shape\":[";
        for (size_t i = 0; i < entry.shape.size(); ++i) {
            if (i) header << ',';
            header << entry.shape[i];
        }
        header << "],\"data_offsets\":[" << entry.begin << "," << entry.end << "]}";
    }
    header << "}";
    std::string json = header.str();
    while (json.size() % 8) json.push_back(' ');
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    const uint64_t header_size = json.size();
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(json.data(), static_cast<std::streamsize>(json.size()));
    for (const auto& entry : entries) out.write(reinterpret_cast<const char*>(entry.data.data()),
                                                  static_cast<std::streamsize>(entry.data.size() * sizeof(float)));
    out.close();
    if (!out) { fs::remove(temporary); return false; }
    std::error_code ec;
    fs::rename(temporary, path, ec);
    if (ec) { fs::remove(temporary); return false; }
    return true;
}

#endif
