#include "../tools/sd15_lora_export.hpp"
#include "ggml-cpu.h"
#include "json.hpp"
#include "name_conversion.h"
#include <chrono>
#include <iostream>
#include <limits>

int main() {
    const auto path = fs::temp_directory_path() / ("sd15-export-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
    auto backend = ggml_backend_cpu_init();
    bool good = true;
    for (int rank : {1, 32}) {
        auto ctx = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        auto down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, rank);
        auto up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, 2);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> a(2 * rank, .25f), b(2 * rank, .5f);
        ggml_backend_tensor_set(down, a.data(), 0, a.size() * sizeof(float));
        ggml_backend_tensor_set(up, b.data(), 0, b.size() * sizeof(float));
        const float alpha = rank / 2.0f;
        const std::string unet = "lora.model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight";
        const std::string clip = "lora.cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight";
        const std::map<std::string, std::string> exported = {
            {unet, "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q"},
            {clip, "lora_te_text_model_encoder_layers_0_self_attn_k_proj"}};
        std::map<std::string, ggml_tensor*> params;
        for (const auto& prefix : {unet, clip}) {
            params[prefix + ".lora_down"] = down;
            params[prefix + ".lora_up"] = up;
        }
        good = save_lora(path, params, "cat", 3, rank, alpha) && good;
        std::ifstream input(path, std::ios::binary);
        uint64_t size = 0;
        input.read(reinterpret_cast<char*>(&size), sizeof(size));
        std::string raw(size, '\0');
        input.read(raw.data(), static_cast<std::streamsize>(size));
        auto header = nlohmann::json::parse(raw);
        good = good && std::stoi(header["__metadata__"]["ss_network_dim"].get<std::string>()) == rank;
        for (const auto& prefix : {unet, clip}) {
            const auto& key = exported.at(prefix);
            if (!header.contains(key + ".lora_down.weight")) {
                std::cerr << "FAIL: missing interoperable export key " << key << '\n';
                good = false;
                continue;
            }
            good = good && header[key + ".alpha"]["shape"].empty();
            good = good && header[key + ".lora_down.weight"]["shape"] == nlohmann::json::array({rank, 2});
            for (const auto& suffix : {".lora_down", ".lora_up"})
                good = good && convert_tensor_name("lora." + key + suffix + ".weight", VERSION_SD1) == prefix + suffix;
            good = good && convert_tensor_name("lora." + key + ".alpha", VERSION_SD1) == prefix + ".alpha";
            auto read_value = [&](const std::string& key) {
                const auto offset = header.at(key).at("data_offsets").at(0).get<size_t>();
                input.seekg(static_cast<std::streamoff>(8 + size + offset));
                float value = 0;
                input.read(reinterpret_cast<char*>(&value), sizeof(value));
                good = good && bool(input);
                return value;
            };
            const float loaded_alpha = read_value(key + ".alpha");
            const float loaded_down = read_value(key + ".lora_down.weight");
            const float loaded_up = read_value(key + ".lora_up.weight");
            const float inference_delta = rank * loaded_down * loaded_up * (loaded_alpha / rank);
            good = good && loaded_alpha == alpha && inference_delta == .125f * alpha;
        }
        input.close();
        const auto saved_size = fs::file_size(path);
        a[0] = std::numeric_limits<float>::infinity();
        ggml_backend_tensor_set(down, a.data(), 0, a.size() * sizeof(float));
        try { save_lora(path, params, "cat", 4, rank, alpha); good = false; }
        catch (const std::runtime_error&) {}
        good = good && fs::file_size(path) == saved_size && !fs::exists(path.string() + ".tmp");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    fs::remove(path);
    ggml_backend_free(backend);
    std::cout << (good ? "PASS" : "FAIL") << ": rank-1/rank-32 UNet and CLIP alpha export, scaling and invalid-factor refusal\n";
    return good ? 0 : 1;
}
