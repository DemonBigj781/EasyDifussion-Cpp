#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "ggml-cpu.h"
#include "conditioning/conditioner.hpp"

struct EmbeddingFixture {
    std::string name;
    int width;
    int rows;
    float value;
    bool half = false;
};

static void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

// Generate tiny synthetic tensors, never download or depend on user models.
static void write_fixture(const std::filesystem::path& path, const std::vector<EmbeddingFixture>& tensors) {
    nlohmann::json header;
    std::vector<char> data;
    for (const auto& tensor : tensors) {
        size_t start = data.size();
        for (int i = 0; i < tensor.width * tensor.rows; ++i) {
            if (tensor.half) {
                ggml_fp16_t value = ggml_fp32_to_fp16(tensor.value);
                const char* bytes = reinterpret_cast<const char*>(&value);
                data.insert(data.end(), bytes, bytes + sizeof(value));
            } else {
                const char* bytes = reinterpret_cast<const char*>(&tensor.value);
                data.insert(data.end(), bytes, bytes + sizeof(tensor.value));
            }
        }
        header[tensor.name] = {{"dtype", tensor.half ? "F16" : "F32"},
                               {"shape", {tensor.rows, tensor.width}},
                               {"data_offsets", {start, data.size()}}};
    }
    std::string json = header.dump();
    while (json.size() % 8) {
        json += ' ';
    }
    uint64_t size = json.size();
    std::ofstream out(path, std::ios::binary);
    // safetensors stores its header length as little-endian, including on big-endian hosts.
    for (int i = 0; i < 8; ++i) {
        out.put(static_cast<char>((size >> (8 * i)) & 0xff));
    }
    out.write(json.data(), json.size());
    out.write(data.data(), data.size());
    require(out.good(), "fixture write failed");
}

static void expect_rows(const std::vector<uint8_t>& data, int width, const std::vector<float>& values) {
    require(data.size() == width * values.size() * sizeof(float), "wrong buffer size");
    for (size_t row = 0; row < values.size(); ++row) {
        for (int col = 0; col < width; ++col) {
            float value;
            std::memcpy(&value, data.data() + (row * width + col) * sizeof(float), sizeof(value));
            require(value == values[row], "incorrect embedding value or padding");
        }
    }
}

int main() {
    auto directory = std::filesystem::temp_directory_path() /
                     ("sdkit-textual-inversion-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    auto backend = ggml_backend_cpu_init();
    int result = 0;
    try {
        require(backend != nullptr, "CPU backend unavailable");
        auto path = directory / "embedding.safetensors";
        FrozenCLIPEmbedderWithCustomWords model(backend, {}, {{"test", path.string()}}, VERSION_SDXL);
        std::vector<int32_t> tokens;
        int vocab = model.text_model->model.vocab_size;
        auto load = [&](const char* name, const std::vector<EmbeddingFixture>& tensors, int count) {
            write_fixture(path, tensors);
            tokens.clear();
            int start = model.num_custom_embeddings;
            require(model.load_embedding(name, path.string(), tokens), "embedding load failed");
            require(tokens.size() == static_cast<size_t>(count), "duplicate or missing token IDs");
            for (int i = 0; i < count; ++i) {
                require(tokens[i] == vocab + start + i, "noncontiguous token IDs");
            }
            require(model.num_custom_embeddings == start + count, "wrong shared row count");
        };
        load("pair", {{"clip_l", 768, 1, 1}, {"clip_g", 1280, 1, 2}}, 1);
        expect_rows(model.token_embed_custom, 768, {1});
        expect_rows(model.token_embed_custom2, 1280, {2});
        std::vector<int32_t> cached;
        require(model.load_embedding("pair", path.string(), cached), "cache lookup failed");
        require(cached == tokens && model.num_custom_embeddings == 1, "cached load mutated IDs");
        expect_rows(model.token_embed_custom, 768, {1});
        expect_rows(model.token_embed_custom2, 1280, {2});
        std::cout << "PASS paired tensors and cache\n";

        load("l-only", {{"clip_l", 768, 2, 3}}, 2);
        load("g-only", {{"clip_g", 1280, 1, 4}}, 1);
        expect_rows(model.token_embed_custom, 768, {1, 3, 3, 0});
        expect_rows(model.token_embed_custom2, 1280, {2, 0, 0, 4});
        std::cout << "PASS one-sided tensors and append offsets\n";

        load("unequal", {{"clip_l", 768, 1, 5}, {"clip_g", 1280, 2, 6}}, 2);
        load("reverse", {{"clip_l", 768, 2, 7}, {"clip_g", 1280, 1, 8}}, 2);
        expect_rows(model.token_embed_custom, 768, {1, 3, 3, 0, 5, 0, 7, 7});
        expect_rows(model.token_embed_custom2, 1280, {2, 0, 0, 4, 6, 6, 8, 0});
        std::cout << "PASS unequal rows in both directions\n";

        load("half", {{"clip_l", 768, 1, 9, true}, {"clip_g", 1280, 1, 10, true}}, 1);
        expect_rows(model.token_embed_custom, 768, {1, 3, 3, 0, 5, 0, 7, 7, 9});
        expect_rows(model.token_embed_custom2, 1280, {2, 0, 0, 4, 6, 6, 8, 0, 10});
        std::cout << "PASS F16 source converted to F32 destination\n";

        auto before_l = model.token_embed_custom;
        auto before_g = model.token_embed_custom2;
        auto before_cache = model.embedding_pos_map;
        write_fixture(path, {{"bad", 16, 1, 1}});
        tokens = {123};
        require(!model.load_embedding("wrong-width", path.string(), tokens), "wrong width accepted");
        require(!model.load_embedding("missing", (directory / "missing.safetensors").string(), tokens), "missing file accepted");
        write_fixture(path, {{"clip_l", 768, 1, 1}, {"clip_g", 1280, 1, 2}});
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 64);
        require(!model.load_embedding("truncated", path.string(), tokens), "truncated file accepted");
        require(model.num_custom_embeddings == 9 && tokens == std::vector<int32_t>{123}, "failed load changed IDs");
        require(model.token_embed_custom == before_l && model.token_embed_custom2 == before_g && model.embedding_pos_map == before_cache, "failed load changed buffers/cache");
        std::cout << "PASS failed loads leave state unchanged\n";

        FrozenCLIPEmbedderWithCustomWords sd1(backend, {}, {{"test", path.string()}}, VERSION_SD1);
        write_fixture(path, {{"clip_l", 768, 2, 11, true}});
        tokens.clear();
        require(sd1.load_embedding("sd1", path.string(), tokens), "SD1 embedding failed");
        require(tokens.size() == 2 && sd1.num_custom_embeddings == 2 && sd1.token_embed_custom2.empty(), "SD1 token state changed");
        expect_rows(sd1.token_embed_custom, 768, {11, 11});
        std::cout << "PASS single-encoder SD1 compatibility\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        result = 1;
    }
    ggml_backend_free(backend);
    std::filesystem::remove_all(directory);
    return result;
}
