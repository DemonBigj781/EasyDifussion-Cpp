#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "model_loader.h"

// Exercise the production loader conversion without widening its public API.
void convert_tensor(void*, ggml_type, void*, ggml_type, int, int, std::vector<float>);

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void check_conversion() {
    int32_t source[] = {0, 1, -1, 16777217, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
    float result[8] = {77, 0, 0, 0, 0, 0, 0, 88};
    convert_tensor(source, GGML_TYPE_I32, result + 1, GGML_TYPE_F32, 2, 3, {});
    require(result[0] == 77 && result[7] == 88, "integer conversion wrote past its destination");
    for (int i = 0; i < 6; ++i) require(result[i + 1] == static_cast<float>(source[i]), "integer-to-F32 cast mismatch");
    int32_t copy[6] = {};
    convert_tensor(source, GGML_TYPE_I32, copy, GGML_TYPE_I32, 2, 3, {});
    for (int i = 0; i < 6; ++i) require(copy[i] == source[i], "same-type integer copy changed");
    ggml_fp16_t half[] = {ggml_fp32_to_fp16(-1.5f), ggml_fp32_to_fp16(2.25f)};
    convert_tensor(half, GGML_TYPE_F16, result, GGML_TYPE_F32, 1, 2, {});
    require(result[0] == -1.5f && result[1] == 2.25f, "F16 conversion changed");
}

static void check_scalar_file(bool wide, bool mmap) {
    auto path = std::filesystem::temp_directory_path() /
                ("sdkit-integer-lora-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ignored; std::filesystem::remove(path, ignored); }
    } cleanup{path};
    const std::string dtype = wide ? "I64" : "I32";
    const int width = wide ? 8 : 4;
    std::string header = "{\"lora_test.alpha\":{\"dtype\":\"" + dtype + "\",\"shape\":[],\"data_offsets\":[0," + std::to_string(width) + "]}}";
    while (header.size() % 8 != 0) header += ' ';
    uint64_t length = header.size();
    int64_t value64 = 24;
    int32_t value32 = 24;
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(header.data(), header.size());
        file.write(wide ? reinterpret_cast<const char*>(&value64) : reinterpret_cast<const char*>(&value32), width);
        require(file.good(), "could not write scalar fixture");
    }
    ModelLoader loader;
    require(loader.init_from_file(path.string()), "integer scalar metadata could not be loaded");
    std::vector<float> result;
    require(loader.load_float_tensor("lora_test.alpha", result, 1, mmap), "integer LoRA alpha could not be loaded as F32");
    require(result.size() == 1 && result[0] == 24.0f, "integer LoRA alpha value changed");
}

int main() {
    try {
        check_conversion();
        for (bool wide : {false, true}) {
            for (bool mmap : {false, true}) check_scalar_file(wide, mmap);
        }
        std::cout << "PASS I32 conversion and rounding, bounds, copy/F16 controls, I32/I64 LoRA scalar file loading with/without mmap\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
