#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "model_io/gguf_reader_ext.h"

using Bytes = std::vector<char>;
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template <typename T> static void append(Bytes& bytes, T value) {
    const auto* data = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}
static void string(Bytes& bytes, const std::string& value) {
    append<uint64_t>(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}
static Bytes header(uint64_t tensors = 1) {
    Bytes bytes{'G','G','U','F'};
    append<uint32_t>(bytes, 3);
    append<uint64_t>(bytes, tensors);
    append<uint64_t>(bytes, 1);
    string(bytes, "test.metadata");
    return bytes;
}
static void tensor(Bytes& bytes) {
    string(bytes, "test.weight");
    append<uint32_t>(bytes, 2);
    append<int64_t>(bytes, 4);
    append<int64_t>(bytes, 3);
    append<uint32_t>(bytes, GGML_TYPE_F32);
    append<uint64_t>(bytes, 0);
    while (bytes.size() % 32) bytes.push_back(0);
    bytes.resize(bytes.size() + 12 * sizeof(float), 0);
}
static int width(uint32_t type) {
    if (type == 0 || type == 1 || type == 7) return 1;
    if (type == 2 || type == 3) return 2;
    if (type == 4 || type == 5 || type == 6) return 4;
    return 8;
}
static bool load(const std::filesystem::path& path, const Bytes& bytes, bool check_tensor = false) {
    { std::ofstream file(path, std::ios::binary); file.write(bytes.data(), bytes.size()); require(file.good(), "fixture write failed"); }
    GGUFReader reader;
    bool ok = reader.load(path.string());
    if (ok && check_tensor) {
        require(reader.tensors().size() == 1 && reader.tensors()[0].name == "test.weight" &&
                    reader.tensors()[0].shape == std::vector<int64_t>({4,3}) && reader.tensors()[0].type == GGML_TYPE_F32 &&
                    reader.data_offset() == bytes.size() - 12 * sizeof(float), "metadata skip corrupted tensor descriptor/data offset");
    }
    return ok;
}
int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("sdkit-gguf-metadata-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code ignored; std::filesystem::remove(path, ignored); } } cleanup{path};
    try {
        int valid = 0;
        for (uint32_t type = 0; type <= 12; ++type) {
            if (type == 9) continue;
            for (int count : {-1,0,3}) {
                Bytes bytes = header();
                append<uint32_t>(bytes, count < 0 ? type : 9);
                if (count >= 0) { append<uint32_t>(bytes, type); append<uint64_t>(bytes, count); }
                for (int i = 0; i < (count < 0 ? 1 : count); ++i) {
                    if (type == 8) string(bytes, i == 1 ? "" : "metadata string");
                    else bytes.insert(bytes.end(), width(type), static_cast<char>(i + 1));
                }
                tensor(bytes);
                require(load(path, bytes, true), "valid scalar/array metadata rejected");
                ++valid;
            }
        }
        // Exercise both sides of the buffered-ignore/seek threshold without changing offsets.
        for (size_t length : {size_t{0}, size_t{65535}, size_t{65536}, size_t{65537}, size_t{131072}}) {
            for (bool strings : {false,true}) {
                Bytes bytes = header();
                if (strings) {
                    append<uint32_t>(bytes, 8);
                    string(bytes, std::string(length, 'x'));
                } else {
                    append<uint32_t>(bytes, 9);
                    append<uint32_t>(bytes, 0);
                    append<uint64_t>(bytes, length);
                    bytes.insert(bytes.end(), length, 0);
                }
                tensor(bytes);
                require(load(path, bytes, true), "buffered/seek boundary corrupted tensor offset");
                ++valid;
            }
        }
        {
            Bytes bytes = header();
            append<uint32_t>(bytes, 9);
            append<uint32_t>(bytes, 8);
            append<uint64_t>(bytes, 8192);
            for (int i = 0; i < 8192; ++i) string(bytes, "token_" + std::to_string(i));
            tensor(bytes);
            require(load(path, bytes, true), "many short metadata strings corrupted tensor offset");
            ++valid;
        }
        for (uint32_t type : {0u,4u,8u,12u}) {
            Bytes bytes = header(0);
            append<uint32_t>(bytes, 9);
            append<uint32_t>(bytes, type);
            append<uint64_t>(bytes, std::numeric_limits<uint64_t>::max());
            require(!load(path, bytes), "unbounded metadata count accepted");
        }
        Bytes scalar = header(0);
        append<uint32_t>(scalar, 4);
        require(!load(path, scalar), "truncated scalar metadata skipped beyond EOF");
        Bytes text = header(0);
        append<uint32_t>(text, 8);
        append<uint64_t>(text, std::numeric_limits<uint64_t>::max());
        require(!load(path, text), "oversized metadata string accepted");
        Bytes array = header(0);
        append<uint32_t>(array, 9);
        append<uint32_t>(array, 8);
        append<uint64_t>(array, 2);
        string(array, "first");
        string(array, "second");
        for (size_t length = 0; length < array.size(); ++length) {
            require(!load(path, Bytes(array.begin(), array.begin() + length)), "truncated string array accepted");
        }
        for (uint32_t type : {9u,99u}) {
            Bytes bytes = header(0);
            append<uint32_t>(bytes, 9);
            append<uint32_t>(bytes, type);
            append<uint64_t>(bytes, 1);
            bytes.resize(bytes.size() + 16, 0);
            require(!load(path, bytes), "nested/unknown array element type accepted");
        }
        std::cout << "PASS " << valid << " scalar/array GGUF cases, buffered/seek boundaries, tensor offsets, truncation, excessive counts and invalid types\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
