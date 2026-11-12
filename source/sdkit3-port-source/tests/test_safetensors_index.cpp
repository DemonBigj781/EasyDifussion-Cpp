#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "model_loader.h"
#include "json.hpp"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void write_index(const std::filesystem::path& path, const nlohmann::json& weights) {
    std::ofstream file(path);
    file << nlohmann::json{{"weight_map", weights}}.dump();
    require(file.good(), "could not create temporary index");
}
static void write_scalar(const std::filesystem::path& path, const std::string& name, float value) {
    std::string header = nlohmann::json{{name, {{"dtype", "F32"}, {"shape", nlohmann::json::array()}, {"data_offsets", {0,4}}}}}.dump();
    while (header.size() % 8) header += ' ';
    uint64_t size = header.size();
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(header.data(), header.size());
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    require(file.good(), "could not create scalar shard");
}
int main() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("sdkit-index-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    } cleanup{directory};
    try {
        std::filesystem::create_directories(directory / "shards");
        write_scalar(directory / "shards/a.safetensors", "a", 3.f);
        write_scalar(directory / "shards/b.safetensors", "b", -7.f);
        const auto index = directory / "model.index.json";
        write_index(index, {{"a", "shards/a.safetensors"}, {"alias", "shards/./a.safetensors"}, {"b", "shards/b.safetensors"}});
        for (bool mmap : {false,true}) {
            ModelLoader loader;
            require(loader.init_from_file(index.string(), "prefix."), "valid sharded index rejected");
            require(loader.get_tensor_storage_map().size() == 2, "duplicate shard or prefix handling changed");
            for (auto item : {std::pair<const char*,float>{"prefix.a",3.f}, {"prefix.b",-7.f}}) {
                std::vector<float> values;
                require(loader.load_float_tensor(item.first, values, 1, mmap) && values.size() == 1 && values[0] == item.second,
                        "shard values/file registration/prefix were not preserved");
            }
        }
        // An index names safetensors shards, not another arbitrary model/index.
        write_index(directory / "self.json", {{"x", "self.json"}});
        write_index(directory / "left.json", {{"x", "right.json"}});
        write_index(directory / "right.json", {{"x", "left.json"}});
        write_index(directory / "nested.json", {{"x", "model.index.json"}});
        write_index(directory / "missing.json", {{"x", "missing.safetensors"}});
        { std::ofstream file(directory / "bad.safetensors"); file << "not a tensor shard"; }
        write_index(directory / "malformed.json", {{"x", "bad.safetensors"}});
        for (const char* name : {"self.json", "left.json", "nested.json", "missing.json", "malformed.json"}) {
            ModelLoader loader;
            require(!loader.init_from_file((directory / name).string()), "invalid/nested/cyclic shard index accepted");
        }
        // Shard format is authoritative, rather than its filename suffix.
        write_scalar(directory / "payload.json", "payload", 9.f);
        write_index(directory / "suffix.index.json", {{"payload", "payload.json"}});
        ModelLoader suffix;
        require(suffix.init_from_file((directory / "suffix.index.json").string()) && suffix.get_tensor_storage_map().count("payload") == 1,
                "valid safetensors shard was redispatched based on extension");
        std::cout << "PASS safetensors shards, deduplication, prefix and mmap/non-mmap values; cyclic/nested/missing/malformed shards rejected without recursion\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
