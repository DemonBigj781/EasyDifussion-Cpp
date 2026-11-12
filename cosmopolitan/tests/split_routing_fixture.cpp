// The Python test inserts the production routing block. Only inspection is mocked.
#include "crow.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
#define LOG_DEBUG(...) ((void)0)

static std::string inspectModelType(const std::string& path) {
    const auto stem = fs::path(path).stem().string();
    return stem == "second_vae" ? "vae" : stem;
}

static std::vector<std::string> route(const crow::json::rvalue& options) {
/* PRODUCTION_COMPANION_ROUTING */
    return {vae_path_str, clip_l_path_str, clip_g_path_str, t5xxl_path_str, llm_path_str, llm_vision_path_str};
}

int main(int argc, char** argv) {
    assert(argc == 2);
    std::vector<std::string> files;
    for (const auto* role : {"vae", "clip_l", "clip_g", "t5xxl", "llm", "llm_vision", "second_vae", "unknown"}) {
        const auto path = (fs::path(argv[1]) / (std::string(role) + ".weights")).string();
        std::ofstream(path) << "fixture, not a model";
        files.push_back(path);
    }
    auto options = [](const std::vector<std::string>& modules) {
        crow::json::wvalue value;
        value["forge_additional_modules"] = modules;
        return crow::json::load(value.dump());
    };
    const std::vector<std::string> all(files.begin(), files.begin() + 6);
    assert(route(options(all)) == all);
    // A VAE alone remains valid for an otherwise complete checkpoint.
    const auto vae_only = route(options({files[0]}));
    assert(vae_only[0] == files[0]);
    for (size_t i = 1; i < vae_only.size(); ++i) assert(vae_only[i].empty());
    for (const auto& path : route(options({}))) assert(path.empty());
    auto rejects = [&](const crow::json::rvalue& value, const char* expected) {
        try {
            (void)route(value);
            assert(false);
        } catch (const std::invalid_argument& error) {
            assert(std::string(error.what()).find(expected) != std::string::npos);
        }
    };
    rejects(options({files[0], files[6]}), "Multiple companion modules for role: vae");
    rejects(options({files[1], files[1]}), "Multiple companion modules for role: clip_l");
    rejects(options({files[7]}), "Unsupported companion module type");
    rejects(options({(fs::path(argv[1]) / "missing.weights").string()}), "file not found or invalid");
    rejects(options({argv[1]}), "file not found or invalid");
    rejects(options({""}), "file not found or invalid");
    rejects(options({files[0] + std::string("\0suffix", 7)}), "file not found or invalid");
    rejects(crow::json::load("{\"forge_additional_modules\":false}"), "must be an array");
    rejects(crow::json::load("{\"forge_additional_modules\":[1]}"), "must be strings");
}
