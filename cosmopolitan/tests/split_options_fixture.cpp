// Production configuration and OptionsManager only. No engine or backend.
#include "config.hpp"
#include "options_manager.h"
#include "json.hpp"
#include <cassert>
#include <fstream>
#include <iterator>
#include <stdexcept>

using Json = nlohmann::json;

static std::string read(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    assert(input);
    return std::string(std::istreambuf_iterator<char>(input), {});
}

int main(int argc, char** argv) {
    assert(cosmo_config_prepare(&argc, &argv) == -1);
    const auto path = Json::parse(cosmo_config_document())["config_path"].get<std::string>();
    const Json saved = {{"sd_model_checkpoint", "saved.gguf"},
                        {"forge_additional_modules", {"/models/saved-clip.gguf", "/models/saved-vae.gguf"}}};
    cosmo_config_set_options(saved.dump());
    const auto before_disk = read(path);
    const auto before_options = Json::parse(cosmo_config_options());
    OptionsManager manager("unused-options.json");
    assert(manager.load());
    auto check_unchanged = [&] {
        assert(read(path) == before_disk);
        assert(Json::parse(cosmo_config_options()) == before_options);
        assert(Json::parse(manager.getOptions(false).dump()) == before_options);
    };
    const Json one_shot = {{"sd_model_checkpoint", "request.gguf"},
                           {"forge_additional_modules", {"/models/request clip.gguf", "/models/request-vae.gguf"}}};
    manager.beginRequestOverrides(crow::json::load(one_shot.dump()));
    auto effective = Json::parse(manager.getOptions().dump());
    for (const auto& item : one_shot.items()) assert(effective[item.key()] == item.value());
    check_unchanged();
    assert(!manager.setOptions(crow::json::load("{\"forge_additional_modules\":[]}")));
    assert(manager.save());
    try {
        manager.beginRequestOverrides(crow::json::load("{}"));
        assert(false);
    } catch (const std::runtime_error&) {}
    check_unchanged();
    manager.endRequestOverrides();
    assert(Json::parse(manager.getOptions().dump()) == before_options);
    manager.beginRequestOverrides(crow::json::load("{\"forge_additional_modules\":[]}"));
    assert(Json::parse(manager.getOptions().dump())["forge_additional_modules"] == Json::array());
    check_unchanged();
    manager.endRequestOverrides();

    Json too_many = Json::array();
    for (int i = 0; i < 7; ++i) too_many.push_back("/module" + std::to_string(i));
    const std::vector<Json> bad = {nullptr, false, Json::object(), "path", {1}, {nullptr},
        {Json::array()}, {""}, {"relative.gguf"}, {"https://example.test/model"},
        {"C:/models/vae.gguf"}, {"//server/share/model"}, {"/models\\vae.gguf"},
        {std::string("/bad\0path", 9)}, {"/" + std::string(4096, 'x')},
        {"/same", "/a/../same"}, too_many};
    for (const auto& modules : bad) {
        const Json overrides = {{"sd_model_checkpoint", "must-not-leak.gguf"},
                                {"forge_additional_modules", modules}};
        try {
            manager.beginRequestOverrides(crow::json::load(overrides.dump()));
            assert(false);
        } catch (const std::invalid_argument&) {}
        check_unchanged();
        assert(Json::parse(manager.getOptions().dump()) == before_options);
        // A failed overlay has not left the manager active or partially changed.
        manager.beginRequestOverrides(crow::json::load("{}"));
        assert(Json::parse(manager.getOptions().dump()) == before_options);
        manager.endRequestOverrides();
    }
    check_unchanged();
}
