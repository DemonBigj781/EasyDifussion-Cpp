#ifdef __COSMOPOLITAN__
#ifndef _COSMO_SOURCE
#define _COSMO_SOURCE
#endif
#endif
#include "config.hpp"
#ifdef COSMO_WEBGPU_BACKEND
#include "cosmo-webgpu.h"
#endif
#include "json.hpp"
#ifdef __COSMOPOLITAN__
#include <libc/dce.h>
#endif
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
constexpr size_t kMaxConfigBytes = 1024 * 1024;
std::mutex config_mutex;
Json saved, effective;
fs::path config_path;
std::string disk_contents;
bool initialized = false;
std::string provider = "auto", device = "auto", backend = "cpu";
std::vector<std::string> arguments;
std::vector<char *> argument_pointers;

Json defaults() {
    return {{"schema", 1}, {"server", {{"port", 8188}, {"log_level", "info"}}},
            {"models", {{"checkpoint_dir", "models/checkpoints"}}},
            {"compute", {{"backend", "cpu"}, {"provider", "auto"}, {"device", "auto"}}},
            {"options", {{"sd_model_checkpoint", ""}, {"live_previews_enable", false},
                         {"show_progress_every_n_steps", 5}, {"CLIP_stop_at_last_layers", -1},
                         {"sdxl_clip_l_skip", false}, {"samples_format", "png"},
                         {"forge_additional_modules", Json::array()}}}};
}

void require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

void keys(const Json &value, const Json &pattern, const std::string &where, bool complete) {
    require(value.is_object(), where + " must be an object");
    for (const auto &item : value.items())
        require(pattern.contains(item.key()), "Unknown configuration key: " + where + "." + item.key());
    if (complete)
        for (const auto &item : pattern.items())
            require(value.contains(item.key()), "Missing configuration key: " + where + "." + item.key());
}

std::string string_value(const Json &value, const std::string &where, size_t max = 4096) {
    require(value.is_string(), where + " must be a string");
    std::string text = value.get<std::string>();
    require(text.size() <= max && text.find('\0') == std::string::npos, where + " is too long or contains NUL");
    return text;
}

long long integer(const Json &value, const std::string &where, long long low, long long high) {
    require(value.is_number_integer(), where + " must be an integer");
    if (value.is_number_unsigned()) require(value.get<unsigned long long>() <= static_cast<unsigned long long>(high), where + " is out of range");
    const auto result = value.get<long long>();
    require(result >= low && result <= high, where + " is out of range");
    return result;
}

void validate(const Json &value) {
    const auto base = defaults();
    keys(value, base, "config", true);
    integer(value.at("schema"), "schema", 1, 1);
    for (const auto *section : {"server", "models", "compute", "options"})
        keys(value.at(section), base.at(section), section, true);
    integer(value.at("server").at("port"), "server.port", 1, 65535);
    const auto log = string_value(value["server"]["log_level"], "server.log_level");
    require(log == "verbose" || log == "debug" || log == "info" || log == "warning" || log == "error", "Unsupported server.log_level");
    require(!string_value(value["models"]["checkpoint_dir"], "models.checkpoint_dir").empty(), "models.checkpoint_dir must not be empty");
    const auto kind = string_value(value["compute"]["backend"], "compute.backend");
    const auto policy = string_value(value["compute"]["provider"], "compute.provider");
    const auto selector = string_value(value["compute"]["device"], "compute.device", 128);
    require(kind == "cpu" || kind == "webgpu", "compute.backend must be cpu or webgpu");
    require(policy == "auto" || policy == "native" || policy == "embedded", "compute.provider must be auto, native or embedded");
    require(!selector.empty() && selector.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-") == std::string::npos,
            "compute.device must be auto or an exact registry device selector");
    require(kind != "cpu" || selector == "auto" || selector == "CPU", "A CPU backend requires compute.device auto or CPU");
    const auto &options = value["options"];
    string_value(options["sd_model_checkpoint"], "options.sd_model_checkpoint");
    require(options["live_previews_enable"].is_boolean() && !options["live_previews_enable"].get<bool>(), "Live previews are unsupported; options.live_previews_enable must be false");
    integer(options["show_progress_every_n_steps"], "options.show_progress_every_n_steps", 5, 5);
    integer(options["CLIP_stop_at_last_layers"], "options.CLIP_stop_at_last_layers", -1, 32);
    require(options["sdxl_clip_l_skip"].is_boolean(), "options.sdxl_clip_l_skip must be boolean");
    require(string_value(options["samples_format"], "options.samples_format") == "png", "Only PNG output is supported");
    require(options["forge_additional_modules"].is_array() && options["forge_additional_modules"].empty(), "Companion modules are unsupported in portable settings");
}

Json parse(const std::string &text) {
    require(text.size() <= kMaxConfigBytes, "Configuration exceeds 1 MiB");
    try {
        std::vector<std::set<std::string>> object_keys;
        return Json::parse(text, [&](int, Json::parse_event_t event, Json &value) {
            if (event == Json::parse_event_t::object_start) object_keys.emplace_back();
            else if (event == Json::parse_event_t::object_end) object_keys.pop_back();
            else if (event == Json::parse_event_t::key)
                require(object_keys.back().insert(value.get<std::string>()).second,
                        "Duplicate JSON configuration key: " + value.get<std::string>());
            return true;
        });
    }
    catch (const Json::exception &error) { throw std::invalid_argument(std::string("Invalid configuration JSON: ") + error.what()); }
}

std::string read_disk(const fs::path &path) {
    const auto size = fs::file_size(path);
    require(size <= kMaxConfigBytes, "Configuration exceeds 1 MiB");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read configuration: " + path.string());
    std::string text((std::istreambuf_iterator<char>(input)), {});
    if (input.bad()) throw std::runtime_error("Failed reading configuration: " + path.string());
    return text;
}

void persist(const Json &value, bool creating = false) {
    validate(value);
    if (creating) require(!fs::exists(config_path), "Configuration already exists; init does not overwrite it");
    else require(fs::exists(config_path) && read_disk(config_path) == disk_contents,
                 "Configuration changed outside this process; restart to load it before saving");
    fs::create_directories(config_path.parent_path());
    const std::string text = value.dump(2) + "\n";
    std::string temporary = config_path.string() + ".tmp.XXXXXX";
    std::vector<char> name(temporary.begin(), temporary.end()); name.push_back(0);
    int fd = mkstemp(name.data());
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "Create configuration temporary file");
    try {
        size_t offset = 0;
        while (offset < text.size()) {
            const ssize_t count = write(fd, text.data() + offset, text.size() - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::system_error(errno ? errno : EIO, std::generic_category(), "Write configuration");
            offset += static_cast<size_t>(count);
        }
        if (fsync(fd)) throw std::system_error(errno, std::generic_category(), "Sync configuration");
        if (close(fd)) { fd = -1; throw std::system_error(errno, std::generic_category(), "Close configuration"); }
        fd = -1;
        if (rename(name.data(), config_path.c_str())) throw std::system_error(errno, std::generic_category(), "Replace configuration");
    } catch (...) {
        if (fd >= 0) close(fd);
        unlink(name.data());
        throw;
    }
    disk_contents = text;
    saved = value;
}

Json merged(Json value, const Json &patch) {
    keys(patch, defaults(), "config", false);
    for (const auto &section : patch.items()) {
        if (section.key() == "schema") { value["schema"] = section.value(); continue; }
        keys(section.value(), defaults().at(section.key()), section.key(), false);
        for (const auto &item : section.value().items()) value[section.key()][item.key()] = item.value();
    }
    validate(value);
    return value;
}

fs::path portable_path(std::string value) {
#ifdef __COSMOPOLITAN__
    if (IsWindows()) {
        // libc++ uses POSIX path parsing in this APE. Convert Windows drive
        // spelling before absolute()/is_relative(), then use Cosmo's /C/... form.
        for (char &character : value) if (character == '\\') character = '/';
        if (value.size() >= 2 && value[1] == ':') {
            require(value.size() >= 3 && value[2] == '/' &&
                    ((value[0] >= 'A' && value[0] <= 'Z') || (value[0] >= 'a' && value[0] <= 'z')),
                    "Use an absolute Windows drive path, not a drive-relative path");
            value = "/" + value.substr(0, 1) + value.substr(2);
        }
    }
#endif
    return fs::path(value);
}

Json document() {
    require(initialized, "Configuration has not been initialized");
    Json saved_runtime = saved; saved_runtime.erase("options");
    Json effective_runtime = effective; effective_runtime.erase("options");
    // Relative saved model paths resolve beside the config, not the launcher cwd.
    fs::path path = portable_path(saved_runtime["models"]["checkpoint_dir"].get<std::string>());
    if (path.is_relative()) path = config_path.parent_path() / path;
    saved_runtime["models"]["checkpoint_dir"] = path.lexically_normal().string();
    return {{"schema", 1}, {"config_path", config_path.string()}, {"saved", saved}, {"effective", effective},
            {"restart_required", saved_runtime != effective_runtime},
            {"backend_config", {{"platform", saved["compute"]["backend"]}}},
            {"available_backends", Json::array({"cpu", "webgpu"})},
            {"device_selection_note", "Exact registry selectors are validated when used; names may change when hardware changes."}};
}

bool common_flag(const std::string &flag) {
    return flag == "--config" || flag == "--provider" || flag == "--device" || flag == "--backend" ||
           flag == "--port" || flag == "--ckpt-dir" || flag == "--log-level";
}

bool no_value(const std::string &flag) {
    static const std::set<std::string> flags = {"--help", "-h", "--require-hardware", "--all", "--report-tokens", "--list-devices", "--backend-info", "--image-vae-on-cpu", "--no-half", "--no-half-vae", "--vae-tiling", "--offload-to-cpu", "--mmap", "--no-mmap", "--keep-model-loaded", "--flash-attention", "--diffusion-fa", "--sage-attention", "--stream-layers", "--cuda-malloc", "--cuda-unified-memory", "--xformers", "--split-attention", "--control-net-cpu", "--image-clip-on-cpu", "--image-clip-vision-on-cpu", "--image-ip-adapter-on-cpu", "--video-clip-on-cpu", "--video-vae-on-cpu", "--video-offload-to-cpu", "--video-stream-layers", "--chroma-disable-dit-mask"};
    return flags.count(flag) != 0;
}
} // namespace

extern "C" const char *cosmo_config_provider() { return provider.c_str(); }
extern "C" const char *cosmo_config_device() { return device.c_str(); }
extern "C" const char *cosmo_config_backend() { return backend.c_str(); }
extern "C" const char *cosmo_config_sdkit_device() {
    if (backend == "cpu") return "cpu";
#ifdef COSMO_WEBGPU_BACKEND
    if (cosmo_webgpu_select_device(device.c_str())) return nullptr;
    return cosmo_webgpu_selected_device();
#else
    std::fputs("Configuration: WebGPU is not linked in this build\n", stderr);
    return nullptr;
#endif
}

extern "C" int cosmo_config_prepare(int *argc, char ***argv) {
    try {
        // The native trainer retains its own device/argument contract. It is
        // not yet a consumer of the persisted inference configuration.
        if (*argc > 1 && !std::strcmp((*argv)[1], "train")) return -1;
        Json overrides = Json::object();
        std::string selected_path = "easy-diffusion.json";
        std::set<std::string> seen;
        arguments.clear(); arguments.emplace_back((*argv)[0]);
        for (int i = 1; i < *argc; ++i) {
            const std::string flag((*argv)[i]);
            if (common_flag(flag)) {
                require(seen.insert(flag).second, "Duplicate configuration option: " + flag);
                require(i + 1 < *argc, "Missing value for " + flag);
                const std::string value((*argv)[++i]);
                require(!value.empty(), "Empty value for " + flag);
                if (flag == "--config") selected_path = value;
                else if (flag == "--backend" || flag == "--provider" || flag == "--device") overrides["compute"][flag.substr(2)] = value;
                else if (flag == "--ckpt-dir") overrides["models"]["checkpoint_dir"] = fs::absolute(portable_path(value)).lexically_normal().string();
                else if (flag == "--log-level") overrides["server"]["log_level"] = value;
                else {
                    size_t used = 0; const long long port = std::stoll(value, &used);
                    require(used == value.size(), "Invalid --port"); overrides["server"]["port"] = port;
                }
            } else {
                arguments.push_back(flag);
                // Keep command-specific option values opaque (including prompts
                // whose literal text happens to look like a global option).
                if (flag.rfind("--", 0) == 0 && !no_value(flag) && i + 1 < *argc)
                    arguments.emplace_back((*argv)[++i]);
            }
        }
        const std::string command = arguments.size() > 1 ? arguments[1] : "--help";
        require(command != "train", "Place train immediately after the executable; inference config flags do not apply to the trainer");
        const bool verification = command == "--self-test" || command == "webgpu-test" || command == "webgpu-inplace-test";
        const bool config_command = command == "config";
        const bool device_diagnostic = command == "webgpu-device-test";
        if (device_diagnostic && !seen.count("--backend")) overrides["compute"]["backend"] = "webgpu";
        const bool ordinary = command == "sdkit" || command == "image" || command == "llama" || command == "devices";
        bool help = command == "--help" || command == "-h";
        for (const auto &arg : arguments) help = help || arg == "--help" || arg == "-h";
        config_path = fs::absolute(portable_path(selected_path)).lexically_normal();
        saved = defaults();
        const std::string action = config_command && arguments.size() > 2 ? arguments[2] : "show";
        if (config_command) require(arguments.size() <= 3, "Usage: config init|show|validate|defaults [--config PATH]");
        if (config_command) require(action == "init" || action == "show" || action == "validate" || action == "defaults", "Unknown config action: " + action);
        if (config_command && action == "defaults") { std::cout << defaults().dump(2) << '\n'; return 0; }
        if ((ordinary || config_command || device_diagnostic) && !help) {
            if (fs::exists(config_path)) {
                require(!(config_command && action == "init"), "Configuration already exists; init does not overwrite it");
                disk_contents = read_disk(config_path); saved = parse(disk_contents); validate(saved);
            } else if (!device_diagnostic) {
                require(!config_command || action == "init", "Configuration does not exist; run config init first");
                if (config_command) saved = merged(saved, overrides);
                (void)merged(saved, overrides); // Reject invalid CLI before creating a file.
                persist(saved, true);
                std::cerr << "Created configuration: " << config_path.string() << '\n';
            }
        }
        effective = merged(saved, overrides);
        fs::path checkpoints = portable_path(effective["models"]["checkpoint_dir"].get<std::string>());
        if (checkpoints.is_relative()) checkpoints = config_path.parent_path() / checkpoints;
        effective["models"]["checkpoint_dir"] = checkpoints.lexically_normal().string();
        initialized = true;
        provider = verification ? "embedded" : effective["compute"]["provider"].get<std::string>();
        device = verification ? "auto" : effective["compute"]["device"].get<std::string>();
        backend = effective["compute"]["backend"].get<std::string>();
        if (config_command) {
            require(action == "init" || action == "show" || action == "validate", "Unknown config action: " + action);
            std::cout << document().dump(2) << '\n'; return 0;
        }
        if (ordinary && !help) {
            arguments.insert(arguments.end(), {"--backend", backend});
            if (command == "image" || command == "llama") arguments.insert(arguments.end(), {"--device", device});
            if (command == "sdkit" || command == "devices") {
                arguments.insert(arguments.end(), {"--port", std::to_string(effective["server"]["port"].get<int>()),
                    "--log-level", effective["server"]["log_level"].get<std::string>(),
                    "--ckpt-dir", effective["models"]["checkpoint_dir"].get<std::string>()});
                if (command == "devices") arguments.emplace_back("--list-devices");
            }
        }
        argument_pointers.clear();
        for (auto &arg : arguments) argument_pointers.push_back(arg.data());
        argument_pointers.push_back(nullptr);
        *argc = static_cast<int>(arguments.size()); *argv = argument_pointers.data();
        return -1;
    } catch (const std::exception &error) {
        std::cerr << "Configuration: " << error.what() << '\n'; return 2;
    }
}

std::string cosmo_config_document() {
    std::lock_guard<std::mutex> lock(config_mutex); return document().dump();
}

std::string cosmo_config_update(const std::string &text) {
    std::lock_guard<std::mutex> lock(config_mutex);
    require(initialized, "Configuration has not been initialized");
    Json patch = parse(text);
    // Legacy settings script field is accepted as an explicit backend alias.
    if (patch.is_object() && patch.contains("backend_platform")) {
        require(patch.size() == 1, "backend_platform cannot be combined with schema sections");
        patch = {{"compute", {{"backend", patch["backend_platform"]}}}};
    }
    require(!patch.contains("options"), "Use the native options endpoint to update live generation settings");
    persist(merged(saved, patch));
    return document().dump();
}

std::string cosmo_config_options() {
    std::lock_guard<std::mutex> lock(config_mutex);
    require(initialized, "Configuration has not been initialized");
    return effective["options"].dump();
}

void cosmo_config_set_options(const std::string &text) {
    std::lock_guard<std::mutex> lock(config_mutex);
    require(initialized, "Configuration has not been initialized");
    const Json patch = parse(text);
    const Json candidate = merged(saved, {{"options", patch}});
    persist(candidate);
    effective["options"] = candidate["options"];
}
