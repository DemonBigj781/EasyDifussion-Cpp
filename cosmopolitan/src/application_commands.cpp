/* SPDX-License-Identifier: MIT */
#ifndef _COSMO_SOURCE
#define _COSMO_SOURCE
#endif
#include "application.h"
#include "runtime.h"
#include "shell.h"
#include "json.hpp"
#ifdef __COSMOPOLITAN__
#include <libc/dce.h>
#endif
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
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
constexpr size_t max_request_bytes = 1024 * 1024;
constexpr size_t max_image_bytes = 64 * 1024 * 1024;

void require(bool valid, const std::string &message) {
    if (!valid) throw std::invalid_argument(message);
}

fs::path resolve_path(const char *directory, std::string value) {
    require(!value.empty() && value.find('\0') == std::string::npos, "Path must not be empty or contain NUL");
#ifdef __COSMOPOLITAN__
    if (IsWindows()) {
        for (char &c : value) if (c == '\\') c = '/';
        if (value.size() >= 2 && value[1] == ':') {
            require(value.size() >= 3 && value[2] == '/' && std::isalpha(static_cast<unsigned char>(value[0])),
                    "Use an absolute Windows drive path");
            value = "/" + value.substr(0, 1) + value.substr(2);
        }
    }
#endif
    fs::path path(value);
    if (path.is_relative()) path = fs::path(directory) / path;
    return path.lexically_normal();
}

Json parse_object(const std::string &text) {
    require(text.size() <= max_request_bytes, "Request exceeds 1 MiB");
    std::vector<std::set<std::string>> keys;
    Json value = Json::parse(text, [&](int, Json::parse_event_t event, Json &part) {
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        else if (event == Json::parse_event_t::object_end) keys.pop_back();
        else if (event == Json::parse_event_t::key)
            require(keys.back().insert(part.get<std::string>()).second, "Duplicate request key: " + part.get<std::string>());
        return true;
    });
    require(value.is_object(), "Request must be a JSON object");
    return value;
}

Json read_request(const fs::path &path) {
    require(fs::is_regular_file(path) && fs::file_size(path) <= max_request_bytes,
            "Request file must be a regular file of at most 1 MiB");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read request file: " + path.string());
    std::string text((std::istreambuf_iterator<char>(input)), {});
    if (input.bad()) throw std::runtime_error("Failed reading request file: " + path.string());
    return parse_object(text);
}

Json set_patch(const std::string &key, const std::string &value) {
    require(!key.empty(), "A setting key is required");
    Json patch = Json::object();
    Json *field = &patch;
    size_t start = 0;
    for (;;) {
        size_t end = key.find('.', start);
        std::string part = key.substr(start, end == std::string::npos ? end : end - start);
        require(!part.empty() && part.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789") == std::string::npos,
                "Use a dotted configuration key, for example inference.image.steps");
        field = &(*field)[part];
        if (end == std::string::npos) break;
        start = end + 1;
    }
    *field = Json::parse(value, nullptr, false);
    if (field->is_discarded()) *field = value;
    return patch;
}

int print_json(FILE *file, const std::string &text) {
    if (std::fwrite(text.data(), 1, text.size(), file) != text.size() ||
        std::fputc('\n', file) == EOF || std::fflush(file)) return 1;
    return 0;
}

struct Reply {
    int status;
    std::unique_ptr<char, decltype(&cosmo_app_reply_free)> bytes{nullptr, &cosmo_app_reply_free};
    explicit Reply(cosmo_app *app, cosmo_app_operation operation, const Json &body) {
        const std::string request = body.dump();
        char *result = nullptr;
        status = cosmo_app_request(app, operation, request.c_str(), &result);
        bytes.reset(result);
        if (!result) throw std::runtime_error("Application could not allocate a response");
    }
    bool ok() const { return status >= 200 && status < 300; }
    int print(FILE *output, FILE *error) const {
        if (print_json(ok() ? output : error, bytes.get())) return 1;
        return ok() ? 0 : status == 409 ? 3 : status >= 500 ? 1 : 2;
    }
};

void command_help(FILE *output) {
    std::fputs("  status, devices, models [refresh]\n"
               "  config show | config set DOTTED.KEY VALUE | config patch JSON\n"
               "  options show | options set KEY VALUE | options patch JSON\n"
               "  defaults show | defaults save JSON\n"
               "  infer image --output FILE.png [--prompt TEXT] [--model INDEXED_NAME]\n"
               "      [--width N --height N --steps N --seed N --cfg-scale N]\n"
               "      [--negative-prompt TEXT --sampler NAME --scheduler NAME]\n"
               "      [--device SELECTOR --task-id ID]\n"
               "  infer text [--model FILE.gguf --prompt TEXT --tokens N --threads N]\n"
               "      [--device SELECTOR --task-id ID --output FILE.txt]\n"
               "  infer image|text --request JSON | --request-file FILE [--output PATH]\n"
               "  progress TASK_ID, cancel TASK_ID\n"
               "Image requests inherit saved recipe/checkpoint defaults. Explicit request\n"
               "values apply once. defaults save accepts {options:{...},inference:{...}}.\n"
               "Output files must be new; config writes replace the existing settings\n"
               "atomically. Inference waits here while HTTP progress/cancel stays live.\n",
               output);
}

std::vector<unsigned char> decode_png(const std::string &text) {
    require(!text.empty() && text.size() % 4 == 0 && text.size() <= (max_image_bytes + 2) / 3 * 4,
            "Invalid or oversized PNG response");
    std::vector<unsigned char> result;
    result.reserve(text.size() / 4 * 3);
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < text.size(); i += 4) {
        unsigned int value = 0;
        int padding = 0;
        for (size_t j = 0; j < 4; ++j) {
            if (text[i + j] == '=') {
                require(i + 4 == text.size() && j >= 2, "Invalid PNG base64 padding");
                ++padding; value <<= 6;
            } else {
                const size_t digit = alphabet.find(text[i + j]);
                require(!padding && digit != std::string::npos, "Invalid PNG base64 character");
                value = (value << 6) | static_cast<unsigned int>(digit);
            }
        }
        require(padding < 3, "Invalid PNG base64 padding");
        if (padding == 2) require((value & 0xffff) == 0, "Invalid PNG base64 tail");
        if (padding == 1) require((value & 0xff) == 0, "Invalid PNG base64 tail");
        result.push_back(static_cast<unsigned char>(value >> 16));
        if (padding < 2) result.push_back(static_cast<unsigned char>(value >> 8));
        if (padding < 1) result.push_back(static_cast<unsigned char>(value));
    }
    const unsigned char signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    require(result.size() >= sizeof(signature) && result.size() <= max_image_bytes &&
            !std::memcmp(result.data(), signature, sizeof(signature)), "Application did not return a PNG");
    return result;
}

void save_new_file(const fs::path &path, const unsigned char *bytes, size_t count) {
    int fd = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "Create output " + path.string());
    try {
        size_t offset = 0;
        while (offset < count) {
            ssize_t written = write(fd, bytes + offset, count - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::system_error(errno ? errno : EIO, std::generic_category(), "Write output");
            offset += static_cast<size_t>(written);
        }
        if (fsync(fd)) throw std::system_error(errno, std::generic_category(), "Sync output");
        if (close(fd)) { fd = -1; throw std::system_error(errno, std::generic_category(), "Close output"); }
    } catch (...) {
        if (fd >= 0) close(fd);
        unlink(path.c_str());
        throw;
    }
}

int inference_command(cosmo_app *app, const char *directory, int argc, char **argv, FILE *output, FILE *error) {
    require(argc >= 2 && (!std::strcmp(argv[1], "image") || !std::strcmp(argv[1], "text")),
            "Usage: infer image|text [OPTIONS]");
    const bool image = !std::strcmp(argv[1], "image");
    Json request = Json::object();
    std::string destination;
    std::string backend_choice, device_choice;
    std::set<std::string> seen;
    bool raw_request = false;
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") { command_help(output); return 0; }
        require(i + 1 < argc && seen.insert(flag).second, "Missing value or duplicate inference option: " + flag);
        const std::string value = argv[++i];
        if (flag == "--output") { require(!value.empty(), "Output path must not be empty"); destination = value; continue; }
        if (flag == "--request" || flag == "--request-file") {
            require(request.empty() && backend_choice.empty() && device_choice.empty() && !raw_request,
                    "A raw request cannot be combined with individual inference fields");
            request = flag == "--request" ? parse_object(value) : read_request(resolve_path(directory, value));
            raw_request = true; continue;
        }
        require(!raw_request, "A raw request cannot be combined with individual inference fields");
        if (flag == "--model") {
            require(!value.empty(), "Model must not be empty");
            if (image) request["override_settings"]["sd_model_checkpoint"] = value;
            else request["model"] = resolve_path(directory, value).string();
        } else if (flag == "--prompt") request["prompt"] = value;
        else if (flag == "--negative-prompt" && image) request["negative_prompt"] = value;
        else if (flag == "--sampler" && image) request["sampler_name"] = value;
        else if (flag == "--scheduler" && image) request["scheduler"] = value;
        else if (flag == "--task-id") request["force_task_id"] = value;
        else if (flag == "--device" || flag == "--backend") {
            if (image) {
                require(!value.empty(), "Compute selection must not be empty");
                if (flag == "--backend") {
                    require(value == "cpu" || value == "webgpu", "Backend must be cpu or webgpu");
                    backend_choice = value;
                } else device_choice = value == "cpu" ? "CPU" : value;
            } else if (flag == "--device") {
                request["device"] = value == "cpu" ? "CPU" : value;
                if (!seen.count("--backend") && value != "auto")
                    request["backend"] = value == "CPU" || value == "cpu" ? "cpu" : "webgpu";
            } else request["backend"] = value;
        } else {
            std::string field;
            if (image && (flag == "--width" || flag == "--height" || flag == "--steps" || flag == "--seed")) field = flag.substr(2);
            else if (image && flag == "--cfg-scale") field = "cfg_scale";
            else if (!image && flag == "--tokens") field = "max_tokens";
            else if (!image && flag == "--threads") field = "threads";
            else throw std::invalid_argument("Unknown inference option: " + flag);
            Json number = Json::parse(value, nullptr, false);
            require(number.is_number() && (field == "cfg_scale" || number.is_number_integer()), "Invalid number for " + flag);
            request[field] = number;
        }
    }
    if (image && !raw_request) {
        if (!backend_choice.empty() && device_choice != "auto" && !device_choice.empty()) {
            require((backend_choice == "cpu") == (device_choice == "CPU"), "Backend and device do not match");
        }
        if (!device_choice.empty() && device_choice != "auto") request["backend"] = device_choice;
        else if (!backend_choice.empty()) request["backend"] = backend_choice;
    }
    if (!image && raw_request && request.contains("model")) {
        require(request["model"].is_string(), "Text model must be a file path");
        request["model"] = resolve_path(directory, request["model"].get<std::string>()).string();
    }
    require(!image || !destination.empty(), "infer image requires --output FILE.png");
    fs::path path;
    if (!destination.empty()) {
        path = resolve_path(directory, destination);
        require(!fs::exists(path) && !fs::is_symlink(path), "Output already exists: " + path.string());
        require(fs::is_directory(path.parent_path()), "Output directory does not exist: " + path.parent_path().string());
    }
    Reply reply(app, image ? COSMO_APP_IMAGE : COSMO_APP_TEXT, request);
    if (!reply.ok() || destination.empty()) return reply.print(output, error);
    Json result = Json::parse(reply.bytes.get());
    size_t bytes = 0;
    if (image) {
        require(result.contains("images") && result["images"].is_array(), "Image response has no images array");
        if (result["images"].empty()) return reply.print(output, error);
        require(result["images"].size() == 1 && result["images"][0].is_string(), "Shell output supports one image per request");
        auto png = decode_png(result["images"][0].get<std::string>());
        save_new_file(path, png.data(), png.size()); bytes = png.size();
        result.erase("images");
    } else {
        require(result.contains("text") && result["text"].is_string(), "Text response has no generated text");
        const std::string text = result["text"].get<std::string>();
        save_new_file(path, reinterpret_cast<const unsigned char *>(text.data()), text.size()); bytes = text.size();
    }
    result["output"] = path.string(); result["output_bytes"] = bytes;
    return print_json(output, result.dump());
}

int application_command(void *context, const char *directory, int argc, char **argv, FILE *output, FILE *error) {
    try {
        cosmo_app *app = static_cast<cosmo_app *>(context);
        require(argc > 0 && directory, "Invalid application command");
        const std::string name = argv[0];
        if (name == "help") { command_help(output); return 0; }
        if (name == "infer") return inference_command(app, directory, argc, argv, output, error);
        Json request = Json::object();
        cosmo_app_operation operation;
        if (name == "status" || name == "devices") {
            require(argc == 1, "Usage: " + name);
            operation = name == "status" ? COSMO_APP_STATUS : COSMO_APP_DEVICES;
        } else if (name == "models") {
            require(argc == 1 || (argc == 2 && !std::strcmp(argv[1], "refresh")), "Usage: models [refresh]");
            operation = argc == 1 ? COSMO_APP_MODELS : COSMO_APP_MODELS_REFRESH;
        } else if (name == "progress" || name == "cancel") {
            require(argc == 2 && *argv[1], "Usage: " + name + " TASK_ID");
            operation = name == "progress" ? COSMO_APP_PROGRESS : COSMO_APP_CANCEL;
            request["id_task"] = argv[1];
            if (name == "progress") request["live_preview"] = false;
        } else if (name == "config" || name == "options") {
            const std::string action = argc > 1 ? argv[1] : "show";
            if (action == "show") {
                require(argc <= 2, "Usage: " + name + " show");
                operation = name == "config" ? COSMO_APP_CONFIG_GET : COSMO_APP_OPTIONS_GET;
            } else {
                if (action == "set") { require(argc == 4, "Usage: " + name + " set KEY VALUE"); request = set_patch(argv[2], argv[3]); }
                else if (action == "patch") { require(argc == 3, "Usage: " + name + " patch JSON"); request = parse_object(argv[2]); }
                else throw std::invalid_argument("Unknown " + name + " action");
                if (name == "config" && request.size() == 1 && request.contains("options")) {
                    Json options = request["options"]; request = std::move(options); operation = COSMO_APP_OPTIONS_SET;
                } else operation = name == "config" ? COSMO_APP_CONFIG_SET : COSMO_APP_OPTIONS_SET;
            }
        } else if (name == "defaults") {
            const std::string action = argc > 1 ? argv[1] : "show";
            if (action == "show") {
                require(argc <= 2, "Usage: defaults show");
                Reply reply(app, COSMO_APP_CONFIG_GET, request);
                if (!reply.ok()) return reply.print(output, error);
                return print_json(output, Json::parse(reply.bytes.get()).at("effective").at("inference").dump());
            }
            require(action == "save" && argc == 3, "Usage: defaults save JSON");
            operation = COSMO_APP_SETTINGS_SET; request = parse_object(argv[2]);
        } else return 127;
        return Reply(app, operation, request).print(output, error);
    } catch (const std::exception &exception) {
        print_json(error, Json({{"message", exception.what()}}).dump()); return 2;
    } catch (...) {
        std::fputs("{\"message\":\"Unknown application command failure\"}\n", error); return 1;
    }
}

struct Client { std::vector<char *> arguments; bool shell; };
int run_client(cosmo_app *app, void *opaque) {
    Client &client = *static_cast<Client *>(opaque);
    if (client.shell) {
        const cosmo_shell_services services{app, application_command};
        return cosmo_shell_run(static_cast<int>(client.arguments.size() - 1), client.arguments.data(), &services);
    }
    std::unique_ptr<char, decltype(&std::free)> directory(getcwd(nullptr, 0), &std::free);
    if (!directory) { std::perror("working directory"); return 1; }
    return application_command(app, directory.get(), static_cast<int>(client.arguments.size() - 1),
                                client.arguments.data(), stdout, stderr);
}
} // namespace

extern "C" int cosmo_application_main(int argc, char **argv) {
    try {
        require(argc >= 1, "An application command is required");
        Client client{{}, !std::strcmp(argv[0], "shell")};
        bool serve = false;
        for (int i = 0; i < argc; ++i) {
            if (client.shell && !std::strcmp(argv[i], "--serve")) { require(!serve, "Duplicate --serve"); serve = true; }
            else {
                client.arguments.push_back(argv[i]);
                if (client.shell && (!std::strcmp(argv[i], "-c") || !std::strcmp(argv[i], "--file")) && i + 1 < argc)
                    client.arguments.push_back(argv[++i]);
            }
        }
        client.arguments.push_back(nullptr);
        if (client.shell && client.arguments.size() == 3 &&
            (!std::strcmp(client.arguments[1], "--help") || !std::strcmp(client.arguments[1], "-h"))) {
            const cosmo_shell_services services{nullptr, application_command};
            return cosmo_shell_run(2, client.arguments.data(), &services);
        }
        char error[1024] = {};
        std::unique_ptr<cosmo_app, decltype(&cosmo_app_destroy)> app(cosmo_app_create(error, sizeof(error)), &cosmo_app_destroy);
        if (!app) { std::fprintf(stderr, "Application: %s\n", error); return 1; }
        const int result = cosmo_app_run(app.get(), serve, run_client, &client, error, sizeof(error));
        if (error[0]) std::fprintf(stderr, "Application: %s\n", error);
        return result;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Application: %s\n", error.what()); return 2;
    } catch (...) {
        std::fputs("Application: unknown failure\n", stderr); return 1;
    }
}
