#include "platform.hpp"
#include "process.hpp"
#include "json.hpp"
extern "C" {
#include "sha256.h"
}
#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace {
constexpr size_t manifest_limit = 1024 * 1024;
struct Options {
    std::string root, manifest = "launcher.json", bundle, backend;
    bool detect = false, list = false, dry_run = false;
    std::vector<std::string> extra;
};
struct Program { Command command; int port = 0, timeout = 10000; };
struct Bundle {
    std::string id, os, arch, backend;
    int priority = 0, probe_timeout = 5000;
    std::vector<std::string> features;
    Json files;
    std::vector<Program> programs;
    bool has_probe = false;
    Command probe;
};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
void keys(const Json& value, std::initializer_list<const char*> allowed) {
    require(value.is_object(), "Expected JSON object");
    for (auto it = value.begin(); it != value.end(); ++it) {
        bool found = false;
        for (auto key : allowed) found |= it.key() == key;
        require(found, "Unknown manifest key: " + it.key());
    }
}
std::string text(const Json& value) {
    require(value.is_string(), "Expected JSON string");
    auto s = value.get<std::string>();
    require(s.size() <= 32768 && s.find('\0') == std::string::npos, "Invalid or oversized string");
    return s;
}
int number(const Json& obj, const char* key, int fallback, int min, int max) {
    if (!obj.contains(key)) return fallback;
    require(obj[key].is_number_integer(), std::string("Expected integer: ") + key);
    auto n = obj[key].get<int64_t>();
    require(n >= min && n <= max, std::string("Out of range: ") + key);
    return static_cast<int>(n);
}
std::vector<std::string> strings(const Json& obj, const char* key) {
    std::vector<std::string> result;
    if (!obj.contains(key)) return result;
    require(obj[key].is_array() && obj[key].size() <= 256, std::string("Invalid array: ") + key);
    for (const auto& v : obj[key]) result.push_back(text(v));
    return result;
}
void identifier(const std::string& s) {
    require(!s.empty() && s.size() <= 128 && std::all_of(s.begin(), s.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_' || c == '-' || c == '.';
    }), "Invalid identifier: " + s);
}
void relative_path(const std::string& s, bool directory = false) {
    if (directory && s == ".") return;
    require(!s.empty() && s[0] != '/' && s.find_first_of("\\:\0", 0, 3) == std::string::npos,
        "Expected bundle-relative slash-separated path: " + s);
    size_t pos = 0;
    while (pos <= s.size()) {
        auto end = s.find('/', pos);
        auto part = s.substr(pos, end == std::string::npos ? end : end - pos);
        require(!part.empty() && part != "." && part != "..", "Unsafe relative path: " + s);
        if (end == std::string::npos) break;
        pos = end + 1;
    }
}
std::string inside(const std::string& root, const std::string& path, bool directory = false) {
    relative_path(path, directory);
    auto resolved = canonical_path(root + "/" + path);
    auto prefix = root == "/" ? root : root + "/";
    auto comparable = resolved;
    if (on_windows()) {
        auto lower = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };
        std::transform(prefix.begin(), prefix.end(), prefix.begin(), lower);
        std::transform(comparable.begin(), comparable.end(), comparable.begin(), lower);
    }
    require((directory && resolved == root) || comparable.compare(0, prefix.size(), prefix) == 0,
        "Path escapes bundle root: " + path);
    struct stat st{};
    require(stat(resolved.c_str(), &st) == 0 && (directory ? S_ISDIR(st.st_mode) : S_ISREG(st.st_mode)),
        "Wrong file type: " + path);
    return resolved;
}
std::string hash_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "Cannot read file for SHA-256: " + path);
    sha256_t hash;
    sha256_init(&hash);
    unsigned char buffer[65536];
    while (input) {
        input.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
        sha256_update(&hash, buffer, static_cast<size_t>(input.gcount()));
    }
    require(input.eof(), "Error reading file: " + path);
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);
    std::string result;
    constexpr char hex[] = "0123456789abcdef";
    for (auto c : digest) { result += hex[c >> 4]; result += hex[c & 15]; }
    return result;
}
std::string expand(std::string value, const std::string& root) {
    size_t pos = 0;
    auto replacement = native_path(root);
    while ((pos = value.find("{root}", pos)) != std::string::npos) {
        value.replace(pos, 6, replacement);
        pos += replacement.size();
    }
    return value;
}
void environment(Command& command, const Json& obj) {
    if (obj.contains("env")) {
        require(obj["env"].is_object() && obj["env"].size() <= 128, "Invalid environment map");
        for (auto it = obj["env"].begin(); it != obj["env"].end(); ++it) {
            auto key = it.key();
            require(!key.empty() && key.find_first_of("=\0", 0, 2) == std::string::npos,
                "Invalid environment key");
            command.env[key] = text(it.value());
        }
    }
    for (const auto& key : strings(obj, "unset_env")) {
        require(!key.empty() && key.find('=') == std::string::npos, "Invalid unset_env key");
        command.env.erase(key);
        command.unset_env.push_back(key);
    }
}
Command parse_command(const Json& obj, const Json& bundle, const Json& files) {
    keys(obj, {"id", "role", "executable", "args", "cwd", "env", "unset_env",
        "ready_port", "ready_timeout_ms", "timeout_ms"});
    Command command;
    command.id = text(obj.at("id"));
    identifier(command.id);
    command.executable = text(obj.at("executable"));
    relative_path(command.executable);
    require(files.contains(command.executable), "Executable is missing from files/hashes: " + command.executable);
    command.cwd = obj.contains("cwd") ? text(obj["cwd"]) : ".";
    relative_path(command.cwd, true);
    command.args = strings(obj, "args");
    command.unset_env = {"PYTHONHOME", "PYTHONPATH"};
    command.env["PYTHONNOUSERSITE"] = "1";
    environment(command, bundle);
    environment(command, obj);
    return command;
}
std::vector<Bundle> read_manifest(const std::string& root, const std::string& filename) {
    std::ifstream input(inside(root, filename), std::ios::binary);
    require(input.good(), "Cannot read launcher manifest");
    std::string content(manifest_limit + 1, '\0');
    input.read(content.data(), content.size());
    content.resize(static_cast<size_t>(input.gcount()));
    require(content.size() <= manifest_limit && input.eof(), "Manifest too large or unreadable");
    auto document = Json::parse(content, [](int depth, Json::parse_event_t, Json&) {
        require(depth <= 32, "Manifest nesting limit exceeded"); return true;
    });
    keys(document, {"schema", "bundles"});
    require(document.at("schema").is_number_integer() && document.at("schema") == 1, "Unsupported manifest schema");
    const auto& entries = document.at("bundles");
    require(entries.is_array() && !entries.empty() && entries.size() <= 128, "Invalid bundles array");
    std::set<std::string> ids;
    std::vector<Bundle> result;
    for (const auto& obj : entries) {
        keys(obj, {"id", "os", "arch", "backend", "priority", "cpu_features", "files", "programs", "probe", "env", "unset_env"});
        Bundle bundle;
        bundle.id = text(obj.at("id"));
        identifier(bundle.id);
        require(ids.insert(bundle.id).second, "Duplicate bundle id: " + bundle.id);
        bundle.os = text(obj.at("os"));
        bundle.arch = text(obj.at("arch"));
        bundle.backend = text(obj.at("backend"));
        require(std::set<std::string>{"linux", "mac", "win", "freebsd", "openbsd", "netbsd", "any"}.count(bundle.os), "Unknown OS");
        require(std::set<std::string>{"x64", "arm64", "any"}.count(bundle.arch), "Unknown architecture");
        require(std::set<std::string>{"cpu", "cuda", "rocm", "sycl", "vulkan", "metal", "opencl", "openvino", "directml", "oneapi-cpu"}.count(bundle.backend), "Unknown backend");
        bundle.priority = number(obj, "priority", 0, -1000000, 1000000);
        bundle.features = strings(obj, "cpu_features");
        for (const auto& f : bundle.features)
            require(std::set<std::string>{"avx", "avx2", "fma", "bmi2", "sse4.2", "neon"}.count(f), "Unknown CPU feature: " + f);
        bundle.files = obj.at("files");
        require(bundle.files.is_object() && !bundle.files.empty() && bundle.files.size() <= 4096, "Invalid files map");
        for (auto it = bundle.files.begin(); it != bundle.files.end(); ++it) {
            relative_path(it.key());
            auto hash = text(it.value());
            require(hash.size() == 64 && std::all_of(hash.begin(), hash.end(), [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            }), "Invalid SHA-256 for: " + it.key());
        }
        bundle.has_probe = obj.contains("probe");
        require(bundle.backend == "cpu" || bundle.has_probe, "Accelerated bundle requires a compatibility probe: " + bundle.id);
        if (bundle.has_probe) {
            bundle.probe = parse_command(obj["probe"], obj, bundle.files);
            bundle.probe_timeout = number(obj["probe"], "timeout_ms", 5000, 20, 30000);
        }
        const auto& programs = obj.at("programs");
        require(programs.is_array() && !programs.empty() && programs.size() <= 16, "Invalid programs array");
        std::set<std::string> program_ids;
        for (size_t i = 0; i < programs.size(); ++i) {
            const auto& p = programs[i];
            auto role = text(p.at("role"));
            require(role == (i + 1 == programs.size() ? "main" : "service"), "Services must precede exactly one final main program");
            Program program;
            program.command = parse_command(p, obj, bundle.files);
            require(program_ids.insert(program.command.id).second, "Duplicate program id");
            if (role == "service") {
                program.port = number(p, "ready_port", 0, 1, 65535);
                require(program.port != 0, "Service requires ready_port");
                program.timeout = number(p, "ready_timeout_ms", 10000, 20, 120000);
            } else {
                require(!p.contains("ready_port"), "Main program cannot have ready_port");
            }
            bundle.programs.push_back(std::move(program));
        }
        result.push_back(std::move(bundle));
    }
    std::stable_sort(result.begin(), result.end(), [](const Bundle& a, const Bundle& b) { return a.priority > b.priority; });
    return result;
}

Command resolve(Command cmd, const Bundle& bundle, const std::string& root) {
    cmd.executable = inside(root, cmd.executable);
    require(access(cmd.executable.c_str(), X_OK) == 0, "Not executable: " + cmd.executable);
    cmd.cwd = inside(root, cmd.cwd, true);
    for (auto& arg : cmd.args) arg = expand(arg, root);
    for (auto& [key, value] : cmd.env) value = expand(value, root);
    cmd.env["EDCPP_LAUNCH_BUNDLE"] = bundle.id;
    cmd.env["EDCPP_BUNDLE_ROOT"] = native_path(root);
    return cmd;
}
std::string compatible(const Bundle& b, const Host& h, const Options& options) {
    if (!options.bundle.empty() && b.id != options.bundle) return "bundle override differs";
    if (!options.backend.empty() && b.backend != options.backend) return "backend override differs";
    if (b.os != "any" && b.os != h.os) return "OS differs";
    if (b.arch != "any" && b.arch != h.arch) return "process architecture differs";
    for (const auto& f : b.features) if (!h.features.count(f)) return "missing CPU feature: " + f;
    return "";
}
void verify(const Bundle& b, const std::string& root, std::map<std::string, std::string>& hashes) {
    for (auto it = b.files.begin(); it != b.files.end(); ++it) {
        auto file = inside(root, it.key());
        if (!hashes.count(file)) hashes[file] = hash_file(file);
        require(hashes[file] == it.value(), "SHA-256 mismatch: " + it.key());
    }
    for (const auto& p : b.programs) resolve(p.command, b, root);
    if (b.has_probe) resolve(b.probe, b, root);
}
int launch(const Bundle& bundle, const std::string& root, const Options& options) {
    Processes processes;
    for (size_t i = 0; i < bundle.programs.size(); ++i) {
        if (stop_signal) return 128 + stop_signal;
        processes.poll();
        for (const auto& child : processes.children) require(child.running, "Service exited during startup");
        const auto& program = bundle.programs[i];
        auto command = resolve(program.command, bundle, root);
        bool main = i + 1 == bundle.programs.size();
        if (main) command.args.insert(command.args.end(), options.extra.begin(), options.extra.end());
        if (!main) require(!port_ready(program.port), "Service port already occupied: " + std::to_string(program.port));
        std::cerr << "[launcher] starting " << command.id << '\n';
        auto index = processes.start(command);
        if (!main) {
            auto deadline = Clock::now() + std::chrono::milliseconds(program.timeout);
            while (!port_ready(program.port)) {
                if (stop_signal) return 128 + stop_signal;
                processes.poll();
                for (const auto& child : processes.children) require(child.running, "Service exited before readiness");
                require(Clock::now() < deadline, "Service readiness timed out: " + command.id);
                pause_tick();
            }
        } else {
            while (!stop_signal) {
                processes.poll();
                if (!processes.children[index].running) return processes.children[index].result;
                for (size_t j = 0; j < index; ++j) require(processes.children[j].running, "Required service exited");
                pause_tick();
            }
        }
    }
    return 128 + stop_signal;
}
Options arguments(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--") { while (++i < argc) o.extra.emplace_back(argv[i]); break; }
        auto value = [&]() { require(i + 1 < argc, "Missing value for " + arg); return std::string(argv[++i]); };
        if (arg == "--root") o.root = value();
        else if (arg == "--manifest") o.manifest = value();
        else if (arg == "--bundle") o.bundle = value();
        else if (arg == "--backend") o.backend = value();
        else if (arg == "--detect") o.detect = true;
        else if (arg == "--list") o.list = true;
        else if (arg == "--dry-run") o.dry_run = true;
        else throw std::runtime_error("Unknown argument: " + arg);
    }
    require(int(o.detect) + int(o.list) + int(o.dry_run) <= 1, "Choose only one inspection mode");
    return o;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Easy Diffusion autolauncher (manifest schema 1)\n"
                "  --root PATH         Bundle root (default: directory containing launcher)\n"
                "  --manifest PATH     Relative manifest (default: launcher.json)\n"
                "  --bundle ID         Select a specific bundle; compatibility checks still apply\n"
                "  --backend NAME      Restrict backend selection\n"
                "  --detect            Print host OS, executing ABI and CPU features; no manifest needed\n"
                "  --list              Check candidates and hashes; do not run any processes\n"
                "  --dry-run           Run bounded compatibility probes and print selection only\n"
                "  -- ARGS...          Append literal arguments to the selected main program\n";
            return 0;
        }
        auto options = arguments(argc, argv);
        auto host = detect_host();
        Json host_json{{"os", host.os}, {"arch", host.arch}, {"cpu_features", host.features}};
        if (options.detect) { std::cout << host_json.dump(2) << '\n'; return 0; }
        if (options.root.empty()) {
            auto path = executable_path(argv[0]);
            options.root = path.substr(0, path.find_last_of('/'));
        }
        auto root = canonical_path(options.root);
        auto bundles = read_manifest(root, options.manifest);
        std::map<std::string, std::string> hashes;
        Json reports = Json::array();
        const Bundle* selected = nullptr;
        if (!options.list) { install_signal_handlers(); contain_windows_children(); }
        for (const auto& bundle : bundles) {
            auto reason = compatible(bundle, host, options);
            if (reason.empty()) {
                try { verify(bundle, root, hashes); }
                catch (const std::exception& e) { reason = e.what(); }
            }
            if (reason.empty() && !options.list && bundle.has_probe) {
                try {
                    auto result = run_probe(resolve(bundle.probe, bundle, root), bundle.probe_timeout);
                    if (result) reason = "compatibility probe returned " + std::to_string(result);
                } catch (const std::exception& e) { reason = e.what(); }
            }
            if (stop_signal) return 128 + stop_signal;
            reports.push_back({{"id", bundle.id}, {"backend", bundle.backend}, {"compatible", reason.empty()},
                {"reason", reason}, {"probe_required", bundle.has_probe}});
            if (reason.empty() && !options.list) { selected = &bundle; break; }
        }
        if (options.list) {
            std::cout << Json{{"host", host_json}, {"bundles", reports}, {"probes_run", false}}.dump(2) << '\n';
            return 0;
        }
        if (!selected) {
            std::cerr << "[launcher] No compatible bundle. Nothing launched.\n" << reports.dump(2) << '\n';
            return 4;
        }
        if (options.dry_run) {
            Json commands = Json::array();
            for (size_t i = 0; i < selected->programs.size(); ++i) {
                auto command = resolve(selected->programs[i].command, *selected, root);
                if (i + 1 == selected->programs.size()) command.args.insert(command.args.end(), options.extra.begin(), options.extra.end());
                commands.push_back({{"id", command.id}, {"executable", command.executable}, {"args", command.args}, {"cwd", command.cwd}});
            }
            std::cout << Json{{"host", host_json}, {"bundle", selected->id}, {"backend", selected->backend},
                {"programs", commands}, {"candidates", reports}}.dump(2) << '\n';
            return 0;
        }
        std::cerr << "[launcher] selected " << selected->id << " (" << selected->backend << ")\n";
        return launch(*selected, root, options);
    } catch (const std::exception& e) {
        std::cerr << "[launcher] " << e.what() << '\n';
        return stop_signal ? 128 + stop_signal : 2;
    }
}
