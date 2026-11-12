#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"

struct CapturedLog {
    sd_log_level_t level = SD_LOG_DEBUG;
    std::string text;
    int calls = 0;
};

static void capture(sd_log_level_t level, const char* text, void* data) {
    auto& log = *static_cast<CapturedLog*>(data);
    log.level = level;
    log.text = text;
    ++log.calls;
}

static void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main() {
    CapturedLog log;
    sd_set_log_callback(capture, &log);
    int result = 0;
    try {
        // Fail harmlessly on the old implementation before exercising format directives.
        ggml_log_callback_default(GGML_LOG_LEVEL_INFO, "literal %% marker", nullptr);
        require(log.text.find("literal %% marker") != std::string::npos, "GGML text interpreted as printf format");
        const std::vector<std::pair<ggml_log_level, sd_log_level_t>> levels = {
            {GGML_LOG_LEVEL_DEBUG, SD_LOG_DEBUG}, {GGML_LOG_LEVEL_INFO, SD_LOG_INFO},
            {GGML_LOG_LEVEL_WARN, SD_LOG_WARN}, {GGML_LOG_LEVEL_ERROR, SD_LOG_ERROR},
            {GGML_LOG_LEVEL_CONT, SD_LOG_DEBUG}, {GGML_LOG_LEVEL_NONE, SD_LOG_DEBUG}};
        for (const auto& level : levels) {
            for (const std::string& text : {std::string("100% done; %s %d %n %.*s %%"),
                                           std::string("already terminated\n"), std::string(),
                                           std::string("model-\xc3\xa9.gguf 50%")}) {
                log.calls = 0;
                ggml_log_callback_default(level.first, text.c_str(), nullptr);
                require(log.calls == 1 && log.level == level.second, "callback routing changed");
                size_t separator = log.text.find(" - ");
                require(separator != std::string::npos, "source annotation lost");
                std::string expected = text;
                if (expected.empty() || expected.back() != '\n') {
                    expected += '\n';
                }
                require(log.text.substr(separator + 3) == expected, "literal message or newline changed");
            }
        }
        // Ordinary formatted logging must retain its established contract.
        log_printf(SD_LOG_INFO, "example.cpp", 12, "value=%d", 42);
        require(log.text.find("value=42\n") != std::string::npos, "formatted logging changed");
        sd_set_log_callback(nullptr, nullptr);
        ggml_log_callback_default(GGML_LOG_LEVEL_INFO, "no callback: %n", nullptr);
        std::cout << "PASS literal forwarding, six severity routes, empty/Unicode/newline payloads, formatted logs and unset callback\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        result = 1;
    }
    sd_set_log_callback(nullptr, nullptr);
    return result;
}
