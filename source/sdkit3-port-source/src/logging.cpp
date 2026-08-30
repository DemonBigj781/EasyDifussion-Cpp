#include "logging.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <sstream>

static LogLevel current_log_level = LogLevel::Info;
static std::mutex log_mutex;

const char* log_level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::Verbose:
            return "VERBOSE";
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warning:
            return "WARNING";
        case LogLevel::Error:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
}

const char* log_level_to_color(LogLevel level) {
    switch (level) {
        case LogLevel::Verbose:
            return "\033[35m";  // Magenta
        case LogLevel::Debug:
            return "\033[36m";  // Cyan
        case LogLevel::Info:
            return "\033[32m";  // Green
        case LogLevel::Warning:
            return "\033[33m";  // Yellow
        case LogLevel::Error:
            return "\033[31m";  // Red
        default:
            return "\033[0m";  // Reset
    }
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

void log_message(LogLevel level, const char* format, ...) {
    // Check if we should log this level
    if (level < current_log_level) {
        return;
    }

    std::lock_guard<std::mutex> lock(log_mutex);

    // Format the message
    char buffer[4096];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // Get timestamp
    std::string timestamp = get_timestamp();

    // Print with color (if terminal supports it)
    const char* color = log_level_to_color(level);
    const char* reset = "\033[0m";
    const char* level_str = log_level_to_string(level);

    // Print to stderr for warnings and errors, stdout otherwise
    FILE* out = (level >= LogLevel::Warning) ? stderr : stdout;

    fprintf(out, "[%s] %s%-7s%s %s\n", timestamp.c_str(), color, level_str, reset, buffer);

    fflush(out);
}

void set_log_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex);
    current_log_level = level;
}

void set_log_level(const std::string& level_str) {
    LogLevel level = LogLevel::Info;  // default

    if (level_str == "verbose") {
        level = LogLevel::Verbose;
    } else if (level_str == "debug") {
        level = LogLevel::Debug;
    } else if (level_str == "info") {
        level = LogLevel::Info;
    } else if (level_str == "warning") {
        level = LogLevel::Warning;
    } else if (level_str == "error") {
        level = LogLevel::Error;
    } else {
        fprintf(stderr, "Invalid log level: %s. Using 'info'.\n", level_str.c_str());
    }

    set_log_level(level);
}

// SD callback implementation for stable-diffusion.cpp integration
void sd_log_cb(sd_log_level_t level, const char* log, void* data) {
    // Map sd_log_level_t to LogLevel
    LogLevel mapped_level;
    switch (level) {
        case SD_LOG_DEBUG:
            mapped_level = LogLevel::Debug;
            break;
        case SD_LOG_INFO:
            mapped_level = LogLevel::Info;
            break;
        case SD_LOG_WARN:
            mapped_level = LogLevel::Warning;
            break;
        case SD_LOG_ERROR:
            mapped_level = LogLevel::Error;
            break;
        default:
            mapped_level = LogLevel::Info;
            break;
    }

    // Forward to our logging system (remove trailing newline if present)
    std::string msg = log;
    if (!msg.empty() && msg.back() == '\n') {
        msg.pop_back();
    }
    log_message(mapped_level, "%s", msg.c_str());
}
