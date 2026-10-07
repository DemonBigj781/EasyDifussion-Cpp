#ifndef __ED_LOGGING_H__
#define __ED_LOGGING_H__

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <string>

#include "stable-diffusion.h"

enum class LogLevel { Verbose, Debug, Info, Warning, Error };

void log_message(LogLevel level, const char* format, ...);
void set_log_level(LogLevel level);
void set_log_level(const std::string& level_str);

// Convenience macros for logging
#define LOG_VERBOSE(...) log_message(LogLevel::Verbose, __VA_ARGS__)
#define LOG_DEBUG(...) log_message(LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...) log_message(LogLevel::Info, __VA_ARGS__)
#define LOG_WARNING(...) log_message(LogLevel::Warning, __VA_ARGS__)
#define LOG_ERROR(...) log_message(LogLevel::Error, __VA_ARGS__)

// Native generation errors are local to the thread executing inference.
void reset_sd_generation_error();
bool sd_generation_error_out_of_vram();
void set_sd_generation_error_out_of_vram(bool value);
std::string sd_generation_error_message(const std::string& fallback);

// SD callback for stable-diffusion.cpp integration
void sd_log_cb(sd_log_level_t level, const char* log, void* data);

#endif