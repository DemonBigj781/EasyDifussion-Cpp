#pragma once
#include <set>
#include <string>

struct Host {
    std::string os;
    std::string arch;
    std::set<std::string> features;
};
Host detect_host();
std::string executable_path(const char* argv0);
std::string canonical_path(const std::string& path);
std::string native_path(std::string path);
bool on_windows();
void contain_windows_children();
