#pragma once

#include <cstdint>
#include <string>

namespace api_test::attention_image {

using DeviceResolver = bool (*)(
    const std::string& requested,
    std::string& backend,
    std::string& description,
    std::string& diagnostic);
using LaunchCounter = std::uint64_t (*)();
using AttentionEnabler = bool (*)();

struct BackendConfiguration {
    const char* label = nullptr;
    const char* default_device = nullptr;
    const char* default_output = nullptr;
    const char* attention_name = nullptr;
    AttentionEnabler enable_attention = nullptr;
    LaunchCounter attention_launch_count = nullptr;
    DeviceResolver resolve_device = nullptr;
};

int run(int argc, char** argv, const BackendConfiguration& backend);

} // namespace api_test::attention_image
