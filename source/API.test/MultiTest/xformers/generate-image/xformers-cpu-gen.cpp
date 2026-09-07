#include "xformers-image-gen.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace {

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool resolve_cpu(
        const std::string& requested,
        std::string& backend,
        std::string& description,
        std::string& diagnostic) {
    const std::string normalized = lower_copy(requested);
    if (normalized != "cpu" && normalized != "cpu0" && normalized != "0") {
        diagnostic = "CPU generator accepts -d cpu, cpu0, or 0";
        return false;
    }
    backend = "cpu";
    description = "cpu";
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const api_test::attention_image::BackendConfiguration backend{
        "CPU",
        "cpu",
        "xformers-cpu-output.ppm",
        "reference memory-efficient attention",
        nullptr,
        nullptr,
        &resolve_cpu,
    };
    return api_test::attention_image::run(argc, argv, backend);
}
