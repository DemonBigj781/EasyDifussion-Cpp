#include "xformers-image-gen.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// Linked from the canonical DiffUser.cpp CUDA xFormers implementation through the
// static ggml-cuda library. This is deliberately only an observation hook;
// image execution still enters attention through normal ggml dispatch.
std::uint64_t ggml_cuda_xformers_attn_launch_count() noexcept;

namespace {

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string uuid_text(const cudaUUID_t& uuid) {
    static constexpr std::array<int, 5> group_lengths{4, 2, 2, 2, 6};
    std::ostringstream text;
    text << "GPU-" << std::hex << std::setfill('0');
    int offset = 0;
    for (std::size_t group = 0; group < group_lengths.size(); ++group) {
        if (group != 0) text << '-';
        for (int index = 0; index < group_lengths[group]; ++index) {
            text << std::setw(2)
                 << static_cast<unsigned int>(
                        static_cast<unsigned char>(uuid.bytes[offset++]));
        }
    }
    return text.str();
}

bool parse_device_index(std::string text, int& index) {
    const std::string lowered = lower_copy(text);
    if (lowered.rfind("cuda", 0) == 0) {
        text.erase(0, 4);
    }
    if (text.empty()) return false;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, index);
    return result.ec == std::errc{} && result.ptr == end;
}

bool resolve_cuda(
        const std::string& requested,
        std::string& backend,
        std::string& description,
        std::string& diagnostic) {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count <= 0) {
        diagnostic = std::string("no CUDA device: ") +
            cudaGetErrorString(count_status);
        return false;
    }

    int selected = -1;
    int numeric = -1;
    if (parse_device_index(requested, numeric)) {
        if (numeric < 0 || numeric >= device_count) {
            diagnostic = "CUDA device index is out of range";
            return false;
        }
        selected = numeric;
    } else {
        const std::string needle = lower_copy(requested);
        std::vector<int> matches;
        for (int device = 0; device < device_count; ++device) {
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
                cudaGetLastError();
                continue;
            }
            const std::string name = lower_copy(properties.name);
            const std::string uuid = lower_copy(uuid_text(properties.uuid));
            if (name == needle || uuid == needle ||
                name.find(needle) != std::string::npos) {
                matches.push_back(device);
            }
        }
        if (matches.empty()) {
            diagnostic = "no CUDA device matched id, UUID, or name '" +
                requested + "'";
            return false;
        }
        if (matches.size() != 1) {
            diagnostic = "CUDA device name is ambiguous; use an id or UUID";
            return false;
        }
        selected = matches.front();
    }

    cudaDeviceProp properties{};
    const cudaError_t properties_status =
        cudaGetDeviceProperties(&properties, selected);
    if (properties_status != cudaSuccess) {
        diagnostic = cudaGetErrorString(properties_status);
        return false;
    }
    const cudaError_t select_status = cudaSetDevice(selected);
    if (select_status != cudaSuccess) {
        diagnostic = cudaGetErrorString(select_status);
        return false;
    }

    backend = "cuda" + std::to_string(selected);
    description = backend + " / " + properties.name + " / " +
        uuid_text(properties.uuid);
    return true;
}

std::uint64_t xformers_launch_count() {
    return ggml_cuda_xformers_attn_launch_count();
}

bool enable_xformers() {
#ifdef _WIN32
    return _putenv_s("SD_CUDA_FLASH_COMMON", "0") == 0 &&
           _putenv_s("SD_CUDA_XFORMERS", "1") == 0;
#else
    return setenv("SD_CUDA_FLASH_COMMON", "0", 1) == 0 &&
           setenv("SD_CUDA_XFORMERS", "1", 1) == 0;
#endif
}

} // namespace

int main(int argc, char** argv) {
    const api_test::attention_image::BackendConfiguration backend{
        "CUDA",
        "0",
        "xformers-cuda-output.ppm",
        "normalized xFormers Common",
        &enable_xformers,
        &xformers_launch_count,
        &resolve_cuda,
    };
    return api_test::attention_image::run(argc, argv, backend);
}
