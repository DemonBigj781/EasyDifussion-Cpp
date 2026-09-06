#include "detect.hpp"

#include <utility>
#include <vector>

namespace edcpp::api::detect::opencl::translation {
namespace {

Result detect_all() {
    std::vector<Result> parts;
    parts.push_back(cpu::detect());
    parts.push_back(gpu::detect());
    parts.push_back(npu::detect());
    return combine(Backend::opencl, "opencl", std::move(parts));
}

[[maybe_unused]] const bool registered =
    register_translation(Backend::opencl, &detect_all);

} // namespace
} // namespace edcpp::api::detect::opencl::translation
