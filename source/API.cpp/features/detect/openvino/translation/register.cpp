#include "detect.hpp"

#include <utility>
#include <vector>

namespace edcpp::api::detect::openvino::translation {
namespace {

Result detect_all() {
    std::vector<Result> parts;
    parts.push_back(cpu::detect());
    parts.push_back(gpu::detect());
    parts.push_back(npu::detect());
    return combine(Backend::openvino, "openvino", std::move(parts));
}

[[maybe_unused]] const bool registered =
    register_translation(Backend::openvino, &detect_all);

} // namespace
} // namespace edcpp::api::detect::openvino::translation
