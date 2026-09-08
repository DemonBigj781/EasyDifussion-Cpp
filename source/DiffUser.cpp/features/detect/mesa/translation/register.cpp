#include "detect.hpp"

#include <utility>
#include <vector>

namespace edcpp::api::detect::mesa::translation {
namespace {

Result detect_all() {
    std::vector<Result> parts;
    parts.push_back(cpu::detect());
    parts.push_back(gpu::detect());
    return combine(Backend::mesa, "mesa", std::move(parts));
}

[[maybe_unused]] const bool registered =
    register_translation(Backend::mesa, &detect_all);

} // namespace
} // namespace edcpp::api::detect::mesa::translation
