#include "xformers.hpp"

#include <cstdint>
#include <limits>

namespace edcpp::api::attention::xformers::cuda::definition::gpu {

NativeValidationResult validate_softmax(const NativeRequest& request) noexcept {
    const auto batch = static_cast<std::uint64_t>(request.q.batch);
    const auto heads = static_cast<std::uint64_t>(request.q.heads);
    const auto tokens = static_cast<std::uint64_t>(request.q.tokens);
    const auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    if (batch == 0 || heads == 0 || tokens == 0 ||
        batch > maximum / heads || batch * heads > maximum / tokens) {
        return {false, "CUDA xFormers attention row count exceeds the fused launch limit"};
    }
    return {true, nullptr};
}

} // namespace edcpp::api::attention::xformers::cuda::definition::gpu
