#pragma once

namespace edcpp::api::attention::sage::cuda::translation::gpu {

// Static libraries may otherwise discard support.cpp before its registration
// initializer runs. Calling this anchor makes registration explicit and
// idempotent for application adapters.
bool ensure_registered() noexcept;

} // namespace edcpp::api::attention::sage::cuda::translation::gpu
