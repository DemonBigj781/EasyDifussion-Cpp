#pragma once

#include "features/overflow/common/overflow.hpp"

namespace edcpp::api::overflow::cuda::translation::gpu {

ReleaseResult release(const Resource& resource) noexcept;

} // namespace edcpp::api::overflow::cuda::translation::gpu
