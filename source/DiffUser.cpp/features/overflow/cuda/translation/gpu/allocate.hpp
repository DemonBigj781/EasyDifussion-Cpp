#pragma once

#include "features/overflow/common/overflow.hpp"

namespace edcpp::api::overflow::cuda::translation::gpu {

PlanningResult prepare(const Request& request);
Result allocate(const Request& request, const Candidate& candidate);

} // namespace edcpp::api::overflow::cuda::translation::gpu
