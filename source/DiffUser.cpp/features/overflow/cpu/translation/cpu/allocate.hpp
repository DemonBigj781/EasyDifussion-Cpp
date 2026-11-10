#pragma once

#include "features/overflow/common/overflow.hpp"

namespace edcpp::api::overflow::cpu::translation::cpu {

PlanningResult prepare(const Request& request);
Result allocate(const Request& request, const Candidate& candidate);

} // namespace edcpp::api::overflow::cpu::translation::cpu
