#include "features/detect/opencl/definition/enumerate.hpp"

namespace edcpp::api::detect::opencl::definition::cpu {
NativeResult detect() { return detail::enumerate(NativeClass::cpu); }
} // namespace edcpp::api::detect::opencl::definition::cpu
