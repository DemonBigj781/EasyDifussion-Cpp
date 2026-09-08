#include "features/detect/opencl/definition/enumerate.hpp"

namespace edcpp::api::detect::opencl::definition::gpu {
NativeResult detect() { return detail::enumerate(NativeClass::gpu); }
} // namespace edcpp::api::detect::opencl::definition::gpu
