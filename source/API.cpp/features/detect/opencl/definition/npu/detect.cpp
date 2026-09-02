#include "features/detect/opencl/definition/enumerate.hpp"

namespace edcpp::api::detect::opencl::definition::npu {
NativeResult detect() { return detail::enumerate(NativeClass::npu); }
} // namespace edcpp::api::detect::opencl::definition::npu
