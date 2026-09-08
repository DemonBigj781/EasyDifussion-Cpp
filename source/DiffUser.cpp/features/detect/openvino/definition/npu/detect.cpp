#include "features/detect/openvino/definition/enumerate.hpp"
namespace edcpp::api::detect::openvino::definition::npu {
NativeResult detect() { return detail::enumerate(NativeClass::npu); }
} // namespace edcpp::api::detect::openvino::definition::npu
