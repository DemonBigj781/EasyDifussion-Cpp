#include "features/detect/openvino/definition/enumerate.hpp"
namespace edcpp::api::detect::openvino::definition::cpu {
NativeResult detect() { return detail::enumerate(NativeClass::cpu); }
} // namespace edcpp::api::detect::openvino::definition::cpu
