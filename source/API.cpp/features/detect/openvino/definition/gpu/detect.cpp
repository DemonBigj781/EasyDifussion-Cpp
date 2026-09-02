#include "features/detect/openvino/definition/enumerate.hpp"
namespace edcpp::api::detect::openvino::definition::gpu {
NativeResult detect() { return detail::enumerate(NativeClass::gpu); }
} // namespace edcpp::api::detect::openvino::definition::gpu
