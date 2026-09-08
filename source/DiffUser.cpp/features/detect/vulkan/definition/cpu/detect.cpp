#include "features/detect/vulkan/definition/enumerate.hpp"
namespace edcpp::api::detect::vulkan::definition::cpu {
NativeResult detect() { return detail::enumerate(NativeClass::cpu); }
} // namespace edcpp::api::detect::vulkan::definition::cpu
