#include "features/detect/vulkan/definition/enumerate.hpp"
namespace edcpp::api::detect::vulkan::definition::gpu {
NativeResult detect() { return detail::enumerate(NativeClass::gpu); }
} // namespace edcpp::api::detect::vulkan::definition::gpu
