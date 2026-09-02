#include "features/detect/vulkan/translation/translate.hpp"
namespace edcpp::api::detect::vulkan::translation::gpu {
Result detect() { return detail::translate(definition::gpu::detect(), DeviceClass::gpu); }
} // namespace edcpp::api::detect::vulkan::translation::gpu
