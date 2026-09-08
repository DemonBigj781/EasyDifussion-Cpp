#include "features/detect/vulkan/translation/translate.hpp"
namespace edcpp::api::detect::vulkan::translation::cpu {
Result detect() { return detail::translate(definition::cpu::detect(), DeviceClass::cpu); }
} // namespace edcpp::api::detect::vulkan::translation::cpu
