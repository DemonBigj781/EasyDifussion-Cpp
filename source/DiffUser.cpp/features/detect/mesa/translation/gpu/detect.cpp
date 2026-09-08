#include "features/detect/mesa/translation/translate.hpp"
namespace edcpp::api::detect::mesa::translation::gpu {
Result detect() { return detail::translate(definition::gpu::detect(), DeviceClass::gpu); }
} // namespace edcpp::api::detect::mesa::translation::gpu
