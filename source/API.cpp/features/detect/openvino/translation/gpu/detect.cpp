#include "features/detect/openvino/translation/translate.hpp"
namespace edcpp::api::detect::openvino::translation::gpu {
Result detect() { return detail::translate(definition::gpu::detect(), DeviceClass::gpu); }
} // namespace edcpp::api::detect::openvino::translation::gpu
