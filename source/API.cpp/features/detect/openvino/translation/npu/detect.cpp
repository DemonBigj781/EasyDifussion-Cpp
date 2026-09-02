#include "features/detect/openvino/translation/translate.hpp"
namespace edcpp::api::detect::openvino::translation::npu {
Result detect() { return detail::translate(definition::npu::detect(), DeviceClass::npu); }
} // namespace edcpp::api::detect::openvino::translation::npu
