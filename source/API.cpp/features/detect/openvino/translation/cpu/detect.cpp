#include "features/detect/openvino/translation/translate.hpp"
namespace edcpp::api::detect::openvino::translation::cpu {
Result detect() { return detail::translate(definition::cpu::detect(), DeviceClass::cpu); }
} // namespace edcpp::api::detect::openvino::translation::cpu
