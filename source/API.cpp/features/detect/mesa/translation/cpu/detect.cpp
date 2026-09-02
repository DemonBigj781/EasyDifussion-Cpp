#include "features/detect/mesa/translation/translate.hpp"
namespace edcpp::api::detect::mesa::translation::cpu {
Result detect() { return detail::translate(definition::cpu::detect(), DeviceClass::cpu); }
} // namespace edcpp::api::detect::mesa::translation::cpu
