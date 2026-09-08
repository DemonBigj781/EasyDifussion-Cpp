#include "features/detect/oneapi/translation/translate.hpp"

namespace edcpp::api::detect::oneapi::translation::cpu {

Result detect() {
    return detail::translate(definition::cpu::detect(), DeviceClass::cpu);
}

} // namespace edcpp::api::detect::oneapi::translation::cpu
