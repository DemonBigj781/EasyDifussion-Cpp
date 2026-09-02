#include "features/detect/oneapi/translation/translate.hpp"

namespace edcpp::api::detect::oneapi::translation::npu {

Result detect() {
    return detail::translate(definition::npu::detect(), DeviceClass::npu);
}

} // namespace edcpp::api::detect::oneapi::translation::npu
