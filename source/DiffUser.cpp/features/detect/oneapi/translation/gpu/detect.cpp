#include "features/detect/oneapi/translation/translate.hpp"

namespace edcpp::api::detect::oneapi::translation::gpu {

Result detect() {
    return detail::translate(definition::gpu::detect(), DeviceClass::gpu);
}

} // namespace edcpp::api::detect::oneapi::translation::gpu
