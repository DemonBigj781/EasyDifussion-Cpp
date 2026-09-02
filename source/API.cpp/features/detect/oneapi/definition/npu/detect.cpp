#include "features/detect/oneapi/definition/enumerate.hpp"

namespace edcpp::api::detect::oneapi::definition::npu {

NativeResult detect() {
    return detail::enumerate(NativeClass::npu);
}

} // namespace edcpp::api::detect::oneapi::definition::npu
