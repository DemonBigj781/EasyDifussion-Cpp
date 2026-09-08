#include "features/detect/oneapi/definition/enumerate.hpp"

namespace edcpp::api::detect::oneapi::definition::gpu {

NativeResult detect() {
    return detail::enumerate(NativeClass::gpu);
}

} // namespace edcpp::api::detect::oneapi::definition::gpu
