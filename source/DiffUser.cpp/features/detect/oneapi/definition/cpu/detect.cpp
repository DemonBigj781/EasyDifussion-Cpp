#include "features/detect/oneapi/definition/enumerate.hpp"

namespace edcpp::api::detect::oneapi::definition::cpu {

NativeResult detect() {
    return detail::enumerate(NativeClass::cpu);
}

} // namespace edcpp::api::detect::oneapi::definition::cpu
