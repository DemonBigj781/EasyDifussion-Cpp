#include "features/detect/mesa/definition/enumerate.hpp"
namespace edcpp::api::detect::mesa::definition::cpu {
NativeResult detect() { return detail::enumerate(NativeClass::cpu); }
} // namespace edcpp::api::detect::mesa::definition::cpu
