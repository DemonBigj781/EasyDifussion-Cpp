#include "features/detect/mesa/definition/enumerate.hpp"
namespace edcpp::api::detect::mesa::definition::gpu {
NativeResult detect() { return detail::enumerate(NativeClass::gpu); }
} // namespace edcpp::api::detect::mesa::definition::gpu
