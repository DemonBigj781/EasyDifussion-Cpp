#include "features/detect/opencl/translation/translate.hpp"

namespace edcpp::api::detect::opencl::translation::cpu {
Result detect() { return detail::translate(definition::cpu::detect(), DeviceClass::cpu); }
} // namespace edcpp::api::detect::opencl::translation::cpu
