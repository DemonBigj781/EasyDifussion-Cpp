#include "features/detect/opencl/translation/translate.hpp"

namespace edcpp::api::detect::opencl::translation::npu {
Result detect() { return detail::translate(definition::npu::detect(), DeviceClass::npu); }
} // namespace edcpp::api::detect::opencl::translation::npu
