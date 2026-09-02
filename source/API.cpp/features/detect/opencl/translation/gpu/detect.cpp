#include "features/detect/opencl/translation/translate.hpp"

namespace edcpp::api::detect::opencl::translation::gpu {
Result detect() { return detail::translate(definition::gpu::detect(), DeviceClass::gpu); }
} // namespace edcpp::api::detect::opencl::translation::gpu
