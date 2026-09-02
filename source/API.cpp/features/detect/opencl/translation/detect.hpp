#pragma once

#include "features/detect/common/detect.hpp"

namespace edcpp::api::detect::opencl::translation {
namespace cpu { Result detect(); }
namespace gpu { Result detect(); }
namespace npu { Result detect(); }
} // namespace edcpp::api::detect::opencl::translation
