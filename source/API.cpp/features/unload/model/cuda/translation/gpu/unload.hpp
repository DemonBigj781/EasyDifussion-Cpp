#pragma once

#include "features/load/common/load.hpp"
#include "features/unload/common/unload.hpp"

namespace edcpp::api::unload::model::cuda::translation::gpu {

Result unload(const load::Resource& resource);

} // namespace edcpp::api::unload::model::cuda::translation::gpu
