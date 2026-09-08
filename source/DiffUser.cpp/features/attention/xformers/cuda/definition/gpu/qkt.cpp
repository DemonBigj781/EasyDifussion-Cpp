#include "xformers.hpp"

#include <cstddef>
#include <limits>

namespace edcpp::api::attention::xformers::cuda::definition::gpu {
namespace {

NativeValidationResult invalid(const char* message) noexcept {
    return {false, message};
}

std::size_t element_size(NativeDType dtype) noexcept {
    return dtype == NativeDType::f32 ? sizeof(float) : sizeof(std::uint16_t);
}

bool valid_tensor_extent(const NativeTensor4D& tensor) noexcept {
    if (tensor.batch <= 0 || tensor.heads <= 0 || tensor.tokens <= 0 ||
        tensor.head_dim <= 0) {
        return false;
    }

    std::size_t elements = 1;
    const std::int64_t dimensions[] = {
        tensor.batch, tensor.heads, tensor.tokens, tensor.head_dim,
    };
    for (const std::int64_t dimension : dimensions) {
        const auto value = static_cast<std::size_t>(dimension);
        if (value > std::numeric_limits<std::size_t>::max() / elements) {
            return false;
        }
        elements *= value;
    }
    return elements <=
               std::numeric_limits<std::size_t>::max() / element_size(tensor.dtype) &&
           (tensor.byte_strides[0] == 0 ||
            tensor.byte_strides[0] == element_size(tensor.dtype));
}

} // namespace

NativeValidationResult validate_qkt(const NativeRequest& request) noexcept {
    if (request.q.data == nullptr || request.k.data == nullptr) {
        return invalid("CUDA xFormers QKT requires non-null Q and K device buffers");
    }
    if (!valid_tensor_extent(request.q) || !valid_tensor_extent(request.k)) {
        return invalid("CUDA xFormers Q and K dimensions must be positive and addressable");
    }
    if (request.q.batch % request.k.batch != 0) {
        return invalid("CUDA xFormers requires the K batch count to divide the Q batch count");
    }
    if (request.q.head_dim != request.k.head_dim) {
        return invalid("CUDA xFormers requires matching Q and K head dimensions");
    }
    if (request.q.heads % request.k.heads != 0) {
        return invalid("CUDA xFormers requires the K head count to divide the Q head count");
    }
    return {true, nullptr};
}

} // namespace edcpp::api::attention::xformers::cuda::definition::gpu
