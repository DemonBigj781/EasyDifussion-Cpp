#include "xformers.hpp"

#include <cstddef>
#include <limits>

namespace edcpp::api::attention::xformers::cuda::definition::gpu {
namespace {

constexpr std::int64_t maximum_value_dimension = 512;

std::size_t element_size(NativeDType dtype) noexcept {
    return dtype == NativeDType::f32 ? sizeof(float) : sizeof(std::uint16_t);
}

template <typename Tensor>
bool valid_tensor_extent(const Tensor& tensor) noexcept {
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

NativeValidationResult validate_av(const NativeRequest& request) noexcept {
    if (request.v.data == nullptr || request.out.data == nullptr) {
        return {false, "CUDA xFormers AV requires non-null V and output device buffers"};
    }
    if (request.v.batch <= 0 || request.v.heads <= 0 || request.v.tokens <= 0 ||
        request.v.head_dim <= 0 || !valid_tensor_extent(request.v) ||
        !valid_tensor_extent(request.out)) {
        return {false, "CUDA xFormers V and output dimensions must be positive and addressable"};
    }
    if (request.v.head_dim > maximum_value_dimension) {
        return {false, "CUDA xFormers value head dimension exceeds the fused kernel limit of 512"};
    }
    if (request.out.dtype != NativeDType::f32) {
        return {false, "CUDA xFormers output must use F32 storage"};
    }
    if (request.q.batch % request.v.batch != 0) {
        return {false, "CUDA xFormers requires the V batch count to divide the Q batch count"};
    }
    if (request.k.tokens != request.v.tokens) {
        return {false, "CUDA xFormers requires matching K and V token counts"};
    }
    if (request.q.heads % request.v.heads != 0) {
        return {false, "CUDA xFormers requires the V head count to divide the Q head count"};
    }
    if (request.out.batch != request.q.batch || request.out.heads != request.q.heads ||
        request.out.tokens != request.q.tokens || request.out.head_dim != request.v.head_dim) {
        return {false, "CUDA xFormers output shape must be [Q batch, Q heads, Q tokens, V head_dim]"};
    }
    return {true, nullptr};
}

} // namespace edcpp::api::attention::xformers::cuda::definition::gpu
