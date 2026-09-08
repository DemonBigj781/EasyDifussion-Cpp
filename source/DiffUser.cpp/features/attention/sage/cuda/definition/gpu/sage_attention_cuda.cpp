#include "sage_attention_cuda.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace edcpp::api::attention::sage::cuda::definition::gpu {
namespace {

std::size_t element_size(NativeDType dtype) noexcept {
    switch (dtype) {
        case NativeDType::f32: return sizeof(float);
        case NativeDType::f16:
        case NativeDType::bf16: return sizeof(std::uint16_t);
    }
    return 0;
}

template <typename Tensor>
bool valid_layout(const Tensor& tensor) noexcept {
    const std::int64_t dimensions[] = {
        tensor.head_dim, tensor.tokens, tensor.heads, tensor.batch};
    for (const auto dimension : dimensions) {
        if (dimension <= 0 ||
            static_cast<std::uint64_t>(dimension) >
                std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
    }

    const std::size_t item_size = element_size(tensor.dtype);
    if (item_size == 0) return false;
    std::array<std::size_t, 4> strides = tensor.byte_strides;
    std::size_t contiguous = item_size;
    for (std::size_t axis = 0; axis < strides.size(); ++axis) {
        if (strides[axis] == 0) strides[axis] = contiguous;
        if (strides[axis] % item_size != 0) {
            return false;
        }
        const auto dimension = static_cast<std::size_t>(dimensions[axis]);
        if (dimension > std::numeric_limits<std::size_t>::max() / contiguous) {
            return false;
        }
        contiguous *= dimension;
    }

    struct Axis {
        std::size_t stride;
        std::size_t dimension;
    };
    std::array<Axis, 4> layout{};
    for (std::size_t axis = 0; axis < layout.size(); ++axis) {
        layout[axis] = {
            strides[axis], static_cast<std::size_t>(dimensions[axis])};
    }
    std::sort(layout.begin(), layout.end(), [](const Axis& left, const Axis& right) {
        return left.stride < right.stride;
    });
    std::size_t addressed_span = item_size;
    for (const Axis& axis : layout) {
        if (axis.dimension <= 1) continue;
        if (axis.stride < addressed_span) return false;
        const std::size_t repetitions = axis.dimension - 1;
        if (repetitions >
            (std::numeric_limits<std::size_t>::max() - addressed_span) /
                axis.stride) {
            return false;
        }
        addressed_span += repetitions * axis.stride;
    }
    return true;
}

bool valid_output_layout(const NativeMutableTensor4D& output) noexcept {
    if (!valid_layout(output)) return false;
    const std::size_t item_size = element_size(output.dtype);
    const std::array<std::size_t, 4> expected{
        item_size,
        item_size * static_cast<std::size_t>(output.head_dim) *
            static_cast<std::size_t>(output.heads),
        item_size * static_cast<std::size_t>(output.head_dim),
        item_size * static_cast<std::size_t>(output.head_dim) *
            static_cast<std::size_t>(output.heads) *
            static_cast<std::size_t>(output.tokens),
    };
    return output.byte_strides == expected;
}

NativeResult invalid(const char* message) noexcept {
    return {false, message};
}

} // namespace

NativeCapabilities capabilities() noexcept {
    return {
        true,
        false,
        true,
        true,
        true,
        8,
        9,
    };
}

NativeResult support(const NativeSupportRequest& request) noexcept {
    if (request.device_index < 0 || request.compute_major != 8 ||
        request.compute_minor < 0) {
        return invalid("CUDA SageAttention currently requires an Ampere/Ada SM8x device");
    }
    if (request.query.dtype != NativeDType::f32 ||
        request.key.dtype != NativeDType::f16 ||
        request.value.dtype != NativeDType::f16 ||
        request.output.dtype != NativeDType::f32) {
        return invalid("CUDA SageAttention requires F32 query/output and F16 key/value tensors");
    }
    if ((request.query.head_dim != 64 && request.query.head_dim != 128) ||
        request.key.head_dim != request.query.head_dim ||
        request.value.head_dim != request.query.head_dim) {
        return invalid("CUDA SageAttention supports matching Q/K/V head dimensions of 64 or 128");
    }
    if (request.key.tokens != request.value.tokens ||
        request.key.heads != request.value.heads ||
        request.query.heads % request.key.heads != 0 ||
        request.query.batch != request.key.batch ||
        request.query.batch != request.value.batch) {
        return invalid("CUDA SageAttention requires matching K/V shapes and grouped-query-compatible heads");
    }
    if (request.output.head_dim != request.value.head_dim ||
        request.output.tokens != request.query.tokens ||
        request.output.heads != request.query.heads ||
        request.output.batch != request.query.batch) {
        return invalid("CUDA SageAttention output shape does not match the normalized contract");
    }
    if (!valid_layout(request.query) || !valid_layout(request.key) ||
        !valid_layout(request.value) || !valid_output_layout(request.output)) {
        return invalid("CUDA SageAttention tensor strides are not compatible with the SM80 kernel");
    }
    if (request.additive_mask || request.attention_sinks ||
        request.max_bias != 0.0f || request.logit_softcap != 0.0f) {
        return invalid("CUDA SageAttention does not yet support masks, sinks, bias, or logit soft-capping");
    }
    if (!std::isfinite(request.scale) || request.scale < 0.0f) {
        return invalid("CUDA SageAttention scale must be finite and non-negative");
    }
    return {true, nullptr};
}

} // namespace edcpp::api::attention::sage::cuda::definition::gpu
