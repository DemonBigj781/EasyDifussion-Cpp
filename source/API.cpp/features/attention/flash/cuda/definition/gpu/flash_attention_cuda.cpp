#include "flash_attention_cuda.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace edcpp::api::attention::flash::cuda::definition::gpu {
namespace {

constexpr std::int64_t maximum_value_dimension = 512;

std::size_t element_size(NativeDType dtype) noexcept {
    switch (dtype) {
        case NativeDType::f32: return sizeof(float);
        case NativeDType::f16:
        case NativeDType::bf16: return sizeof(std::uint16_t);
    }
    return 0;
}

bool valid_dtype(NativeDType dtype) noexcept {
    return element_size(dtype) != 0;
}

template <typename Tensor>
bool valid_tensor(const Tensor& tensor) noexcept {
    if (tensor.data == nullptr || !valid_dtype(tensor.dtype) ||
        tensor.head_dim <= 0 || tensor.tokens <= 0 || tensor.heads <= 0 ||
        tensor.batch <= 0) {
        return false;
    }
    const std::size_t item_size = element_size(tensor.dtype);
    if (reinterpret_cast<std::uintptr_t>(tensor.data) % item_size != 0) {
        return false;
    }

    const std::int64_t dimensions[] = {
        tensor.head_dim, tensor.tokens, tensor.heads, tensor.batch,
    };
    std::size_t elements = 1;
    for (const std::int64_t dimension : dimensions) {
        const auto value = static_cast<std::size_t>(dimension);
        if (value > std::numeric_limits<std::size_t>::max() / elements) {
            return false;
        }
        elements *= value;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / item_size) {
        return false;
    }

    std::array<std::size_t, 4> strides = tensor.byte_strides;
    std::size_t contiguous_span = item_size;
    for (std::size_t axis = 0; axis < tensor.byte_strides.size(); ++axis) {
        if (strides[axis] == 0) {
            strides[axis] = contiguous_span;
        }
        if (strides[axis] % item_size != 0) {
            return false;
        }
        const auto dimension = static_cast<std::size_t>(dimensions[axis]);
        if (dimension >
            std::numeric_limits<std::size_t>::max() / contiguous_span) {
            return false;
        }
        contiguous_span *= dimension;
    }

    struct AxisLayout {
        std::size_t stride;
        std::size_t dimension;
    };
    std::array<AxisLayout, 4> layout{};
    for (std::size_t axis = 0; axis < layout.size(); ++axis) {
        layout[axis] = {
            strides[axis], static_cast<std::size_t>(dimensions[axis])};
    }
    std::sort(
        layout.begin(), layout.end(),
        [](const AxisLayout& left, const AxisLayout& right) {
            return left.stride < right.stride;
        });

    std::size_t addressed_span = item_size;
    for (const AxisLayout& axis : layout) {
        if (axis.dimension <= 1) {
            continue;
        }
        if (axis.stride < addressed_span) {
            return false;
        }
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

NativeResult invalid(const char* message) noexcept {
    return {false, message};
}

} // namespace

NativeCapabilities capabilities() noexcept {
    NativeCapabilities result;
    result.fused_forward = true;
    result.additive_mask = true;
    result.alibi_bias = true;
    result.logit_softcap = true;
    result.grouped_query = true;
    result.f32_accumulation = true;
    result.f32 = true;
    result.f16 = true;
    result.bf16 = true;
    result.maximum_value_dimension = maximum_value_dimension;
    result.minimum_compute_major = 6;
    return result;
}

NativeResult validate_forward(const NativeRequest& request) noexcept {
    if (!valid_tensor(request.query) || !valid_tensor(request.key) ||
        !valid_tensor(request.value) || !valid_tensor(request.output)) {
        return invalid("CUDA FlashAttention tensors must be non-null, positive, addressable, and non-overlapping");
    }
    if (request.output.dtype != NativeDType::f32) {
        return invalid("CUDA FlashAttention output must use F32 storage");
    }
    if (request.query.head_dim != request.key.head_dim ||
        request.key.tokens != request.value.tokens) {
        return invalid("CUDA FlashAttention requires matching Q/K dimensions and K/V token counts");
    }
    if (request.query.batch != request.key.batch ||
        request.query.batch != request.value.batch ||
        request.query.heads % request.key.heads != 0 ||
        request.query.heads % request.value.heads != 0) {
        return invalid("CUDA FlashAttention requires matching batches and grouped-query-compatible heads");
    }
    if (request.output.head_dim != request.value.head_dim ||
        request.output.tokens != request.query.tokens ||
        request.output.heads != request.query.heads ||
        request.output.batch != request.query.batch) {
        return invalid("CUDA FlashAttention output shape does not match the normalized contract");
    }
    if (request.value.head_dim > maximum_value_dimension) {
        return invalid("CUDA FlashAttention value dimension exceeds the fused kernel limit of 512");
    }
    if (request.mask.data != nullptr) {
        if (!valid_tensor(request.mask) ||
            request.mask.head_dim != request.key.tokens ||
            request.mask.tokens != request.query.tokens ||
            request.query.heads % request.mask.heads != 0 ||
            request.query.batch % request.mask.batch != 0) {
            return invalid("CUDA FlashAttention mask shape or layout is not broadcast-compatible");
        }
    }
    if (!std::isfinite(request.scale) || request.scale < 0.0f ||
        !std::isfinite(request.max_bias) || request.max_bias < 0.0f ||
        !std::isfinite(request.logit_softcap) ||
        request.logit_softcap < 0.0f) {
        return invalid("CUDA FlashAttention parameters must be finite and non-negative");
    }

    const auto batch = static_cast<std::uint64_t>(request.query.batch);
    const auto heads = static_cast<std::uint64_t>(request.query.heads);
    const auto tokens = static_cast<std::uint64_t>(request.query.tokens);
    const auto maximum = static_cast<std::uint64_t>(
        std::numeric_limits<std::int32_t>::max());
    if (batch > maximum / heads || batch * heads > maximum / tokens) {
        return invalid("CUDA FlashAttention row count exceeds the fused launch limit");
    }
    return {true, nullptr};
}

} // namespace edcpp::api::attention::flash::cuda::definition::gpu
