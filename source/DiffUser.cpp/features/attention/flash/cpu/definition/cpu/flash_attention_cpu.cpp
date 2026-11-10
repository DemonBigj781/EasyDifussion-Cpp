#include "flash_attention_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace edcpp::api::attention::flash::cpu::definition::cpu {
namespace {

std::size_t scalar_size(NativeDType type) noexcept {
    return type == NativeDType::f32 ? sizeof(float) : sizeof(std::uint16_t);
}

float f16_to_float(std::uint16_t value) noexcept {
    const std::uint32_t sign =
        static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int adjusted = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --adjusted;
            }
            bits = sign |
                (static_cast<std::uint32_t>(adjusted + 127) << 23) |
                ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) |
            (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float load(
    const void* base, NativeDType type,
    std::size_t byte_offset) noexcept {
    const auto* bytes =
        static_cast<const std::uint8_t*>(base) + byte_offset;
    if (type == NativeDType::f32) {
        float value = 0.0f;
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    }
    std::uint16_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    if (type == NativeDType::f16) return f16_to_float(value);
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename Tensor>
std::size_t offset(
    const Tensor& tensor, std::int64_t dimension, std::int64_t token,
    std::int64_t head, std::int64_t batch) noexcept {
    const std::size_t item = scalar_size(tensor.dtype);
    const std::size_t stride_dimension =
        tensor.byte_strides[0] ? tensor.byte_strides[0] : item;
    const std::size_t stride_token = tensor.byte_strides[1]
        ? tensor.byte_strides[1]
        : item * static_cast<std::size_t>(tensor.head_dim);
    const std::size_t stride_head = tensor.byte_strides[2]
        ? tensor.byte_strides[2]
        : stride_token * static_cast<std::size_t>(tensor.tokens);
    const std::size_t stride_batch = tensor.byte_strides[3]
        ? tensor.byte_strides[3]
        : stride_head * static_cast<std::size_t>(tensor.heads);
    return static_cast<std::size_t>(dimension) * stride_dimension +
        static_cast<std::size_t>(token) * stride_token +
        static_cast<std::size_t>(head) * stride_head +
        static_cast<std::size_t>(batch) * stride_batch;
}

NativeResult invalid(const char* message) noexcept {
    return {false, message};
}

float alibi_slope(
    std::int64_t head, std::int64_t heads, float max_bias) noexcept {
    if (max_bias == 0.0f) return 1.0f;
    std::uint32_t power = 1;
    while ((power << 1) <= static_cast<std::uint32_t>(heads)) {
        power <<= 1;
    }
    const float first =
        std::pow(2.0f, -max_bias / static_cast<float>(power));
    const float second =
        std::pow(2.0f, -(max_bias * 0.5f) / static_cast<float>(power));
    return head < power
        ? std::pow(first, static_cast<float>(head + 1))
        : std::pow(
              second, static_cast<float>(2 * (head - power) + 1));
}

} // namespace

NativeCapabilities capabilities() noexcept {
    return {true, true, true, true, true, true};
}

NativeResult validate(const NativeRequest& request) noexcept {
    if (request.query.data == nullptr || request.key.data == nullptr ||
        request.value.data == nullptr || request.output.data == nullptr) {
        return invalid(
            "CPU FlashAttention requires Q, K, V, and output buffers");
    }
    if (request.execution.thread_index != 0 ||
        request.execution.thread_count != 1) {
        return invalid(
            "CPU FlashAttention currently owns a single-thread execution context");
    }
    if (request.output.dtype != NativeDType::f32) {
        return invalid("CPU FlashAttention currently writes F32 output");
    }
    if (request.query.batch <= 0 || request.query.heads <= 0 ||
        request.query.tokens <= 0 || request.query.head_dim <= 0 ||
        request.key.batch != request.query.batch ||
        request.value.batch != request.query.batch ||
        request.key.tokens <= 0 ||
        request.key.tokens != request.value.tokens ||
        request.key.head_dim != request.query.head_dim ||
        request.value.head_dim <= 0 || request.key.heads <= 0 ||
        request.value.heads <= 0 ||
        request.query.heads % request.key.heads != 0 ||
        request.query.heads % request.value.heads != 0) {
        return invalid(
            "CPU FlashAttention received incompatible Q/K/V shapes");
    }
    if (request.output.batch != request.query.batch ||
        request.output.heads != request.query.heads ||
        request.output.tokens != request.query.tokens ||
        request.output.head_dim != request.value.head_dim) {
        return invalid("CPU FlashAttention output shape is incompatible");
    }
    if (!std::isfinite(request.scale) ||
        !std::isfinite(request.max_bias) ||
        !std::isfinite(request.logit_softcap) ||
        request.max_bias < 0.0f || request.logit_softcap < 0.0f) {
        return invalid(
            "CPU FlashAttention scalar parameters must be finite and non-negative");
    }
    if (request.mask.data != nullptr &&
        (request.mask.head_dim != request.key.tokens ||
         request.mask.tokens != request.query.tokens ||
         (request.mask.heads != 1 &&
          request.mask.heads != request.query.heads) ||
         (request.mask.batch != 1 &&
          request.mask.batch != request.query.batch))) {
        return invalid(
            "CPU FlashAttention mask is not broadcast-compatible");
    }
    return {true, nullptr};
}

NativeResult forward(const NativeRequest& request) noexcept {
    const NativeResult checked = validate(request);
    if (!checked.ok) return checked;

    std::vector<float> scores(
        static_cast<std::size_t>(request.key.tokens));
    for (std::int64_t batch = 0;
         batch < request.query.batch; ++batch) {
        for (std::int64_t head = 0;
             head < request.query.heads; ++head) {
            const auto key_head =
                head / (request.query.heads / request.key.heads);
            const auto value_head =
                head / (request.query.heads / request.value.heads);
            const float slope = alibi_slope(
                head, request.query.heads, request.max_bias);
            for (std::int64_t query = 0;
                 query < request.query.tokens; ++query) {
                float maximum =
                    -std::numeric_limits<float>::infinity();
                for (std::int64_t key = 0;
                     key < request.key.tokens; ++key) {
                    float score = 0.0f;
                    for (std::int64_t dimension = 0;
                         dimension < request.query.head_dim;
                         ++dimension) {
                        score += load(
                                     request.query.data,
                                     request.query.dtype,
                                     offset(
                                         request.query, dimension, query,
                                         head, batch)) *
                                 load(
                                     request.key.data,
                                     request.key.dtype,
                                     offset(
                                         request.key, dimension, key,
                                         key_head, batch));
                    }
                    score *= request.scale;
                    if (request.mask.data != nullptr) {
                        const auto mask_head =
                            request.mask.heads == 1 ? 0 : head;
                        const auto mask_batch =
                            request.mask.batch == 1 ? 0 : batch;
                        score += load(
                                     request.mask.data,
                                     request.mask.dtype,
                                     offset(
                                         request.mask, key, query,
                                         mask_head, mask_batch)) *
                                 slope;
                    }
                    if (request.logit_softcap > 0.0f) {
                        score = request.logit_softcap *
                            std::tanh(score / request.logit_softcap);
                    }
                    scores[static_cast<std::size_t>(key)] = score;
                    maximum = std::max(maximum, score);
                }

                float sum = 0.0f;
                for (float& score : scores) {
                    score = std::exp(score - maximum);
                    sum += score;
                }
                for (std::int64_t dimension = 0;
                     dimension < request.value.head_dim;
                     ++dimension) {
                    float result = 0.0f;
                    for (std::int64_t key = 0;
                         key < request.key.tokens; ++key) {
                        result +=
                            scores[static_cast<std::size_t>(key)] / sum *
                            load(
                                request.value.data,
                                request.value.dtype,
                                offset(
                                    request.value, dimension, key,
                                    value_head, batch));
                    }
                    const auto output_offset = offset(
                        request.output, dimension, query, head, batch);
                    std::memcpy(
                        static_cast<std::uint8_t*>(request.output.data) +
                            output_offset,
                        &result, sizeof(result));
                }
            }
        }
    }
    return {true, nullptr};
}

} // namespace edcpp::api::attention::flash::cpu::definition::cpu
