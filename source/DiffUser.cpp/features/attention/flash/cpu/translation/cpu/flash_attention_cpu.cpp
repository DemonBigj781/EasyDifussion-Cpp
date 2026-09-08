#include "features/attention/flash/common/flash_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace edcpp::api::attention::flash::cpu::translation {
namespace {

std::size_t scalar_size(DType type) noexcept {
    return type == DType::f32 ? sizeof(float) : sizeof(std::uint16_t);
}

float f16_to_float(std::uint16_t value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int adjusted = -14;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; --adjusted; }
            bits = sign | (static_cast<std::uint32_t>(adjusted + 127) << 23) |
                   ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) bits = sign | 0x7f800000u | (mantissa << 13);
    else bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float load(const void* base, DType type, std::size_t byte_offset) noexcept {
    const auto* bytes = static_cast<const std::uint8_t*>(base) + byte_offset;
    if (type == DType::f32) {
        float value;
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    }
    std::uint16_t value;
    std::memcpy(&value, bytes, sizeof(value));
    if (type == DType::f16) return f16_to_float(value);
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename Tensor>
std::size_t offset(const Tensor& t, std::int64_t d, std::int64_t token,
                   std::int64_t head, std::int64_t batch) noexcept {
    const std::size_t item = scalar_size(t.dtype);
    const std::size_t s0 = t.byte_strides[0] ? t.byte_strides[0] : item;
    const std::size_t s1 = t.byte_strides[1] ? t.byte_strides[1] : item * t.head_dim;
    const std::size_t s2 = t.byte_strides[2] ? t.byte_strides[2] : s1 * t.tokens;
    const std::size_t s3 = t.byte_strides[3] ? t.byte_strides[3] : s2 * t.heads;
    return d * s0 + token * s1 + head * s2 + batch * s3;
}

Result invalid(const char* message) noexcept { return {false, message}; }

Result validate_cpu(const Request& r) noexcept {
    if (!r.query.data || !r.key.data || !r.value.data || !r.output.data)
        return invalid("CPU FlashAttention requires Q, K, V, and output buffers");
    if (r.execution.thread_index != 0 || r.execution.thread_count != 1)
        return invalid("CPU FlashAttention currently owns a single-thread execution context");
    if (r.output.dtype != DType::f32)
        return invalid("CPU FlashAttention currently writes F32 output");
    if (r.query.batch <= 0 || r.query.heads <= 0 || r.query.tokens <= 0 || r.query.head_dim <= 0 ||
        r.key.batch != r.query.batch || r.value.batch != r.query.batch || r.key.tokens <= 0 ||
        r.key.tokens != r.value.tokens || r.key.head_dim != r.query.head_dim || r.value.head_dim <= 0 ||
        r.key.heads <= 0 || r.value.heads <= 0 ||
        r.query.heads % r.key.heads != 0 || r.query.heads % r.value.heads != 0)
        return invalid("CPU FlashAttention received incompatible Q/K/V shapes");
    if (r.output.batch != r.query.batch || r.output.heads != r.query.heads ||
        r.output.tokens != r.query.tokens || r.output.head_dim != r.value.head_dim)
        return invalid("CPU FlashAttention output shape is incompatible");
    if (!std::isfinite(r.scale) || !std::isfinite(r.max_bias) ||
        !std::isfinite(r.logit_softcap) || r.max_bias < 0.0f || r.logit_softcap < 0.0f)
        return invalid("CPU FlashAttention scalar parameters must be finite and non-negative");
    if (r.mask.data && (r.mask.head_dim != r.key.tokens || r.mask.tokens != r.query.tokens ||
        (r.mask.heads != 1 && r.mask.heads != r.query.heads) ||
        (r.mask.batch != 1 && r.mask.batch != r.query.batch)))
        return invalid("CPU FlashAttention mask is not broadcast-compatible");
    return {true, nullptr};
}

float alibi_slope(std::int64_t head, std::int64_t heads, float max_bias) noexcept {
    if (max_bias == 0.0f) return 1.0f;
    std::uint32_t power = 1;
    while ((power << 1) <= static_cast<std::uint32_t>(heads)) power <<= 1;
    const float m0 = std::pow(2.0f, -max_bias / static_cast<float>(power));
    const float m1 = std::pow(2.0f, -(max_bias * 0.5f) / static_cast<float>(power));
    return head < power ? std::pow(m0, static_cast<float>(head + 1))
                        : std::pow(m1, static_cast<float>(2 * (head - power) + 1));
}

Result forward_cpu(const Request& r) noexcept {
    const auto checked = validate_cpu(r);
    if (!checked.ok) return checked;
    std::vector<float> scores(static_cast<std::size_t>(r.key.tokens));
    for (std::int64_t b = 0; b < r.query.batch; ++b) for (std::int64_t h = 0; h < r.query.heads; ++h) {
        const auto kh = h / (r.query.heads / r.key.heads);
        const auto vh = h / (r.query.heads / r.value.heads);
        const float slope = alibi_slope(h, r.query.heads, r.max_bias);
        for (std::int64_t q = 0; q < r.query.tokens; ++q) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t k = 0; k < r.key.tokens; ++k) {
                float score = 0.0f;
                for (std::int64_t d = 0; d < r.query.head_dim; ++d)
                    score += load(r.query.data, r.query.dtype, offset(r.query, d, q, h, b)) *
                             load(r.key.data, r.key.dtype, offset(r.key, d, k, kh, b));
                score *= r.scale;
                if (r.mask.data) {
                    const auto mh = r.mask.heads == 1 ? 0 : h;
                    const auto mb = r.mask.batch == 1 ? 0 : b;
                    score += load(r.mask.data, r.mask.dtype, offset(r.mask, k, q, mh, mb)) * slope;
                }
                if (r.logit_softcap > 0.0f) score = r.logit_softcap * std::tanh(score / r.logit_softcap);
                scores[static_cast<std::size_t>(k)] = score;
                maximum = std::max(maximum, score);
            }
            float sum = 0.0f;
            for (float& score : scores) { score = std::exp(score - maximum); sum += score; }
            for (std::int64_t d = 0; d < r.value.head_dim; ++d) {
                float result = 0.0f;
                for (std::int64_t k = 0; k < r.key.tokens; ++k)
                    result += scores[static_cast<std::size_t>(k)] / sum *
                              load(r.value.data, r.value.dtype, offset(r.value, d, k, vh, b));
                const float value = result;
                const auto out_offset = offset(r.output, d, q, h, b);
                std::memcpy(static_cast<std::uint8_t*>(r.output.data) + out_offset, &value, sizeof(value));
            }
        }
    }
    return {true, nullptr};
}

const Translation cpu_translation = [] {
    Translation t;
    t.backend = Backend::cpu;
    t.name = "cpu-self-contained";
    t.capabilities = {true, true, true, true, true, true};
    t.validate = &validate_cpu;
    t.forward = &forward_cpu;
    return t;
}();
const bool registered = register_translation(&cpu_translation);

} // namespace
} // namespace edcpp::api::attention::flash::cpu::translation
