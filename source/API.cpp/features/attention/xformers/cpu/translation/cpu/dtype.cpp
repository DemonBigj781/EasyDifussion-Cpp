#include "features/attention/xformers/common/xformers.hpp"

#include <cstring>

namespace edcpp::api::attention::xformers::cpu::translation {

std::uint16_t float_to_f16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) return static_cast<std::uint16_t>(sign | (mantissa ? 0x7e00u : 0x7c00u));
    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (adjusted <= 0) {
        if (adjusted < -10) return static_cast<std::uint16_t>(sign);
        std::uint32_t normalized = mantissa | 0x800000u;
        const int shift = 14 - adjusted;
        std::uint32_t half_mantissa = normalized >> shift;
        const std::uint32_t round_bit = 1u << (shift - 1);
        const std::uint32_t remainder = normalized & (round_bit - 1u);
        if ((normalized & round_bit) && (remainder != 0 || (half_mantissa & 1u))) ++half_mantissa;
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    std::uint32_t half_exp = static_cast<std::uint32_t>(adjusted) << 10;
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t round = mantissa & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half_mantissa & 1u))) {
        if (++half_mantissa == 0x400u) { half_mantissa = 0; half_exp += 0x400u; }
    }
    return static_cast<std::uint16_t>(sign | half_exp | half_mantissa);
}

float f16_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int exp = -14;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; --exp; }
            mantissa &= 0x03ffu;
            bits = sign | (static_cast<std::uint32_t>(exp + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) bits = sign | 0x7f800000u | (mantissa << 13);
    else bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float load_scalar(const void* data, std::size_t index, DType dtype) {
    switch (dtype) {
        case DType::f32: return static_cast<const float*>(data)[index];
        case DType::f16: return f16_to_float(static_cast<const std::uint16_t*>(data)[index]);
        case DType::bf16: return bf16_to_float(static_cast<const std::uint16_t*>(data)[index]);
    }
    return 0.0f;
}

void store_scalar(void* data, std::size_t index, DType dtype, float value) {
    switch (dtype) {
        case DType::f32: static_cast<float*>(data)[index] = value; break;
        case DType::f16: static_cast<std::uint16_t*>(data)[index] = float_to_f16(value); break;
        case DType::bf16: static_cast<std::uint16_t*>(data)[index] = float_to_bf16(value); break;
    }
}

} // namespace edcpp::api::attention::xformers::cpu::translation
