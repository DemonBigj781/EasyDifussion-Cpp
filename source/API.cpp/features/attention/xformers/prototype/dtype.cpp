#include "xformers.hpp"

#include <cstring>
#include <limits>

namespace xformers::prototype {

std::uint16_t float_to_f16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    const std::uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        return static_cast<std::uint16_t>(sign | 0x7e00u);
    }

    const int adjusted_exp = static_cast<int>(exponent) - 127 + 15;
    if (adjusted_exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    if (adjusted_exp <= 0) {
        if (adjusted_exp < -10) {
            return static_cast<std::uint16_t>(sign);
        }

        std::uint32_t normalized = mantissa | 0x800000u;
        const int shift = 14 - adjusted_exp;
        std::uint32_t half_mantissa = normalized >> shift;
        const std::uint32_t round_bit = 1u << (shift - 1);
        const std::uint32_t remainder = normalized & (round_bit - 1u);
        const bool tie = (normalized & round_bit) != 0;
        if (tie && (remainder != 0 || (half_mantissa & 1u))) {
            ++half_mantissa;
        }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_exp = static_cast<std::uint32_t>(adjusted_exp) << 10;
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t round = mantissa & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            half_exp += 0x400u;
            if (half_exp >= 0x7c00u) {
                half_exp = 0x7c00u;
            }
        }
    }

    return static_cast<std::uint16_t>(sign | half_exp | half_mantissa);
}

float f16_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int exp = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exp;
            }
            mantissa &= 0x03ffu;
            const std::uint32_t exp32 = static_cast<std::uint32_t>(exp + 127);
            bits = sign | (exp32 << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        const std::uint32_t exp32 = exponent - 15 + 127;
        bits = sign | (exp32 << 23) | (mantissa << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float load_scalar(const void * data, std::size_t index, DType dtype) {
    switch (dtype) {
        case DType::F32:
            return static_cast<const float *>(data)[index];
        case DType::F16:
            return f16_to_float(static_cast<const std::uint16_t *>(data)[index]);
        case DType::BF16:
            return 0.0f;
    }
    return 0.0f;
}

void store_scalar(void * data, std::size_t index, DType dtype, float value) {
    switch (dtype) {
        case DType::F32:
            static_cast<float *>(data)[index] = value;
            return;
        case DType::F16:
            static_cast<std::uint16_t *>(data)[index] = float_to_f16(value);
            return;
        case DType::BF16:
            return;
    }
}

} // namespace xformers::prototype
