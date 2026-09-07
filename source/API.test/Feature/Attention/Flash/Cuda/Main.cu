#include "features/attention/flash/common/flash_attention.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flash = edcpp::api::attention::flash;

namespace {

void require_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        require_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
            "cudaMalloc");
        require_cuda(cudaMemset(data_, 0, count_ * sizeof(T)), "cudaMemset");
    }

    explicit DeviceBuffer(const std::vector<T>& values)
        : DeviceBuffer(values.size()) {
        require_cuda(
            cudaMemcpy(data_, values.data(), count_ * sizeof(T),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy host-to-device");
    }

    DeviceBuffer(std::initializer_list<T> values)
        : DeviceBuffer(std::vector<T>(values)) {}

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }

    std::vector<T> copy_to_host() const {
        std::vector<T> result(count_);
        require_cuda(
            cudaMemcpy(result.data(), data_, count_ * sizeof(T),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy device-to-host");
        return result;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

bool check(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
        return false;
    }
    std::cout << "[PASS] " << message << '\n';
    return true;
}

bool check_array(
        const std::vector<float>& actual,
        const std::vector<float>& expected,
        std::string_view message,
        float tolerance = 2.0e-5f) {
    if (actual.size() != expected.size()) {
        return check(false, std::string(message) + ": size mismatch");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float allowed_difference =
            tolerance + 2.0e-6f * std::fabs(expected[index]);
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) > allowed_difference) {
            std::cerr << "[FAIL] " << message << " at index " << index
                      << ": expected " << expected[index]
                      << " but received " << actual[index] << '\n';
            return false;
        }
    }
    return check(true, message);
}

flash::Request basic_request(
        const DeviceBuffer<float>& query,
        const DeviceBuffer<float>& key,
        const DeviceBuffer<float>& value,
        DeviceBuffer<float>& output) {
    flash::Request request;
    request.query = {query.get(), flash::DType::f32, 2, 2, 1, 1, {}};
    request.key = {key.get(), flash::DType::f32, 2, 2, 1, 1, {}};
    request.value = {value.get(), flash::DType::f32, 2, 2, 1, 1, {}};
    request.output = {output.get(), flash::DType::f32, 2, 2, 1, 1, {}};
    request.scale = 1.0f;
    return request;
}

} // namespace

int main() {
    try {
        using edcpp::api::Backend;

        require_cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp properties{};
        require_cuda(
            cudaGetDeviceProperties(&properties, 0),
            "cudaGetDeviceProperties");
        std::cout << "CUDA device: " << properties.name << " sm_"
                  << properties.major << properties.minor << '\n';

        bool ok = true;
        const auto* translation = flash::translation_for(Backend::cuda);
        ok &= check(
            translation != nullptr,
            "CUDA FlashAttention translation is registered with Common");
        if (translation == nullptr) return EXIT_FAILURE;

        const auto capabilities = flash::capabilities(Backend::cuda);
        ok &= check(
            capabilities.forward && capabilities.additive_mask &&
                capabilities.alibi_bias && capabilities.logit_softcap &&
                capabilities.grouped_query &&
                capabilities.f32_accumulation,
            "CUDA FlashAttention capabilities match the fused Common route");

        std::vector<float> host_query(4);
        std::vector<float> host_key(4);
        std::vector<float> host_value(4);
        std::vector<float> host_output(4);
        flash::Request host_request;
        host_request.query = {
            host_query.data(), flash::DType::f32, 2, 2, 1, 1, {}};
        host_request.key = {
            host_key.data(), flash::DType::f32, 2, 2, 1, 1, {}};
        host_request.value = {
            host_value.data(), flash::DType::f32, 2, 2, 1, 1, {}};
        host_request.output = {
            host_output.data(), flash::DType::f32, 2, 2, 1, 1, {}};
        ok &= check(
            !flash::validate(Backend::cuda, host_request).ok,
            "CUDA validation rejects ordinary host pointers");

        DeviceBuffer<float> query{1.0f, 0.0f, 0.0f, 1.0f};
        DeviceBuffer<float> key{1.0f, 0.0f, 0.0f, 1.0f};
        DeviceBuffer<float> value{10.0f, 20.0f, 30.0f, 40.0f};
        DeviceBuffer<float> output(4);
        flash::Request request = basic_request(query, key, value, output);
        ok &= check(
            flash::validate(Backend::cuda, request).ok,
            "Common validation accepts device-resident F32 tensors");
        ok &= check(
            flash::forward(Backend::cuda, request).ok,
            "Common dispatch executes fused CUDA FlashAttention");

        const float e = std::exp(1.0f);
        const float strong = e / (e + 1.0f);
        const float weak = 1.0f / (e + 1.0f);
        ok &= check_array(
            output.copy_to_host(),
            {
                strong * 10.0f + weak * 30.0f,
                strong * 20.0f + weak * 40.0f,
                weak * 10.0f + strong * 30.0f,
                weak * 20.0f + strong * 40.0f,
            },
            "F32 output matches the analytical attention result");

        DeviceBuffer<float> zero_query_key{0.0f, 0.0f, 0.0f, 0.0f};
        DeviceBuffer<float> scalar_value{10.0f, 30.0f};
        DeviceBuffer<float> additive_mask{
            0.0f, std::log(3.0f), 0.0f, std::log(3.0f)};
        DeviceBuffer<float> masked_output(2);
        flash::Request masked;
        masked.query = {
            zero_query_key.get(), flash::DType::f32, 2, 2, 1, 1, {}};
        masked.key = {
            zero_query_key.get(), flash::DType::f32, 2, 2, 1, 1, {}};
        masked.value = {
            scalar_value.get(), flash::DType::f32, 1, 2, 1, 1, {}};
        masked.mask = {
            additive_mask.get(), flash::DType::f32, 2, 2, 1, 1, {}};
        masked.output = {
            masked_output.get(), flash::DType::f32, 1, 2, 1, 1, {}};
        masked.scale = 1.0f;
        ok &= check(
            flash::forward(Backend::cuda, masked).ok,
            "CUDA FlashAttention applies a normalized additive mask");
        ok &= check_array(
            masked_output.copy_to_host(), {25.0f, 25.0f},
            "Additive mask output matches the analytical result");

        DeviceBuffer<float> grouped_query{1.0f, 1.0f, 1.0f, 1.0f};
        DeviceBuffer<float> grouped_key{1.0f, 0.0f, 0.0f, 1.0f};
        DeviceBuffer<float> grouped_value{10.0f, 30.0f, 100.0f, 300.0f};
        DeviceBuffer<float> grouped_output(4);
        flash::Request grouped;
        grouped.query = {
            grouped_query.get(), flash::DType::f32, 1, 1, 4, 1, {}};
        grouped.key = {
            grouped_key.get(), flash::DType::f32, 1, 2, 2, 1, {}};
        grouped.value = {
            grouped_value.get(), flash::DType::f32, 1, 2, 2, 1, {}};
        grouped.output = {
            grouped_output.get(), flash::DType::f32, 1, 1, 4, 1, {}};
        grouped.scale = 1.0f;
        ok &= check(
            flash::forward(Backend::cuda, grouped).ok,
            "CUDA FlashAttention executes grouped-query attention");
        ok &= check_array(
            grouped_output.copy_to_host(),
            {
                strong * 10.0f + weak * 30.0f,
                strong * 10.0f + weak * 30.0f,
                weak * 100.0f + strong * 300.0f,
                weak * 100.0f + strong * 300.0f,
            },
            "Grouped-query heads map to the expected K/V groups");

        DeviceBuffer<float> permuted_query{0.0f, 0.0f, 0.0f, 0.0f};
        DeviceBuffer<float> permuted_key{0.0f, 0.0f, 0.0f, 0.0f};
        DeviceBuffer<float> permuted_value{10.0f, 30.0f, 100.0f, 300.0f};
        DeviceBuffer<float> permuted_output(4);
        flash::Request permuted;
        permuted.query = {
            permuted_query.get(), flash::DType::f32, 1, 2, 2, 1, {}};
        permuted.key = {
            permuted_key.get(), flash::DType::f32, 1, 2, 2, 1, {}};
        permuted.value = {
            permuted_value.get(), flash::DType::f32, 1, 2, 2, 1, {}};
        permuted.output = {
            permuted_output.get(), flash::DType::f32, 1, 2, 2, 1,
            {sizeof(float), 2 * sizeof(float), sizeof(float),
             4 * sizeof(float)}};
        permuted.scale = 1.0f;
        ok &= check(
            flash::forward(Backend::cuda, permuted).ok,
            "CUDA FlashAttention accepts a non-overlapping GGML output permutation");
        ok &= check_array(
            permuted_output.copy_to_host(), {20.0f, 200.0f, 20.0f, 200.0f},
            "Permuted GGML output strides preserve head/token addressing");
        auto overlapping = permuted;
        overlapping.output.byte_strides = {
            sizeof(float), sizeof(float), sizeof(float), 4 * sizeof(float)};
        ok &= check(
            !flash::validate(Backend::cuda, overlapping).ok,
            "CUDA validation rejects overlapping permuted output strides");

        DeviceBuffer<float> soft_query{2.0f, 0.0f};
        DeviceBuffer<float> soft_key{1.0f, 0.0f, 0.0f, 1.0f};
        DeviceBuffer<float> soft_value{4.0f, 8.0f};
        DeviceBuffer<float> soft_output(1);
        flash::Request softcap;
        softcap.query = {
            soft_query.get(), flash::DType::f32, 2, 1, 1, 1, {}};
        softcap.key = {
            soft_key.get(), flash::DType::f32, 2, 2, 1, 1, {}};
        softcap.value = {
            soft_value.get(), flash::DType::f32, 1, 2, 1, 1, {}};
        softcap.output = {
            soft_output.get(), flash::DType::f32, 1, 1, 1, 1, {}};
        softcap.scale = 1.0f;
        softcap.logit_softcap = 1.0f;
        const float capped_exp = std::exp(std::tanh(2.0f));
        ok &= check(
            flash::forward(Backend::cuda, softcap).ok,
            "CUDA FlashAttention executes logit soft-capping");
        ok &= check_array(
            soft_output.copy_to_host(),
            {(capped_exp * 4.0f + 8.0f) / (capped_exp + 1.0f)},
            "Logit soft-cap output matches the analytical result");

        DeviceBuffer<float> bias_query{0.0f, 0.0f};
        DeviceBuffer<float> bias_key{0.0f, 0.0f};
        DeviceBuffer<float> bias_value{0.0f, 10.0f};
        DeviceBuffer<float> bias_mask{0.0f, std::log(4.0f)};
        DeviceBuffer<float> bias_output(2);
        flash::Request bias;
        bias.query = {
            bias_query.get(), flash::DType::f32, 1, 1, 2, 1, {}};
        bias.key = {
            bias_key.get(), flash::DType::f32, 1, 2, 1, 1, {}};
        bias.value = {
            bias_value.get(), flash::DType::f32, 1, 2, 1, 1, {}};
        bias.mask = {
            bias_mask.get(), flash::DType::f32, 2, 1, 1, 1, {}};
        bias.output = {
            bias_output.get(), flash::DType::f32, 1, 1, 2, 1, {}};
        bias.scale = 1.0f;
        bias.max_bias = 2.0f;
        const float root_two = std::sqrt(2.0f);
        ok &= check(
            flash::forward(Backend::cuda, bias).ok,
            "CUDA FlashAttention executes GGML-style ALiBi mask scaling");
        ok &= check_array(
            bias_output.copy_to_host(),
            {20.0f / 3.0f, 10.0f * root_two / (1.0f + root_two)},
            "ALiBi mask scaling matches the analytical result");

        DeviceBuffer<half> half_query{
            __float2half(1.0f), __float2half(0.0f),
            __float2half(0.0f), __float2half(1.0f)};
        DeviceBuffer<half> half_key{
            __float2half(1.0f), __float2half(0.0f),
            __float2half(0.0f), __float2half(1.0f)};
        DeviceBuffer<half> half_value{
            __float2half(10.0f), __float2half(20.0f),
            __float2half(30.0f), __float2half(40.0f)};
        DeviceBuffer<float> half_output(4);
        flash::Request half_request;
        half_request.query = {
            half_query.get(), flash::DType::f16, 2, 2, 1, 1, {}};
        half_request.key = {
            half_key.get(), flash::DType::f16, 2, 2, 1, 1, {}};
        half_request.value = {
            half_value.get(), flash::DType::f16, 2, 2, 1, 1, {}};
        half_request.output = {
            half_output.get(), flash::DType::f32, 2, 2, 1, 1, {}};
        half_request.scale = 1.0f;
        ok &= check(
            flash::forward(Backend::cuda, half_request).ok,
            "CUDA FlashAttention accepts F16 inputs with F32 accumulation");
        ok &= check_array(
            half_output.copy_to_host(), output.copy_to_host(),
            "F16 inputs agree with the F32 result", 2.0e-4f);

        DeviceBuffer<nv_bfloat16> bf16_query{
            __float2bfloat16(1.0f), __float2bfloat16(0.0f),
            __float2bfloat16(0.0f), __float2bfloat16(1.0f)};
        DeviceBuffer<nv_bfloat16> bf16_key{
            __float2bfloat16(1.0f), __float2bfloat16(0.0f),
            __float2bfloat16(0.0f), __float2bfloat16(1.0f)};
        DeviceBuffer<nv_bfloat16> bf16_value{
            __float2bfloat16(10.0f), __float2bfloat16(20.0f),
            __float2bfloat16(30.0f), __float2bfloat16(40.0f)};
        DeviceBuffer<float> bf16_output(4);
        flash::Request bf16_request;
        bf16_request.query = {
            bf16_query.get(), flash::DType::bf16, 2, 2, 1, 1, {}};
        bf16_request.key = {
            bf16_key.get(), flash::DType::bf16, 2, 2, 1, 1, {}};
        bf16_request.value = {
            bf16_value.get(), flash::DType::bf16, 2, 2, 1, 1, {}};
        bf16_request.output = {
            bf16_output.get(), flash::DType::f32, 2, 2, 1, 1, {}};
        bf16_request.scale = 1.0f;
        ok &= check(
            flash::forward(Backend::cuda, bf16_request).ok,
            "CUDA FlashAttention accepts BF16 inputs with F32 accumulation");
        ok &= check_array(
            bf16_output.copy_to_host(), output.copy_to_host(),
            "BF16 inputs agree with the F32 result", 2.0e-4f);

        auto oversized = request;
        oversized.value.head_dim = 513;
        oversized.output.head_dim = 513;
        ok &= check(
            !flash::validate(Backend::cuda, oversized).ok,
            "CUDA validation rejects value dimensions above 512");

        std::cout << (ok
            ? "CUDA FlashAttention normalized Common route passed\n"
            : "CUDA FlashAttention normalized Common route failed\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
