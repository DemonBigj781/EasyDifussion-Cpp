#include "api/model_lifecycle.hpp"
#include "features/attention/xformers/common/xformers.hpp"
#include "model_fixture.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using edcpp::api::Backend;
using edcpp::api::attention::xformers::AttentionRequest;
using edcpp::api::attention::xformers::DType;
using edcpp::api::attention::xformers::MaskView;
using edcpp::api::attention::xformers::MutableTensor4D;
using edcpp::api::attention::xformers::ScoreBuffer;
using edcpp::api::attention::xformers::Tensor4D;
using edcpp::api::attention::xformers::capabilities;
using edcpp::api::attention::xformers::forward;
using edcpp::api::attention::xformers::qkt;
using edcpp::api::attention::xformers::translation_for;
using edcpp::api::attention::xformers::validate;

constexpr float absolute_tolerance = 2.0e-5f;
constexpr float relative_tolerance = 2.0e-6f;

void require_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class TypedDeviceBuffer {
public:
    explicit TypedDeviceBuffer(std::size_t count) : count_(count) {
        require_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
            "cudaMalloc");
        require_cuda(cudaMemset(data_, 0, count_ * sizeof(T)), "cudaMemset");
    }

    explicit TypedDeviceBuffer(const std::vector<T>& values)
        : TypedDeviceBuffer(values.size()) {
        copy_from(values);
    }

    TypedDeviceBuffer(std::initializer_list<T> values)
        : TypedDeviceBuffer(std::vector<T>(values)) {}

    TypedDeviceBuffer(const TypedDeviceBuffer&) = delete;
    TypedDeviceBuffer& operator=(const TypedDeviceBuffer&) = delete;

    TypedDeviceBuffer(TypedDeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    ~TypedDeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void copy_from(const std::vector<T>& values) {
        if (values.size() != count_) {
            throw std::runtime_error("device-buffer input size mismatch");
        }
        require_cuda(
            cudaMemcpy(data_, values.data(), count_ * sizeof(T),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy host-to-device");
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> values(count_);
        require_cuda(
            cudaMemcpy(values.data(), data_, count_ * sizeof(T),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy device-to-host");
        return values;
    }

    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

using DeviceBuffer = TypedDeviceBuffer<float>;

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
        std::string_view message) {
    if (actual.size() != expected.size()) {
        std::cerr << "[FAIL] " << message << ": size mismatch\n";
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float difference = std::fabs(actual[index] - expected[index]);
        const float allowed_difference =
            absolute_tolerance + relative_tolerance * std::fabs(expected[index]);
        if (!std::isfinite(actual[index]) ||
            difference > allowed_difference) {
            std::cerr << "[FAIL] " << message << " at index " << index
                      << ": expected " << expected[index]
                      << " but received " << actual[index]
                      << " (difference " << difference
                      << ", allowed " << allowed_difference << ")\n";
            return false;
        }
    }
    std::cout << "[PASS] " << message << '\n';
    return true;
}

AttentionRequest basic_request(
        const DeviceBuffer& q,
        const DeviceBuffer& k,
        const DeviceBuffer& v,
        DeviceBuffer& out) {
    AttentionRequest request{};
    request.q = Tensor4D{q.get(), 1, 1, 2, 2};
    request.k = Tensor4D{k.get(), 1, 1, 2, 2};
    request.v = Tensor4D{v.get(), 1, 1, 2, 2};
    request.out = MutableTensor4D{out.get(), 1, 1, 2, 2};
    request.dtype = DType::f32;
    request.scale = 1.0f;
    return request;
}

} // namespace

int main(int argc, char** argv) {
    try {
        require_cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp device{};
        require_cuda(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
        std::cout << "CUDA device: " << device.name << " sm_"
                  << device.major << device.minor << '\n';

        bool ok = true;
        api_test::ModelFixture fixture;
        try {
            fixture = api_test::read_model_fixture(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
        easyapi::CudaModelLifecycleHandler lifecycle;
        ok &= check(std::string_view(lifecycle.name()) == "cuda",
                    "CUDA model lifecycle handler is available");
        const auto* translation = translation_for(Backend::cuda);
        ok &= check(translation != nullptr,
                    "CUDA xFormers translation is registered with Common");
        if (translation == nullptr) return EXIT_FAILURE;

        const auto caps = capabilities(Backend::cuda);
        ok &= check(caps.forward, "CUDA xFormers advertises fused forward support");
        ok &= check(!caps.qkt,
                    "CUDA xFormers does not advertise a materialized Common QKT stage");
        ok &= check(caps.additive_mask, "CUDA xFormers advertises additive masks");
        ok &= check(caps.causal_mask, "CUDA xFormers advertises causal masks");
        ok &= check(caps.alibi, "CUDA xFormers advertises ALiBi");
        ok &= check(caps.softcap, "CUDA xFormers advertises soft-cap support");
        ok &= check(caps.gqa, "CUDA xFormers advertises GQA");
        ok &= check(caps.mqa, "CUDA xFormers advertises MQA");
        ok &= check(caps.attention_sinks,
                    "CUDA xFormers advertises attention-sink support");
        ok &= check(caps.f32 && caps.f16 && caps.bf16,
                    "CUDA xFormers advertises F32, F16, and BF16 inputs");

        const float e = std::exp(1.0f);
        const float strong = e / (e + 1.0f);
        const float weak = 1.0f / (e + 1.0f);

        DeviceBuffer q{1.0f, 0.0f, 0.0f, 1.0f};
        DeviceBuffer k{1.0f, 0.0f, 0.0f, 1.0f};
        DeviceBuffer v{10.0f, 20.0f, 30.0f, 40.0f};
        DeviceBuffer out(4);
        AttentionRequest request = basic_request(q, k, v, out);
        ok &= check(validate(Backend::cuda, request).ok,
                    "Common validation accepts a device-resident CUDA request");
        ScoreBuffer materialized_scores;
        ok &= check(!qkt(Backend::cuda, request, materialized_scores),
                    "Common does not force fused CUDA through a staged score buffer");
        ok &= check(forward(Backend::cuda, request),
                    "Common dispatch executes CUDA fused forward");
        ok &= check_array(
            out.copy_to_host(),
            {
                strong * 10.0f + weak * 30.0f,
                strong * 20.0f + weak * 40.0f,
                weak * 10.0f + strong * 30.0f,
                weak * 20.0f + strong * 40.0f,
            },
            "CUDA deterministic output matches the analytical result");

        DeviceBuffer zero_qk{0.0f, 0.0f, 0.0f, 0.0f};
        DeviceBuffer scalar_v{10.0f, 30.0f};
        DeviceBuffer causal_out(2);
        AttentionRequest causal{};
        causal.q = Tensor4D{zero_qk.get(), 1, 1, 2, 2};
        causal.k = Tensor4D{zero_qk.get(), 1, 1, 2, 2};
        causal.v = Tensor4D{scalar_v.get(), 1, 1, 2, 1};
        causal.out = MutableTensor4D{causal_out.get(), 1, 1, 2, 1};
        causal.dtype = DType::f32;
        causal.scale = 1.0f;
        causal.causal = true;
        ok &= check(forward(Backend::cuda, causal),
                    "CUDA Common dispatch executes causal attention");
        ok &= check_array(causal_out.copy_to_host(), {10.0f, 20.0f},
                          "CUDA causal mask matches the Common semantics");

        DeviceBuffer additive_mask{0.0f, std::log(3.0f)};
        DeviceBuffer mask_out(2);
        AttentionRequest masked = causal;
        masked.causal = false;
        masked.out.data = mask_out.get();
        masked.mask = MaskView{additive_mask.get(), 1, 1, 1, 2};
        ok &= check(forward(Backend::cuda, masked),
                    "CUDA Common dispatch executes a broadcast additive mask");
        ok &= check_array(mask_out.copy_to_host(), {25.0f, 25.0f},
                          "CUDA additive-mask broadcasting matches Common");

        DeviceBuffer alibi_slopes{std::log(3.0f)};
        DeviceBuffer alibi_out(2);
        AttentionRequest alibi = causal;
        alibi.causal = false;
        alibi.out.data = alibi_out.get();
        alibi.alibi.enabled = true;
        alibi.alibi.slopes = alibi_slopes.get();
        alibi.alibi.slope_count = 1;
        ok &= check(forward(Backend::cuda, alibi),
                    "CUDA Common dispatch executes ALiBi");
        ok &= check_array(alibi_out.copy_to_host(), {25.0f, 25.0f},
                          "CUDA ALiBi matches the Common semantics");

        DeviceBuffer soft_q{2.0f, 0.0f};
        DeviceBuffer soft_k{1.0f, 0.0f, 0.0f, 1.0f};
        DeviceBuffer soft_v{4.0f, 8.0f};
        DeviceBuffer soft_out(1);
        AttentionRequest softcap{};
        softcap.q = Tensor4D{soft_q.get(), 1, 1, 1, 2};
        softcap.k = Tensor4D{soft_k.get(), 1, 1, 2, 2};
        softcap.v = Tensor4D{soft_v.get(), 1, 1, 2, 1};
        softcap.out = MutableTensor4D{soft_out.get(), 1, 1, 1, 1};
        softcap.dtype = DType::f32;
        softcap.scale = 1.0f;
        softcap.softcap = 1.0f;
        const float capped_exp = std::exp(std::tanh(2.0f));
        ok &= check(forward(Backend::cuda, softcap),
                    "CUDA Common dispatch executes soft-capped attention");
        ok &= check_array(
            soft_out.copy_to_host(),
            {(capped_exp * 4.0f + 8.0f) / (capped_exp + 1.0f)},
            "CUDA soft-cap matches the analytical result");

        DeviceBuffer gqa_q{
            1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0,
        };
        DeviceBuffer gqa_k{1,0, 0,1, 0,1, 1,0};
        DeviceBuffer gqa_v{10,30, 100,300};
        DeviceBuffer gqa_out(8);
        AttentionRequest gqa{};
        gqa.q = Tensor4D{gqa_q.get(), 1, 4, 2, 2};
        gqa.k = Tensor4D{gqa_k.get(), 1, 2, 2, 2};
        gqa.v = Tensor4D{gqa_v.get(), 1, 2, 2, 1};
        gqa.out = MutableTensor4D{gqa_out.get(), 1, 4, 2, 1};
        gqa.dtype = DType::f32;
        gqa.scale = 1.0f;
        ok &= check(forward(Backend::cuda, gqa),
                    "CUDA Common dispatch executes GQA");
        ok &= check_array(
            gqa_out.copy_to_host(),
            {
                strong*10 + weak*30, strong*10 + weak*30,
                strong*10 + weak*30, strong*10 + weak*30,
                weak*100 + strong*300, weak*100 + strong*300,
                weak*100 + strong*300, weak*100 + strong*300,
            },
            "CUDA GQA maps Q-head groups to the expected K/V heads");

        DeviceBuffer mqa_q{1,0, 0,1, 1,0, 0,1};
        DeviceBuffer mqa_k{1,0, 0,1};
        DeviceBuffer mqa_v{5,9};
        DeviceBuffer mqa_out(4);
        AttentionRequest mqa{};
        mqa.q = Tensor4D{mqa_q.get(), 1, 4, 1, 2};
        mqa.k = Tensor4D{mqa_k.get(), 1, 1, 2, 2};
        mqa.v = Tensor4D{mqa_v.get(), 1, 1, 2, 1};
        mqa.out = MutableTensor4D{mqa_out.get(), 1, 4, 1, 1};
        mqa.dtype = DType::f32;
        mqa.scale = 1.0f;
        ok &= check(forward(Backend::cuda, mqa),
                    "CUDA Common dispatch executes MQA");
        ok &= check_array(
            mqa_out.copy_to_host(),
            {
                strong*5 + weak*9, weak*5 + strong*9,
                strong*5 + weak*9, weak*5 + strong*9,
            },
            "CUDA MQA shares one K/V head across all Q heads");

        DeviceBuffer batch_q{1, 1};
        DeviceBuffer batch_k{1, 0, 0, 1};
        DeviceBuffer batch_v{10, 30, 100, 300};
        DeviceBuffer batch_out(2);
        AttentionRequest batched{};
        batched.q = Tensor4D{batch_q.get(), 2, 1, 1, 1};
        batched.k = Tensor4D{batch_k.get(), 2, 1, 2, 1};
        batched.v = Tensor4D{batch_v.get(), 2, 1, 2, 1};
        batched.out = MutableTensor4D{batch_out.get(), 2, 1, 1, 1};
        batched.dtype = DType::f32;
        batched.scale = 1.0f;
        ok &= check(forward(Backend::cuda, batched),
                    "CUDA Common dispatch executes independent batches");
        ok &= check_array(
            batch_out.copy_to_host(),
            {strong*10 + weak*30, weak*100 + strong*300},
            "CUDA batch addressing keeps inputs and outputs isolated");

        TypedDeviceBuffer<half> half_q{
            __float2half(1.0f), __float2half(0.0f),
            __float2half(0.0f), __float2half(1.0f)};
        TypedDeviceBuffer<half> half_k{
            __float2half(1.0f), __float2half(0.0f),
            __float2half(0.0f), __float2half(1.0f)};
        TypedDeviceBuffer<half> half_v{
            __float2half(10.0f), __float2half(20.0f),
            __float2half(30.0f), __float2half(40.0f)};
        DeviceBuffer half_out(4);
        AttentionRequest half_request{};
        half_request.q = Tensor4D{half_q.get(), 1, 1, 2, 2, DType::f16};
        half_request.k = Tensor4D{half_k.get(), 1, 1, 2, 2, DType::f16};
        half_request.v = Tensor4D{half_v.get(), 1, 1, 2, 2, DType::f16};
        half_request.out = MutableTensor4D{half_out.get(), 1, 1, 2, 2};
        half_request.dtype = DType::f16;
        half_request.scale = 1.0f;
        ok &= check(forward(Backend::cuda, half_request),
                    "CUDA Common dispatch executes F16 input attention");
        ok &= check_array(
            half_out.copy_to_host(),
            {
                strong * 10.0f + weak * 30.0f,
                strong * 20.0f + weak * 40.0f,
                weak * 10.0f + strong * 30.0f,
                weak * 20.0f + strong * 40.0f,
            },
            "CUDA F16 inputs match the analytical result");

        TypedDeviceBuffer<nv_bfloat16> bf16_q{
            __float2bfloat16(1.0f), __float2bfloat16(0.0f),
            __float2bfloat16(0.0f), __float2bfloat16(1.0f)};
        TypedDeviceBuffer<nv_bfloat16> bf16_k{
            __float2bfloat16(1.0f), __float2bfloat16(0.0f),
            __float2bfloat16(0.0f), __float2bfloat16(1.0f)};
        TypedDeviceBuffer<nv_bfloat16> bf16_v{
            __float2bfloat16(10.0f), __float2bfloat16(20.0f),
            __float2bfloat16(30.0f), __float2bfloat16(40.0f)};
        DeviceBuffer bf16_out(4);
        AttentionRequest bf16_request{};
        bf16_request.q = Tensor4D{
            bf16_q.get(), 1, 1, 2, 2, DType::bf16};
        bf16_request.k = Tensor4D{
            bf16_k.get(), 1, 1, 2, 2, DType::bf16};
        bf16_request.v = Tensor4D{
            bf16_v.get(), 1, 1, 2, 2, DType::bf16};
        bf16_request.out = MutableTensor4D{bf16_out.get(), 1, 1, 2, 2};
        bf16_request.dtype = DType::bf16;
        bf16_request.scale = 1.0f;
        ok &= check(forward(Backend::cuda, bf16_request),
                    "CUDA Common dispatch executes BF16 input attention");
        ok &= check_array(
            bf16_out.copy_to_host(),
            {
                strong * 10.0f + weak * 30.0f,
                strong * 20.0f + weak * 40.0f,
                weak * 10.0f + strong * 30.0f,
                weak * 20.0f + strong * 40.0f,
            },
            "CUDA BF16 inputs match the analytical result");

        DeviceBuffer strided_q{1, 0, 99, 0, 1, 99};
        DeviceBuffer strided_k{1, 0, 99, 0, 1, 99};
        DeviceBuffer strided_v{10, 99, 30, 99};
        DeviceBuffer strided_out(4);
        AttentionRequest strided{};
        strided.q = Tensor4D{
            strided_q.get(), 1, 1, 2, 2, DType::f32,
            {sizeof(float), 3 * sizeof(float), 6 * sizeof(float),
             6 * sizeof(float)}};
        strided.k = Tensor4D{
            strided_k.get(), 1, 1, 2, 2, DType::f32,
            {sizeof(float), 3 * sizeof(float), 6 * sizeof(float),
             6 * sizeof(float)}};
        strided.v = Tensor4D{
            strided_v.get(), 1, 1, 2, 1, DType::f32,
            {sizeof(float), 2 * sizeof(float), 4 * sizeof(float),
             4 * sizeof(float)}};
        strided.out = MutableTensor4D{
            strided_out.get(), 1, 1, 2, 1, DType::f32,
            {sizeof(float), 2 * sizeof(float), 4 * sizeof(float),
             4 * sizeof(float)}};
        strided.scale = 1.0f;
        ok &= check(forward(Backend::cuda, strided),
                    "CUDA Common dispatch executes byte-strided tensors");
        const auto strided_values = strided_out.copy_to_host();
        ok &= check_array(
            {strided_values[0], strided_values[2]},
            {strong * 10.0f + weak * 30.0f,
             weak * 10.0f + strong * 30.0f},
            "CUDA byte strides preserve logical tensor addressing");

        DeviceBuffer sink_values{0.0f};
        DeviceBuffer sink_out(2);
        AttentionRequest sinks = causal;
        sinks.causal = false;
        sinks.out.data = sink_out.get();
        sinks.sinks.enabled = true;
        sinks.sinks.values = sink_values.get();
        sinks.sinks.value_count = 1;
        ok &= check(forward(Backend::cuda, sinks),
                    "CUDA Common dispatch executes attention sinks");
        ok &= check_array(
            sink_out.copy_to_host(),
            {40.0f / 3.0f, 40.0f / 3.0f},
            "CUDA attention sinks contribute to the softmax denominator");

        bool cycle_ok = true;
        const int cycle_count =
            fixture.bytes.size() > (64u * 1024u * 1024u) ? 1 : 32;
        const std::size_t prefix_size =
            std::min<std::size_t>(fixture.bytes.size(), 64u);
        std::vector<unsigned char> loaded_prefix(prefix_size);
        for (int cycle = 0; cycle < cycle_count; ++cycle) {
            auto loaded = lifecycle.load(
                fixture.bytes.data(), fixture.bytes.size(), 0);
            cycle_ok &= loaded.success && loaded.resource.loaded();
            cycle_ok &= loaded.resource.storage == easyapi::ModelStorage::cuda;
            cycle_ok &= loaded.resource.size == fixture.bytes.size();
            if (loaded.success) {
                const cudaError_t copy_status = cudaMemcpy(
                    loaded_prefix.data(), loaded.resource.native_handle,
                    prefix_size, cudaMemcpyDeviceToHost);
                cycle_ok &= copy_status == cudaSuccess;
                cycle_ok &= std::equal(
                    loaded_prefix.begin(), loaded_prefix.end(),
                    fixture.bytes.begin());
            }

            DeviceBuffer cycle_q{1.0f, 0.0f, 0.0f, 1.0f};
            DeviceBuffer cycle_k{1.0f, 0.0f, 0.0f, 1.0f};
            DeviceBuffer cycle_v{10.0f, 20.0f, 30.0f, 40.0f};
            DeviceBuffer cycle_out(4);
            AttentionRequest cycle_request =
                basic_request(cycle_q, cycle_k, cycle_v, cycle_out);
            cycle_ok &= forward(Backend::cuda, cycle_request);
            const auto values = cycle_out.copy_to_host();
            cycle_ok &= std::fabs(values[0] -
                (strong * 10.0f + weak * 30.0f)) <= absolute_tolerance;
            cycle_ok &= std::fabs(values[3] -
                (weak * 20.0f + strong * 40.0f)) <= absolute_tolerance;
            if (loaded.success) {
                const auto unloaded = lifecycle.unload(loaded.resource);
                cycle_ok &= unloaded.success && !loaded.resource.loaded();
            }
        }
        require_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after cycles");
        ok &= check(cycle_ok,
                    "CUDA xFormers survives Common model load/forward/unload cycles");
        std::cout << "CUDA lifecycle cycles: " << cycle_count
                  << " using " << fixture.path << '\n';

        AttentionRequest host_pointer = request;
        const float host_q[] = {1, 0, 0, 1};
        host_pointer.q.data = host_q;
        ok &= check(!validate(Backend::cuda, host_pointer).ok,
                    "CUDA Common validation rejects pageable host tensor pointers");

        AttentionRequest bad_heads = request;
        bad_heads.q.heads = 3;
        bad_heads.k.heads = 2;
        bad_heads.v.heads = 2;
        bad_heads.out.heads = 3;
        ok &= check(!validate(Backend::cuda, bad_heads).ok,
                    "CUDA Common validation rejects non-divisible K/V heads");

        AttentionRequest bad_mask = request;
        DeviceBuffer invalid_mask{0.0f};
        bad_mask.mask = MaskView{invalid_mask.get(), 2, 1, 1, 1};
        ok &= check(!validate(Backend::cuda, bad_mask).ok,
                    "CUDA Common validation rejects incompatible mask dimensions");

        AttentionRequest bad_output = request;
        bad_output.out.tokens = 1;
        ok &= check(!validate(Backend::cuda, bad_output).ok,
                    "CUDA Common validation rejects a mismatched output shape");

        AttentionRequest negative_scale = request;
        negative_scale.scale = -1.0f;
        ok &= check(!validate(Backend::cuda, negative_scale).ok,
                    "CUDA Common validation rejects a negative scale");

        AttentionRequest bad_softcap = request;
        bad_softcap.softcap = std::numeric_limits<float>::infinity();
        ok &= check(!validate(Backend::cuda, bad_softcap).ok,
                    "CUDA Common validation rejects a non-finite soft-cap");

        AttentionRequest missing_alibi = request;
        missing_alibi.alibi.enabled = true;
        missing_alibi.alibi.slopes = nullptr;
        missing_alibi.alibi.slope_count = 0;
        ok &= check(!validate(Backend::cuda, missing_alibi).ok,
                    "CUDA Common validation rejects missing ALiBi slopes");

        AttentionRequest null_q = request;
        null_q.q.data = nullptr;
        ok &= check(!validate(Backend::cuda, null_q).ok,
                    "CUDA Common validation rejects a null Q buffer");

        AttentionRequest oversized_value = request;
        oversized_value.v.head_dim = 513;
        oversized_value.out.head_dim = 513;
        ok &= check(!validate(Backend::cuda, oversized_value).ok,
                    "CUDA Common validation rejects value dimensions above 512");

        if (!ok) {
            std::cerr << "CUDA xFormers Common API test failed\n";
            return EXIT_FAILURE;
        }
        std::cout << "CUDA xFormers Common API test passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] CUDA test setup: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
