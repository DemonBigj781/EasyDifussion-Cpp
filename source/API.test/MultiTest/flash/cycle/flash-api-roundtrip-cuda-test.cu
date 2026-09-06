#include "api/model_lifecycle.hpp"
#include "features/attention/flash/common/flash_attention.hpp"
#include "model_fixture.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
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

class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::vector<float>& values)
        : count_(values.size()) {
        require_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(float)),
            "cudaMalloc");
        require_cuda(
            cudaMemcpy(data_, values.data(), count_ * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy host-to-device");
    }

    explicit DeviceBuffer(std::size_t count)
        : count_(count) {
        require_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(float)),
            "cudaMalloc");
        require_cuda(cudaMemset(data_, 0, count_ * sizeof(float)), "cudaMemset");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    float* get() noexcept { return data_; }
    const float* get() const noexcept { return data_; }

    std::vector<float> copy_to_host() const {
        std::vector<float> result(count_);
        require_cuda(
            cudaMemcpy(result.data(), data_, count_ * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy device-to-host");
        return result;
    }

private:
    float* data_ = nullptr;
    std::size_t count_ = 0;
};

bool nearly_equal(float actual, float expected) {
    const float difference = std::fabs(actual - expected);
    return std::isfinite(actual) &&
        difference <= 2.0e-5f + 2.0e-6f * std::fabs(expected);
}

bool check(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
        return false;
    }
    std::cout << "[PASS] " << message << '\n';
    return true;
}

flash::Request make_request(
        const DeviceBuffer& query,
        const DeviceBuffer& key,
        const DeviceBuffer& value,
        DeviceBuffer& output) {
    flash::Request request;
    request.query = {
        query.get(), flash::DType::f32, 2, 2, 1, 1, {}};
    request.key = {
        key.get(), flash::DType::f32, 2, 2, 1, 1, {}};
    request.value = {
        value.get(), flash::DType::f32, 2, 2, 1, 1, {}};
    request.output = {
        output.get(), flash::DType::f32, 2, 2, 1, 1, {}};
    request.scale = 1.0f;
    return request;
}

} // namespace

int main(int argc, char** argv) {
    try {
        using edcpp::api::Backend;

        const api_test::ModelFixture fixture =
            api_test::read_model_fixture(argc, argv);
        require_cuda(cudaSetDevice(0), "cudaSetDevice");

        cudaDeviceProp properties{};
        require_cuda(
            cudaGetDeviceProperties(&properties, 0),
            "cudaGetDeviceProperties");
        std::cout << "CUDA device: " << properties.name << " sm_"
                  << properties.major << properties.minor << '\n';

        bool ok = true;
        easyapi::CudaModelLifecycleHandler lifecycle;
        ok &= check(
            std::string_view(lifecycle.name()) == "cuda",
            "CUDA model lifecycle API is available");
        ok &= check(
            flash::translation_for(Backend::cuda) != nullptr,
            "CUDA FlashAttention translation is registered with Common");
        if (!ok) return EXIT_FAILURE;

        DeviceBuffer query(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f});
        DeviceBuffer key(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f});
        DeviceBuffer value(
            std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f});
        DeviceBuffer output(4);
        flash::Request request = make_request(query, key, value, output);

        const float e = std::exp(1.0f);
        const float strong = e / (e + 1.0f);
        const float weak = 1.0f / (e + 1.0f);
        const std::vector<float> expected = {
            strong * 10.0f + weak * 30.0f,
            strong * 20.0f + weak * 40.0f,
            weak * 10.0f + strong * 30.0f,
            weak * 20.0f + strong * 40.0f,
        };

        const int cycle_count =
            fixture.bytes.size() > (64u * 1024u * 1024u) ? 1 : 32;
        std::vector<unsigned char> model_roundtrip(fixture.bytes.size());
        bool cycles_ok = true;
        for (int cycle = 0; cycle < cycle_count; ++cycle) {
            auto loaded = lifecycle.load(
                fixture.bytes.data(), fixture.bytes.size(), 0);
            if (!loaded.success || !loaded.resource.loaded() ||
                loaded.resource.storage != easyapi::ModelStorage::cuda ||
                loaded.resource.size != fixture.bytes.size()) {
                std::cerr << "[FAIL] cycle " << cycle
                          << " model load: " << loaded.diagnostic << '\n';
                cycles_ok = false;
                break;
            }

            const cudaError_t copy_status = cudaMemcpy(
                model_roundtrip.data(), loaded.resource.native_handle,
                model_roundtrip.size(), cudaMemcpyDeviceToHost);
            if (copy_status != cudaSuccess ||
                model_roundtrip != fixture.bytes) {
                std::cerr << "[FAIL] cycle " << cycle
                          << " model-byte roundtrip\n";
                cycles_ok = false;
            }

            const flash::Result validation =
                flash::validate(Backend::cuda, request);
            const flash::Result forward =
                flash::forward(Backend::cuda, request);
            if (!validation.ok || !forward.ok) {
                const char* diagnostic =
                    validation.ok ? forward.message : validation.message;
                std::cerr << "[FAIL] cycle " << cycle
                          << " FlashAttention API roundtrip: "
                          << (diagnostic == nullptr ? "no diagnostic" : diagnostic)
                          << '\n';
                cycles_ok = false;
            } else {
                const std::vector<float> actual = output.copy_to_host();
                for (std::size_t index = 0; index < actual.size(); ++index) {
                    if (!nearly_equal(actual[index], expected[index])) {
                        std::cerr << "[FAIL] cycle " << cycle
                                  << " output " << index
                                  << ": expected " << expected[index]
                                  << " but received " << actual[index] << '\n';
                        cycles_ok = false;
                        break;
                    }
                }
            }

            const auto unloaded = lifecycle.unload(loaded.resource);
            if (!unloaded.success || loaded.resource.loaded()) {
                std::cerr << "[FAIL] cycle " << cycle
                          << " model unload: " << unloaded.diagnostic << '\n';
                cycles_ok = false;
                break;
            }
            if (!cycles_ok) break;
        }

        require_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize after API roundtrips");
        ok &= check(cycles_ok, "CUDA model bytes and FlashAttention results complete API roundtrips");
        std::cout << "Completed API roundtrips: " << cycle_count << '\n';
        std::cout << "Roundtrip fixture: " << fixture.path << '\n';

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] CUDA FlashAttention roundtrip setup: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
