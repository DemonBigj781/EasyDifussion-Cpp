#include "api/gguf_cpu_weights.hpp"
#include "features/weights/common/weights.hpp"
#include "ggml.h"
#include "gguf.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

static_assert(!std::is_copy_constructible_v<easyapi::GgufCpuWeights>);
static_assert(std::is_nothrow_move_constructible_v<easyapi::GgufCpuWeights>);
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Function> static void rejects(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid GGUF operation was accepted");
}
const std::array<float, 6> weights{1, -2, 3, 4, 0.5f, 6};
const std::array<float, 3> bias{0.25f, 0.5f, 0.75f};

static std::size_t write_fixture(const std::filesystem::path& file) {
    auto context = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({4096, nullptr, false}), &ggml_free);
    auto metadata = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_empty(), &gguf_free);
    require(context && metadata, "fixture context allocation failed");
    auto* w = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 3, 2);
    auto* b = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 3);
    ggml_set_name(w, "linear.weight");
    ggml_set_name(b, "linear.bias");
    std::memcpy(w->data, weights.data(), sizeof(weights));
    std::memcpy(b->data, bias.data(), sizeof(bias));
    gguf_set_val_str(metadata.get(), "general.architecture", "synthetic-linear-fixture");
    gguf_add_tensor(metadata.get(), w);
    gguf_add_tensor(metadata.get(), b);
    require(gguf_write_to_file(metadata.get(), file.string().c_str(), false), "fixture write failed");
    return gguf_get_meta_size(metadata.get()) + gguf_get_tensor_offset(metadata.get(), 1) + sizeof(bias);
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto root = std::filesystem::path(argv[1]) /
        ("gguf-contract-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(root)) return 3;
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
    } cleanup{root};
    try {
        const auto file = root / "synthetic.gguf";
        const auto required_file = write_fixture(file);
        for (int cycle = 0; cycle < 16; ++cycle) {
            auto loaded = easyapi::GgufCpuWeights::load(file.string());
            require(loaded.tensor_count() == 2 && loaded.backend_bytes() >= sizeof(weights) + sizeof(bias),
                    "wrong tensor resource size");
            const auto info = loaded.tensor_info(0);
            require(info.name == "linear.weight" && info.type == "f32" &&
                    info.shape == std::vector<std::int64_t>({3, 2}) && info.bytes == sizeof(weights),
                    "tensor metadata changed");
            auto moved = std::move(loaded);
            require(loaded.empty() && loaded.tensor_count() == 0, "moved resource retained ownership");
            auto output = moved.read_tensor("linear.weight", 0, sizeof(weights));
            require(std::memcmp(output.data(), weights.data(), sizeof(weights)) == 0, "weight upload/readback changed");
            output = moved.read_tensor("linear.bias", sizeof(float), 2 * sizeof(float));
            require(std::memcmp(output.data(), bias.data() + 1, 2 * sizeof(float)) == 0, "partial readback changed");
            require(moved.read_tensor("linear.bias", sizeof(bias), 0).empty(), "zero-size read failed");
            rejects([&] { moved.read_tensor("missing", 0, 1); });
            rejects([&] { moved.read_tensor("linear.bias", sizeof(bias), 1); });
            rejects([&] { moved.tensor_info(2); });
            if (cycle == 0) {
                auto replacement = easyapi::GgufCpuWeights::load(file.string());
                replacement = std::move(moved);
                require(moved.empty() && replacement.tensor_count() == 2, "move assignment lost weight ownership");
                moved = std::move(replacement);
                auto* self = &moved;
                moved = std::move(*self);
                require(moved.tensor_count() == 2, "self move lost weights");
                std::filesystem::remove(file);
                require(moved.read_tensor("linear.weight", 0, sizeof(weights)).size() == sizeof(weights),
                        "backend weights still depend on open source file");
                write_fixture(file);
                std::atomic<int> failures{0};
                std::vector<std::thread> readers;
                for (int thread = 0; thread < 4; ++thread) readers.emplace_back([&] {
                    try {
                        for (int repeat = 0; repeat < 64; ++repeat) {
                            auto result = moved.read_tensor("linear.weight", 0, sizeof(weights));
                            if (std::memcmp(result.data(), weights.data(), sizeof(weights))) ++failures;
                        }
                    } catch (...) { ++failures; }
                });
                for (auto& reader : readers) reader.join();
                require(failures == 0, "concurrent immutable readback failed");
            }
            moved.reset();
            moved.reset();
            rejects([&] { moved.read_tensor("linear.weight", 0, 1); });
        }
        auto limits = easyapi::GgufLimits{};
        limits.max_file_bytes = 1;
        rejects([&] { (void)easyapi::GgufCpuWeights::load(file.string(), limits); });
        limits = {}; limits.max_backend_bytes = 1;
        rejects([&] { (void)easyapi::GgufCpuWeights::load(file.string(), limits); });
        limits = {}; limits.max_tensors = 1;
        rejects([&] { (void)easyapi::GgufCpuWeights::load(file.string(), limits); });
        limits = {}; limits.max_metadata_entries = 0;
        rejects([&] { (void)easyapi::GgufCpuWeights::load(file.string(), limits); });
        std::filesystem::resize_file(file, required_file - 1);
        rejects([&] { (void)easyapi::GgufCpuWeights::load(file.string()); });
        write_fixture(file);
        { std::fstream corrupt(file, std::ios::binary | std::ios::in | std::ios::out); corrupt.write("NOPE", 4); }
        rejects([&] { (void)easyapi::GgufCpuWeights::load(file.string()); });
        rejects([&] { (void)edcpp::api::weights::load(edcpp::api::Backend::cuda, "not-opened", {}); });
        std::cout << "GGUF CPU weights: exact/partial readback, 16 lifecycles, 256 concurrent reads and rejection gates passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
