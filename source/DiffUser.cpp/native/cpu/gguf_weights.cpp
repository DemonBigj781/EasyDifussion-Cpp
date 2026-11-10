#include "native/cpu/gguf_weights.hpp"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace edcpp::native::cpu {
struct Weights::State {
    gguf_context* metadata = nullptr;
    ggml_context* tensors = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<TensorInfo> info;
    std::vector<ggml_tensor*> handles;
    std::unordered_map<std::string, std::size_t> by_name;
    std::size_t allocated = 0;
    ~State() {
        // Tensor handles never outlive their buffer/context; backend is linked
        // for process lifetime rather than loaded/unloaded by this object.
        ggml_backend_buffer_free(buffer);
        if (tensors) ggml_free(tensors);
        if (metadata) gguf_free(metadata);
    }
};
namespace {
std::uint64_t little_endian(const unsigned char* bytes, unsigned count) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < count; ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
    return value;
}
std::vector<unsigned char> read_file(const std::string& name, const Limits& limits) {
    std::ifstream file(name, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open GGUF file");
    auto end = file.tellg();
    if (end < 24 || static_cast<std::uint64_t>(end) > limits.max_file_bytes ||
        static_cast<std::uint64_t>(end) > std::uint64_t(std::numeric_limits<std::streamsize>::max()))
        throw std::invalid_argument("GGUF file is truncated or exceeds its file budget");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("Could not read the complete GGUF file snapshot");
    if (std::memcmp(bytes.data(), "GGUF", 4) != 0)
        throw std::invalid_argument("Expected a GGUF tensor container");
    const auto version = little_endian(bytes.data() + 4, 4);
    const auto tensors = little_endian(bytes.data() + 8, 8);
    const auto keys = little_endian(bytes.data() + 16, 8);
    if ((version != 2 && version != 3) || tensors == 0 ||
        tensors > limits.max_tensors || keys > limits.max_metadata_entries)
        throw std::invalid_argument("Unsupported GGUF version or metadata-count budget exceeded");
    return bytes;
}
}
Weights::Weights() noexcept = default;
Weights::~Weights() noexcept = default;
Weights::Weights(Weights&&) noexcept = default;
Weights& Weights::operator=(Weights&& other) noexcept {
    if (this != &other) state_ = std::move(other.state_);
    return *this;
}
Weights Weights::load(const std::string& filename, Limits limits) {
    auto bytes = read_file(filename, limits);
    auto state = std::make_unique<State>();
    gguf_init_params params{true, &state->tensors};
    state->metadata = gguf_init_from_buffer(bytes.data(), bytes.size(), params);
    if (!state->metadata || !state->tensors)
        throw std::invalid_argument("GGUF metadata could not be parsed");
    const auto count = gguf_get_n_tensors(state->metadata);
    if (count <= 0 || static_cast<std::uint64_t>(count) > limits.max_tensors)
        throw std::invalid_argument("GGUF tensor count exceeds its budget");
    const auto data_start = gguf_get_data_offset(state->metadata);
    if (data_start > bytes.size()) throw std::invalid_argument("GGUF tensor data is truncated");
    std::vector<std::size_t> offsets;
    std::size_t total = 0;
    for (std::int64_t i = 0; i < count; ++i) {
        const std::string name = gguf_get_tensor_name(state->metadata, i);
        auto* tensor = ggml_get_tensor(state->tensors, name.c_str());
        if (!tensor || !state->by_name.emplace(name, state->handles.size()).second)
            throw std::invalid_argument("GGUF has missing or duplicate tensor names");
        const auto size = ggml_nbytes(tensor);
        const auto offset = gguf_get_tensor_offset(state->metadata, i);
        if (offset > bytes.size() - data_start || size > bytes.size() - data_start - offset ||
            size == 0 || size > limits.max_backend_bytes - total)
            throw std::invalid_argument("GGUF tensor payload is truncated or over budget");
        total += size;
        TensorInfo info{name, ggml_type_name(tensor->type), {}, size};
        for (int dimension = 0; dimension < ggml_n_dims(tensor); ++dimension)
            info.shape.push_back(tensor->ne[dimension]);
        state->info.push_back(std::move(info));
        state->handles.push_back(tensor);
        offsets.push_back(data_start + offset);
    }
    auto* cpu = ggml_backend_cpu_buffer_type();
    const auto required = ggml_backend_alloc_ctx_tensors_from_buft_size(state->tensors, cpu);
    if (required == 0 || required > limits.max_backend_bytes)
        throw std::invalid_argument("Aligned GGML CPU allocation exceeds its backend budget");
    state->buffer = ggml_backend_alloc_ctx_tensors_from_buft(state->tensors, cpu);
    if (!state->buffer) throw std::bad_alloc();
    state->allocated = ggml_backend_buffer_get_size(state->buffer);
    if (state->allocated > limits.max_backend_bytes)
        throw std::invalid_argument("GGML CPU buffer exceeds its backend budget");
    for (std::size_t i = 0; i < state->handles.size(); ++i)
        ggml_backend_tensor_set(state->handles[i], bytes.data() + offsets[i], 0, state->info[i].bytes);
    Weights owned;
    owned.state_ = std::move(state);
    return owned;
}
bool Weights::empty() const noexcept { return !state_; }
std::size_t Weights::tensor_count() const noexcept { return state_ ? state_->info.size() : 0; }
std::size_t Weights::backend_bytes() const noexcept { return state_ ? state_->allocated : 0; }
TensorInfo Weights::tensor_info(std::size_t index) const {
    if (!state_ || index >= state_->info.size()) throw std::out_of_range("GGUF tensor index is out of range");
    return state_->info[index];
}
std::vector<std::byte> Weights::read_tensor(const std::string& name,
                                                std::size_t offset, std::size_t count) const {
    if (!state_) throw std::logic_error("GGUF weight resource is empty");
    auto found = state_->by_name.find(name);
    if (found == state_->by_name.end()) throw std::out_of_range("Unknown GGUF tensor");
    const auto index = found->second, size = state_->info[index].bytes;
    if (offset > size || count > size - offset) throw std::out_of_range("GGUF tensor read is out of bounds");
    std::vector<std::byte> output(count);
    if (count) ggml_backend_tensor_get(state_->handles[index], output.data(), offset, count);
    return output;
}
void Weights::reset() noexcept { state_.reset(); }
}
