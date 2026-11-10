#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace easyapi {
struct GgufLimits {
    std::size_t max_file_bytes = 256u * 1024 * 1024;
    std::size_t max_backend_bytes = 256u * 1024 * 1024;
    std::uint64_t max_tensors = 4096;
    std::uint64_t max_metadata_entries = 65536;
};
struct TensorInfo {
    std::string name;
    std::string type;
    std::vector<std::int64_t> shape;
    std::size_t bytes = 0;
};

// Owns a GGUF container's CPU tensors, not an inference model or graph.
// No GPU/offload or model-family interpretation. Independent immutable reads
// are allowed; callers must not race a read against move/reset/destruction.
class GgufCpuWeights final {
public:
    GgufCpuWeights() noexcept;
    ~GgufCpuWeights() noexcept;
    GgufCpuWeights(const GgufCpuWeights&) = delete;
    GgufCpuWeights& operator=(const GgufCpuWeights&) = delete;
    GgufCpuWeights(GgufCpuWeights&&) noexcept;
    GgufCpuWeights& operator=(GgufCpuWeights&&) noexcept;
    static GgufCpuWeights load(const std::string& filename, GgufLimits limits = {});
    bool empty() const noexcept;
    std::size_t tensor_count() const noexcept;
    std::size_t backend_bytes() const noexcept;
    TensorInfo tensor_info(std::size_t index) const;
    std::vector<std::byte> read_tensor(const std::string& name,
                                      std::size_t offset, std::size_t count) const;
    void reset() noexcept;
private:
    struct State;
    std::unique_ptr<State> state_;
};
}
