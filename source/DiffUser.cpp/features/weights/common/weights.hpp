#pragma once
#include "api/gguf_cpu_weights.hpp"
#include "common/api.hpp"
#include <memory>

namespace edcpp::api::weights {
// Normalized, owned tensor-container contract. No backend pointer escapes.
class Resource {
public:
    virtual ~Resource() = default;
    virtual std::size_t tensor_count() const noexcept = 0;
    virtual std::size_t backend_bytes() const noexcept = 0;
    virtual easyapi::TensorInfo tensor_info(std::size_t index) const = 0;
    virtual std::vector<std::byte> read_tensor(const std::string& name,
                                              std::size_t offset, std::size_t count) const = 0;
};
std::unique_ptr<Resource> load(Backend backend, const std::string& filename,
                               const easyapi::GgufLimits& limits);
}
