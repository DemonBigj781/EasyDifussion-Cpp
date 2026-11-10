#include "api/gguf_cpu_weights.hpp"
#include "features/weights/common/weights.hpp"
#include <stdexcept>
#include <utility>

namespace easyapi {
struct GgufCpuWeights::State { std::unique_ptr<edcpp::api::weights::Resource> resource; };
GgufCpuWeights::GgufCpuWeights() noexcept = default;
GgufCpuWeights::~GgufCpuWeights() noexcept = default;
GgufCpuWeights::GgufCpuWeights(GgufCpuWeights&&) noexcept = default;
GgufCpuWeights& GgufCpuWeights::operator=(GgufCpuWeights&& other) noexcept {
    if (this != &other) state_ = std::move(other.state_);
    return *this;
}
GgufCpuWeights GgufCpuWeights::load(const std::string& filename, GgufLimits limits) {
    auto state = std::make_unique<State>();
    state->resource = edcpp::api::weights::load(edcpp::api::Backend::cpu, filename, limits);
    GgufCpuWeights result;
    result.state_ = std::move(state);
    return result;
}
bool GgufCpuWeights::empty() const noexcept { return !state_; }
std::size_t GgufCpuWeights::tensor_count() const noexcept { return state_ ? state_->resource->tensor_count() : 0; }
std::size_t GgufCpuWeights::backend_bytes() const noexcept { return state_ ? state_->resource->backend_bytes() : 0; }
TensorInfo GgufCpuWeights::tensor_info(std::size_t index) const {
    if (!state_) throw std::logic_error("GGUF weight resource is empty");
    return state_->resource->tensor_info(index);
}
std::vector<std::byte> GgufCpuWeights::read_tensor(const std::string& name,
                                                std::size_t offset, std::size_t count) const {
    if (!state_) throw std::logic_error("GGUF weight resource is empty");
    return state_->resource->read_tensor(name, offset, count);
}
void GgufCpuWeights::reset() noexcept { state_.reset(); }
}
