#include "features/weights/common/weights.hpp"
#include "native/cpu/gguf_weights.hpp"
#include <utility>

namespace edcpp::api::weights::cpu::translation {
namespace {
class CpuResource final : public Resource {
public:
    CpuResource(const std::string& filename, const easyapi::GgufLimits& limits)
        : weights_(edcpp::native::cpu::Weights::load(filename,
            {limits.max_file_bytes, limits.max_backend_bytes,
             limits.max_tensors, limits.max_metadata_entries})) {}
    std::size_t tensor_count() const noexcept override { return weights_.tensor_count(); }
    std::size_t backend_bytes() const noexcept override { return weights_.backend_bytes(); }
    easyapi::TensorInfo tensor_info(std::size_t index) const override {
        auto native = weights_.tensor_info(index);
        return {std::move(native.name), std::move(native.type), std::move(native.shape), native.bytes};
    }
    std::vector<std::byte> read_tensor(const std::string& name, std::size_t offset,
                                      std::size_t count) const override {
        return weights_.read_tensor(name, offset, count);
    }
private:
    edcpp::native::cpu::Weights weights_;
};
}
std::unique_ptr<Resource> load(const std::string& filename, const easyapi::GgufLimits& limits) {
    return std::make_unique<CpuResource>(filename, limits);
}
}
