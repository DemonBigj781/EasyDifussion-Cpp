#include "features/weights/common/weights.hpp"
#include <stdexcept>
namespace edcpp::api::weights::cpu::translation {
std::unique_ptr<Resource> load(const std::string&, const easyapi::GgufLimits&);
}
namespace edcpp::api::weights {
std::unique_ptr<Resource> load(Backend backend, const std::string& filename,
                               const easyapi::GgufLimits& limits) {
    if (backend != Backend::cpu)
        throw std::invalid_argument("GGUF tensor resources support only the registered CPU route");
    return cpu::translation::load(filename, limits);
}
}
