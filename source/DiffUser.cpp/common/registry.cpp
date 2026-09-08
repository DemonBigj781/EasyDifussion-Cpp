#include "registry.hpp"

namespace edcpp::api {

bool Registry::register_backend(const BackendInterface& backend) noexcept {
    if (backend.backend == Backend::none || backend.dispatch == nullptr) {
        return false;
    }

    for (std::size_t i = 0; i < size_; ++i) {
        if (backends_[i].backend == backend.backend) {
            backends_[i] = backend;
            return true;
        }
    }

    if (size_ >= max_backends) {
        return false;
    }

    backends_[size_++] = backend;
    return true;
}

const BackendInterface* Registry::find(Backend backend) const noexcept {
    for (std::size_t i = 0; i < size_; ++i) {
        if (backends_[i].backend == backend) {
            return &backends_[i];
        }
    }
    return nullptr;
}

bool Registry::supports(Backend backend, Capability capability) const noexcept {
    const auto* entry = find(backend);
    return entry != nullptr && has_capability(entry->capabilities, capability);
}

bool Registry::dispatch(Backend backend, DispatchContext& context, Operation operation, void* payload) const noexcept {
    const auto* entry = find(backend);
    if (entry == nullptr || entry->dispatch == nullptr) {
        return false;
    }
    return entry->dispatch(context, operation, payload);
}

Registry& registry() noexcept {
    static Registry instance;
    return instance;
}

} // namespace edcpp::api
