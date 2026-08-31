#pragma once

#include "api.hpp"

#include <array>
#include <cstddef>

namespace edcpp::api {

class Registry {
public:
    static constexpr std::size_t max_backends = 16;

    bool register_backend(const BackendInterface& backend) noexcept;
    const BackendInterface* find(Backend backend) const noexcept;
    bool supports(Backend backend, Capability capability) const noexcept;
    bool dispatch(Backend backend, DispatchContext& context, Operation operation, void* payload) const noexcept;
    std::size_t size() const noexcept { return size_; }

private:
    std::array<BackendInterface, max_backends> backends_{};
    std::size_t size_ = 0;
};

Registry& registry() noexcept;

} // namespace edcpp::api
