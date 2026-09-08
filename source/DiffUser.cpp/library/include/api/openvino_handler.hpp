#pragma once
#include "backend.hpp"
namespace easyapi {
class OpenvinoHandler final : public BackendHandler {
public:
    const char* name() const noexcept override;
    bool available() const noexcept override;
    std::vector<DeviceInfo> devices() const override;
};
} // namespace easyapi
