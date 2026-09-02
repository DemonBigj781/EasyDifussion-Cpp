#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace easyapi {

enum class ModelStorage : std::uint8_t {
    none = 0,
    cpu,
    cuda,
};

struct ModelResource {
    ModelStorage storage = ModelStorage::none;
    void* native_handle = nullptr;
    std::size_t size = 0;
    int device_index = -1;

    bool loaded() const noexcept {
        return storage != ModelStorage::none &&
               native_handle != nullptr &&
               size != 0;
    }
};

struct LoadModelResult {
    bool success = false;
    ModelResource resource;
    std::string diagnostic;
};

struct UnloadModelResult {
    bool success = false;
    std::string diagnostic;
};

class ModelLifecycleHandler {
public:
    virtual ~ModelLifecycleHandler() = default;
    virtual const char* name() const noexcept = 0;
    virtual LoadModelResult load(
        const void* data, std::size_t size, int device_index = 0) const = 0;
    virtual UnloadModelResult unload(ModelResource& resource) const = 0;
};

class CpuModelLifecycleHandler final : public ModelLifecycleHandler {
public:
    const char* name() const noexcept override;
    LoadModelResult load(
        const void* data, std::size_t size, int device_index = 0) const override;
    UnloadModelResult unload(ModelResource& resource) const override;
};

class CudaModelLifecycleHandler final : public ModelLifecycleHandler {
public:
    const char* name() const noexcept override;
    LoadModelResult load(
        const void* data, std::size_t size, int device_index = 0) const override;
    UnloadModelResult unload(ModelResource& resource) const override;
};

} // namespace easyapi
